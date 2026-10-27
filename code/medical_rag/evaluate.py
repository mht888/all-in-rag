"""
医学RAG系统 - 检索策略对比评估脚本
================================================
对比三种检索策略 (hybrid / colbert / embedding_route) 的检索与回答质量。

检索级评估（零 LLM 成本）:
    - 匹配口径: chunk 级内容哈希匹配（chunk_id 是 uuid4 每次运行会变，故用
      page_content 的 MD5 作为稳定匹配键；问题集与分块参数绑定）
    - Hit Rate@top_k: 检索结果中命中出题源块的比例
    - MRR@top_k:     每题最早命中排名倒数均值（miss 计 0）
    - 每道 miss 题打印 期望教材 vs 实际检索到的教材列表（区分真 miss / 块级
      miss / 跨教材 miss）

响应级评估（LLM-as-a-Judge）:
    - 忠实度: 评审"回答 + 检索上下文"，检测幻觉（回答是否被资料支撑）
    - 相关性: 评审"回答 + 问题"，不给上下文——避免评审员看到上下文后给出
      "基于资料而言相关"的宽容判定，掩盖检索策略差异

公平性设计:
    - 三策略共用同一份合并索引 / 同一批问题 / 同一生成与评审 LLM
    - 不走 query_rewrite 与学科关键词过滤（保证纯检索能力对比）
    - 生成上下文直接使用检索子块（父文档 page_content 为空，见 setup 注释）
    - route_info 固定为 general 模板（省去每题的路由 LLM 调用，三策略一致）

产物:
    - eval_questions.json: 评估问题集缓存（首跑生成后打印全部题目，之后直接加载）
    - eval_results.json:   每题每策略完整明细 + 汇总指标

运行前提: 已运行 build_index.py 且各学科索引齐全
运行方式: python evaluate.py
"""

import os
import sys
import json
import time
import random
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import Counter

# 修复Windows终端编码问题
if sys.platform == 'win32':
    os.system('chcp 65001 >nul 2>&1')

# 添加模块路径
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
from config import DEFAULT_CONFIG
from rag_modules import (
    DataPreparationModule,
    IndexConstructionModule,
    HybridSearchModule,
    GenerationIntegrationModule,
)
from rag_modules.colbert_rerank import ColbertRerankModule
from rag_modules.embedding_routing import EmbeddingRoutingModule

# 加载环境变量
load_dotenv()

# 评估脚本以 print 为主输出，压制模块内部 INFO 日志噪音
logging.basicConfig(level=logging.WARNING)

# ------------------------------------------------------------------
# 常量配置
# ------------------------------------------------------------------
EVAL_DIR = Path(__file__).parent
QUESTIONS_PATH = EVAL_DIR / "eval_questions.json"
RESULTS_PATH = EVAL_DIR / "eval_results.json"

STRATEGIES = ["hybrid", "colbert", "embedding_route"]
QUESTIONS_PER_SUBJECT = 2   # 每学科出题数（4 学科 = 8 题）
BACKUP_PER_SUBJECT = 3      # 每学科额外备用采样数（出题失败时替补）
RANDOM_SEED = 42            # 固定采样种子，保证问题集可复现
MIN_CHUNK_LEN = 200         # 太短的子块出不了高质量题目
TOP_K_DISPLAY = 3           # 汇总表显示的 Hit Rate 口径（实际用 config.top_k）

# 固定路由信息：三策略统一走 general 模板（见模块 docstring 公平性设计）
FIXED_ROUTE_INFO = {"route": "general", "urgency": "normal", "expertise": "patient"}


def chunk_hash(text: str) -> str:
    """
    子块内容哈希（检索级匹配键）

    注意: chunk_id 是 uuid4 生成的，跨次运行会变化，不能作为匹配键；
    必须用 page_content 的 MD5，且问题集与分块参数绑定（见 prepare_questions）。
    """
    return hashlib.md5(text.strip().encode("utf-8")).hexdigest()


# ------------------------------------------------------------------
# LLM Prompt 模板（JSON 示例中的花括号在拼接时使用，不走 format）
# ------------------------------------------------------------------
QUESTION_GEN_PROMPT = """你是一个医学RAG评估数据集的构建助手。请基于以下医学教材内容出一道评估题目。

要求:
1. 问题必须能从给定内容中找到答案（答案锚定在教材原文，而不是你的知识）
2. 问题要具体、明确，适合检验检索系统（例如"XX疾病的诊断标准是什么"）
3. 不要出急救/紧急类问题
4. 判断该知识点是否也常见于其他医学学科的教材（例如某药物的作用机制在药理学和内科学中都可能出现）

教材内容:
"""

QUESTION_GEN_SUFFIX = """

请输出 JSON（不要输出其他内容）:
{"question": "题目", "reference": "1-2句参考答案要点", "cross_subject": true或false}"""

FAITHFULNESS_PROMPT = """你是医学RAG系统的质量评审员。请判断下面的回答是否忠实于给定的参考资料。

判定标准:
- 回答中的医学论断必须能在参考资料中找到依据；回答可以合理概括、组织资料，但不得编造资料中没有的实质性结论
- 回答遗漏了资料中的细节不算不忠实
- 只有当回答包含参考资料无法支撑的实质性医学论断时，才判 faithful=false

参考资料:
{context}

回答:
{answer}

请输出 JSON（不要输出其他内容）:
{{"faithful": true或false, "reason": "一句话理由"}}"""

RELEVANCY_PROMPT = """你是医学RAG系统的质量评审员。请判断下面的回答是否切实回答了用户的医学问题。

判定标准:
- 回答回应了问题所问的核心即算相关 (relevant=true)
- 答非所问、只回答了问题的次要方面、或回避问题实质，判 relevant=false
- 不要求回答完全正确，只判断是否切题

用户问题:
{question}

回答:
{answer}

请输出 JSON（不要输出其他内容）:
{{"relevant": true或false, "reason": "一句话理由"}}"""


class MedicalRAGEvaluator:
    """医学RAG三策略对比评估器"""

    def __init__(self, config=DEFAULT_CONFIG):
        self.config = config
        self.data_module: Optional[DataPreparationModule] = None
        self.index_module: Optional[IndexConstructionModule] = None
        self.generation_module: Optional[GenerationIntegrationModule] = None
        self.retrieval_modules: Dict[str, Any] = {}
        self.all_chunks: List = []

    # ==================================================================
    # 第 0 步: 初始化（加载合并索引 + 分块 + 三策略模块 + 生成模块）
    # ==================================================================
    def setup(self):
        print("=" * 60)
        print("🔨 医学RAG检索策略对比评估")
        print("=" * 60)

        # 1. 数据模块（chunk_documents 依赖 documents 列表，必须先 load）
        self.data_module = DataPreparationModule(self.config.data_path)
        self.data_module.load_documents()

        # 2. 索引模块：按学科加载已构建索引并合并（只加载不构建，缺失即报错）
        self.index_module = IndexConstructionModule(
            model_name=self.config.embedding_model,
            index_save_path=self.config.index_save_path,
            device=self.config.device
        )

        folders = sorted({doc.metadata.get('folder_name')
                          for doc in self.data_module.documents})
        print(f"\n📦 加载学科索引: {' / '.join(folders)}")

        vectorstores = []
        for folder in folders:
            if not self.index_module.index_exists(folder):
                raise FileNotFoundError(
                    f"学科索引不存在或无效: {self.config.index_save_path}/{folder}\n"
                    f"请先运行 python build_index.py 构建索引"
                )
            vectorstores.append(self.index_module.load_index(folder))

        merged_vectorstore = vectorstores[0]
        for vs in vectorstores[1:]:
            merged_vectorstore.merge_from(vs)
        print("🔗 索引合并完成")

        # 3. 分块（检索级评估需要 all_chunks 做采样出题池；
        #    hybrid/embedding_route 模块也依赖 chunks 列表）
        print("📄 解析文档并分块（耗时较长，请耐心等待）...")
        self.all_chunks = self.data_module.chunk_documents(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        print(f"   ✅ 分块完成: {len(self.all_chunks)} 个子块")

        # 4. 生成模块（评估全程复用其 llm 实例做出题与评审）
        print("🤖 初始化生成模块 (LLM 兼任出题与评审)...")
        self.generation_module = GenerationIntegrationModule(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

        # 5. 三种检索模块（共享同一份合并索引与 chunks）
        print("🔍 初始化三个检索模块...")
        self.retrieval_modules = {
            "hybrid": HybridSearchModule(
                merged_vectorstore,
                self.all_chunks,
                top_k=self.config.top_k,
                rrf_k=self.config.rrf_k
            ),
            "colbert": ColbertRerankModule(
                merged_vectorstore,
                model_name=self.config.reranker_model,
                initial_top_k=self.config.colbert_initial_top_k,
                final_top_k=self.config.top_k
            ),
            "embedding_route": EmbeddingRoutingModule(
                merged_vectorstore,
                self.all_chunks,
                embedding_model=self.config.embedding_model,
                top_k=self.config.top_k
            ),
        }
        print("✅ 初始化完成！")

    # ==================================================================
    # 第 1 步: 评估问题集（缓存优先，首跑生成后打印全部题目，不中断）
    # ==================================================================
    def prepare_questions(self) -> List[Dict[str, Any]]:
        print("\n" + "=" * 60)
        print("📝 评估问题集")
        print("=" * 60)

        current_chunk_config = {
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
        }

        if QUESTIONS_PATH.exists():
            with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 分块参数与生成时不一致 → 所有内容哈希失效，必须重新生成
            if data.get("chunk_config") != current_chunk_config:
                raise ValueError(
                    f"分块参数({current_chunk_config})与问题集生成时"
                    f"({data.get('chunk_config')})不一致，内容哈希已失效。\n"
                    f"请删除 {QUESTIONS_PATH.name} 后重新运行以重新生成问题集"
                )

            questions = data["questions"]
            print(f"加载缓存问题集: {QUESTIONS_PATH.name} ({len(questions)} 题)")
        else:
            print(f"生成评估问题集（每学科 {QUESTIONS_PER_SUBJECT} 题）...")
            questions = self._generate_questions()

            with open(QUESTIONS_PATH, "w", encoding="utf-8") as f:
                json.dump({
                    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "chunk_config": current_chunk_config,
                    "questions": questions,
                }, f, ensure_ascii=False, indent=2)
            print(f"✅ 已保存缓存: {QUESTIONS_PATH.name}")

        # 打印全部题目及来源 chunk 信息（仅供终端查看，不中断）
        print(f"\n📋 评估问题集（{len(questions)} 题）:")
        for i, q in enumerate(questions, 1):
            print(f"   Q{i:02d} [{q['category']}] {q['question']}")
            print(f"       源: 《{q['source_doc']}》 | chunk {q['source_chunk_hash'][:8]}"
                  f" | 跨学科: {'是' if q['cross_subject'] else '否'}")
            print(f"       预览: {q['chunk_preview']}...")

        return questions

    def _generate_questions(self) -> List[Dict[str, Any]]:
        """按学科采样子块 → LLM 逐题生成 → 失败替补"""
        random.seed(RANDOM_SEED)

        # 按学科分组出题池（过滤过短子块）
        pools: Dict[str, List] = {}
        for c in self.all_chunks:
            if len(c.page_content) < MIN_CHUNK_LEN:
                continue
            folder = c.metadata.get('folder_name', 'unknown')
            pools.setdefault(folder, []).append(c)

        if not pools:
            raise ValueError("没有符合条件的出题子块，请检查文档分块结果")

        questions = []
        for folder, pool in pools.items():
            category = pool[0].metadata.get('category', '未知')
            random.shuffle(pool)
            picked = pool[:QUESTIONS_PER_SUBJECT + BACKUP_PER_SUBJECT]

            got = 0
            for chunk in picked:
                if got >= QUESTIONS_PER_SUBJECT:
                    break
                q = self._generate_one_question(chunk)
                if q:
                    questions.append(q)
                    got += 1
            print(f"   [{category}] 出题 {got}/{QUESTIONS_PER_SUBJECT} ✅")

        if not questions:
            raise ValueError("问题集生成失败，请检查 LLM 配置与网络")
        return questions

    def _generate_one_question(self, chunk) -> Optional[Dict[str, Any]]:
        """基于单个子块出题（含 1 次重试）"""
        content = chunk.page_content
        prompt = QUESTION_GEN_PROMPT + content + QUESTION_GEN_SUFFIX

        for _ in range(2):
            parsed = self._invoke_llm_json(prompt)
            question = str(parsed.get("question", "")).strip()
            if question and len(question) >= 5:
                return {
                    "question": question,
                    "reference": str(parsed.get("reference", "")).strip(),
                    "cross_subject": bool(parsed.get("cross_subject", False)),
                    "source_doc": chunk.metadata.get('doc_name', '未知文档'),
                    "category": chunk.metadata.get('category', '未知'),
                    "source_chunk_hash": chunk_hash(content),
                    "chunk_preview": content[:80].replace('\n', ' '),
                }
        return None

    # ==================================================================
    # 第 2 步: 检索级评估（chunk 级哈希匹配 → Hit Rate / MRR）
    # ==================================================================
    def run_retrieval_eval(self, questions: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """
        每题 × 3 策略执行核心检索方法并做 chunk 级匹配。

        Returns:
            {strategy: [ {qid, hit, rank, rr, chunks, retrieved_docs} × 每题 ]}
            chunks 会缓存在条目中，响应级评估直接复用（避免重复检索）
        """
        print(f"\n{'=' * 60}")
        print(f"🔍 检索级评估（chunk 级匹配，top_k={self.config.top_k}，零 LLM 成本）")
        print(f"{'=' * 60}")

        results: Dict[str, List[Dict]] = {s: [] for s in STRATEGIES}

        for qi, q in enumerate(questions, 1):
            qid = f"Q{qi:02d}"
            print(f"\n[{qid}] {q['question']}")

            for strategy in STRATEGIES:
                chunks = self._retrieve(strategy, q["question"])

                # chunk 级匹配：检索结果逐块哈希比对出题源块
                hit_rank = None
                for rank, c in enumerate(chunks, 1):
                    if chunk_hash(c.page_content) == q["source_chunk_hash"]:
                        hit_rank = rank
                        break

                retrieved_docs = Counter(
                    c.metadata.get('doc_name', '未知') for c in chunks
                )

                entry = {
                    "qid": qid,
                    "hit": hit_rank is not None,
                    "rank": hit_rank,
                    "rr": (1.0 / hit_rank) if hit_rank else 0.0,
                    "chunks": chunks,
                    "retrieved_docs": dict(retrieved_docs),
                }
                results[strategy].append(entry)

                if hit_rank is not None:
                    print(f"   {strategy:<16} 命中 (rank {hit_rank})")
                else:
                    # miss 明细：期望教材 vs 实际检索到的教材
                    # （一眼区分: 真miss=期望教材完全没出现 / 块级miss=教材对了但
                    #   没召回出题块 / 跨教材miss=召回的都是他教材相关内容）
                    docs_str = ", ".join(
                        f"《{name}》×{n}" for name, n in retrieved_docs.items()
                    ) if retrieved_docs else "（空）"
                    print(f"   {strategy:<16} miss")
                    print(f"      ⚠ 期望教材: 《{q['source_doc']}》"
                          f" chunk {q['source_chunk_hash'][:8]}")
                    print(f"        实际检索到: {docs_str}")

        return results

    def _retrieve(self, strategy: str, query: str) -> List:
        """调用各策略的核心检索方法（与 main.py ask_question 同口径，无过滤）"""
        module = self.retrieval_modules[strategy]
        if strategy == "hybrid":
            return module.hybrid_search(query)
        if strategy == "colbert":
            chunks, _ = module.retrieve_and_rerank(query)
            return chunks
        return module.search(query)  # embedding_route

    # ==================================================================
    # 第 3 步: 响应级评估（生成回答 → 忠实度/相关性双评审）
    # ==================================================================
    def run_response_eval(self, questions: List[Dict[str, Any]],
                          retrieval_results: Dict[str, List[Dict]]):
        print(f"\n{'=' * 60}")
        print("✍️ 响应级评估（LLM-as-a-Judge）")
        print(f"{'=' * 60}")

        for qi, q in enumerate(questions, 1):
            print(f"\n[Q{qi:02d}] {q['question'][:40]}")

            for strategy in STRATEGIES:
                entry = retrieval_results[strategy][qi - 1]
                chunks = entry["chunks"]

                # 检索为空 → 不调生成，忠实/相关双 fail（省 3 次 LLM 调用）
                if not chunks:
                    entry.update(answer="", faithful=False, relevant=False,
                                 faithful_reason="检索为空", relevant_reason="检索为空")
                    print(f"   {strategy:<16} 检索为空，双 fail")
                    continue

                # 生成回答：检索子块直接作为上下文
                # （父文档 page_content 为空，不可作为生成上下文；子块 metadata
                #   含 doc_name/category，_build_context 可正常构建）
                try:
                    answer = self.generation_module.generate_answer(
                        q["question"], chunks,
                        route_info=dict(FIXED_ROUTE_INFO),
                        stream=False
                    )
                except Exception as e:
                    entry.update(answer="", faithful=False, relevant=False,
                                 faithful_reason=f"生成失败: {e}",
                                 relevant_reason="生成失败")
                    print(f"   {strategy:<16} 生成异常: {e}")
                    continue

                # 忠实度评审：复用生成时的同一份上下文构建，保证口径一致
                context = self.generation_module._build_context(chunks)
                f_result = self._judge(
                    FAITHFULNESS_PROMPT.format(context=context, answer=answer),
                    key="faithful"
                )
                entry["answer"] = answer
                entry["faithful"] = bool(f_result.get("faithful", False))
                entry["faithful_reason"] = str(f_result.get("reason", ""))

                # 相关性评审：只看 问题 + 回答（不给上下文，见模块 docstring）
                r_result = self._judge(
                    RELEVANCY_PROMPT.format(question=q["question"], answer=answer),
                    key="relevant"
                )
                entry["relevant"] = bool(r_result.get("relevant", False))
                entry["relevant_reason"] = str(r_result.get("reason", ""))

                f_mark = "✅" if entry["faithful"] else "❌"
                r_mark = "✅" if entry["relevant"] else "❌"
                print(f"   {strategy:<16} 忠实 {f_mark}  相关 {r_mark}")

    def _judge(self, prompt: str, key: str) -> Dict[str, Any]:
        """执行一次评审（解析失败重试 1 次，仍失败记保守值并注明）"""
        for _ in range(2):
            parsed = self._invoke_llm_json(prompt)
            if key in parsed:
                return parsed
        return {key: False, "reason": "评审输出解析失败（保守计 fail）"}

    # ==================================================================
    # 第 4 步: 汇总对比 + 保存明细
    # ==================================================================
    def summarize_and_save(self, questions: List[Dict[str, Any]],
                           retrieval_results: Dict[str, List[Dict]]):
        n = len(questions)
        summary = {}
        for strategy in STRATEGIES:
            entries = retrieval_results[strategy]
            summary[strategy] = {
                "hit_rate": sum(1 for e in entries if e["hit"]) / n,
                "mrr": sum(e["rr"] for e in entries) / n,
                "faithfulness": sum(1 for e in entries if e.get("faithful")) / n,
                "relevancy": sum(1 for e in entries if e.get("relevant")) / n,
            }

        print(f"\n{'=' * 60}")
        print(f"🏁 最终对比（{n} 题，top_k={self.config.top_k}）")
        print(f"{'=' * 60}")
        print(f"{'策略':<18}{'Hit Rate':<12}{'MRR':<10}{'忠实度':<10}{'相关性':<8}")
        print("-" * 58)
        for strategy in STRATEGIES:
            s = summary[strategy]
            print(f"{strategy:<18}{s['hit_rate']:<12.1%}{s['mrr']:<10.2f}"
                  f"{s['faithfulness']:<10.1%}{s['relevancy']:<8.1%}")

        # 跨学科题单独统计（解读绝对值时留意跨教材误伤）
        cross_ids = {i for i, q in enumerate(questions) if q.get("cross_subject")}
        if cross_ids:
            print(f"\n⚠ 其中跨学科题 {len(cross_ids)} 题（知识点可能出现在多本教材，"
                  f"Hit Rate 绝对值或被低估，详见明细）")

        # 保存完整明细
        details = []
        for qi, q in enumerate(questions, 1):
            per_strategy = {}
            for strategy in STRATEGIES:
                e = retrieval_results[strategy][qi - 1]
                chunks = e["chunks"]
                per_strategy[strategy] = {
                    "hit": e["hit"],
                    "rank": e["rank"],
                    "retrieved_docs": e["retrieved_docs"],
                    "retrieved_previews": [
                        c.page_content[:100].replace('\n', ' ') for c in chunks
                    ],
                    "answer": e.get("answer", ""),
                    "faithful": e.get("faithful"),
                    "faithful_reason": e.get("faithful_reason", ""),
                    "relevant": e.get("relevant"),
                    "relevant_reason": e.get("relevant_reason", ""),
                }
            details.append({
                "qid": f"Q{qi:02d}",
                "question": q["question"],
                "category": q["category"],
                "source_doc": q["source_doc"],
                "source_chunk_hash": q["source_chunk_hash"],
                "cross_subject": q.get("cross_subject", False),
                "reference": q.get("reference", ""),
                "results": per_strategy,
            })

        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "llm_model": self.config.llm_model,
                "top_k": self.config.top_k,
                "num_questions": n,
                "summary": summary,
                "details": details,
            }, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 明细已保存: {RESULTS_PATH.name}")

    # ==================================================================
    # 工具方法
    # ==================================================================
    def _invoke_llm_json(self, prompt: str) -> Dict[str, Any]:
        """调用 LLM 并解析 JSON 输出（复用生成模块的 llm 实例）"""
        try:
            response = self.generation_module.llm.invoke(prompt)
            return self._parse_json(response.content)
        except Exception as e:
            logging.warning(f"LLM 调用异常: {e}")
            return {}

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        """
        解析 LLM 输出的 JSON（剥离 markdown 围栏，qwen/glm 常把 JSON 包在 ``` 里）
        """
        text = text.strip()
        if "```" in text:
            for part in text.split("```"):
                part = part.strip()
                if part.lower().startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    try:
                        return json.loads(part)
                    except json.JSONDecodeError:
                        continue
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}


def main():
    evaluator = MedicalRAGEvaluator()
    evaluator.setup()
    questions = evaluator.prepare_questions()
    retrieval_results = evaluator.run_retrieval_eval(questions)
    evaluator.run_response_eval(questions, retrieval_results)
    evaluator.summarize_and_save(questions, retrieval_results)


if __name__ == "__main__":
    main()
