"""
医学RAG系统主程序
流程：LLM查询路由（多维度分析） → 智能查询重写 → 混合检索 → 父子文档映射 → 按路由模板生成回答
"""

import os
import sys
import logging
from pathlib import Path

# 修复Windows终端编码问题
if sys.platform == 'win32':
    os.system('chcp 65001 >nul 2>&1')

# 添加模块路径
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
from config import DEFAULT_CONFIG, RAGConfig
from rag_modules import (
    DataPreparationModule,
    IndexConstructionModule,
    HybridSearchModule,
    GenerationIntegrationModule
)
from rag_modules.colbert_rerank import ColbertRerankModule
from rag_modules.embedding_routing import EmbeddingRoutingModule

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MedicalRAGSystem:
    """医学RAG系统主类"""

    def __init__(self, config: RAGConfig = None):
        """
        初始化RAG系统

        Args:
            config: RAG系统配置，默认使用DEFAULT_CONFIG
        """
        self.config = config or DEFAULT_CONFIG
        self.data_module = None
        self.index_module = None
        self.retrieval_module = None
        self.generation_module = None
        self.merged_vectorstore = None

        # 检查数据路径
        if not Path(self.config.data_path).exists():
            raise FileNotFoundError(f"数据路径不存在: {self.config.data_path}")

        # 检查API密钥
        if not os.getenv("QWEN_API_KEY"):
            raise ValueError("请设置 QWEN_API_KEY 环境变量")

    def initialize_system(self):
        """初始化所有模块"""
        print("🚀 正在初始化医学RAG系统...")

        # 1. 初始化数据准备模块
        print("初始化数据准备模块...")
        self.data_module = DataPreparationModule(self.config.data_path)

        # 2. 初始化索引构建模块
        print("初始化索引构建模块...")
        self.index_module = IndexConstructionModule(
            model_name=self.config.embedding_model,
            index_save_path=self.config.index_save_path,
            device=self.config.device
        )

        # 3. 初始化生成集成模块
        print("🤖 初始化生成集成模块...")
        self.generation_module = GenerationIntegrationModule(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            context_max_length=self.config.context_max_length
        )

        print("✅ 系统初始化完成！")

    def build_knowledge_base(self):
        """构建知识库（按文件夹分类构建索引，已存在且有效则跳过）"""
        print("\n正在构建知识库...")

        # 1. 加载文档
        print("加载医学文档...")
        self.data_module.load_documents()

        # 2. 按文件夹分组文档（直接用英文文件夹名）
        docs_by_folder = {}
        for doc in self.data_module.documents:
            folder_name = doc.metadata.get('folder_name', 'unknown')
            if folder_name not in docs_by_folder:
                docs_by_folder[folder_name] = []
            docs_by_folder[folder_name].append(doc)

        print(f"发现 {len(docs_by_folder)} 个学科文件夹: {list(docs_by_folder.keys())}")

        # 3. 按文件夹分别构建/加载索引
        all_chunks = []
        folder_vectorstores = {}

        for folder_name, docs in docs_by_folder.items():
            category = docs[0].metadata.get('category', '未知') if docs else '未知'
            print(f"\n处理学科: {category} (文件夹: {folder_name})")

            # 检查该学科的索引是否已存在且有效（文件夹存在、不为空、有 faiss+pkl）
            if self.index_module.index_exists(folder_name):
                print(f"  ✅ 索引已存在且有效，跳过构建")
                vectorstore = self.index_module.load_index(folder_name)
                folder_vectorstores[folder_name] = vectorstore
            else:
                print(f"  📝 索引不存在或无效，开始构建...")

                # 为该学科的文档创建临时模块进行分块
                from rag_modules.data_preparation import DataPreparationModule
                temp_module = DataPreparationModule(self.config.data_path)
                temp_module.documents = docs

                # 分块
                chunks = temp_module.chunk_documents(
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap
                )
                all_chunks.extend(chunks)

                # 构建索引
                vectorstore = self.index_module.build_vector_index(chunks)
                folder_vectorstores[folder_name] = vectorstore

                # 保存索引（直接用文件夹名）
                self.index_module.save_index(folder_name)
                print(f"  ✅ 索引已保存到: {self.config.index_save_path}/{folder_name}")

        # 4. 合并所有 chunks（用于检索模块）
        if not all_chunks:
            # 如果所有索引都已存在，仍需加载文档和分块用于检索模块
            print("\n加载文档分块用于检索模块...")
            all_chunks = self.data_module.chunk_documents(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )

        # 4. 合并所有 vectorstores（用于检索模块）
        # 使用第一个 vectorstore 作为基础，合并其他
        if folder_vectorstores:
            first_folder = list(folder_vectorstores.keys())[0]
            merged_vectorstore = folder_vectorstores[first_folder]

            # 如果有多个学科，需要合并索引
            if len(folder_vectorstores) > 1:
                print("\n合并各学科索引...")
                for folder, vs in list(folder_vectorstores.items())[1:]:
                    merged_vectorstore.merge_from(vs)
                print("✅ 索引合并完成")
        else:
            raise ValueError("没有可用的向量索引")

        # 保存合并后的向量存储，供查询时还原父子上下文窗口（docstore 含全部子块）
        self.merged_vectorstore = merged_vectorstore

        # 6. 根据策略初始化检索模块
        strategy = self.config.retrieval_strategy
        print(f"\n初始化检索模块 (策略: {strategy})...")

        if strategy == "hybrid":
            self.retrieval_module = HybridSearchModule(
                merged_vectorstore,
                all_chunks,
                top_k=self.config.top_k,
                rrf_k=self.config.rrf_k
            )
        elif strategy == "colbert":
            self.retrieval_module = ColbertRerankModule(
                merged_vectorstore,
                model_name=self.config.reranker_model,
                initial_top_k=self.config.colbert_initial_top_k,
                final_top_k=self.config.top_k
            )
        elif strategy == "embedding_route":
            self.retrieval_module = EmbeddingRoutingModule(
                merged_vectorstore,
                all_chunks,
                embedding_model=self.config.embedding_model,
                top_k=self.config.top_k,
                device=self.config.device
            )
        else:
            raise ValueError(f"未知的检索策略: {strategy}")

        # 7. 显示统计信息
        stats = self.data_module.get_statistics()
        print(f"\n📊 知识库统计:")
        print(f"   文档总数: {stats['total_documents']}")
        print(f"   文本块数: {stats['total_chunks']}")
        print(f"   学科分布: {list(stats['categories'].keys())}")
        print(f"   平均块大小: {stats['avg_chunk_size']:.0f} 字符")

        print("✅ 知识库构建完成！")

    def ask_question(self, question: str, stream: bool = False):
        """
        回答用户问题

        流程：
        1. LLM查询路由（路由类型/紧急程度/专业级别/医学实体）
        2. 智能查询重写（紧急情况跳过，争取处理时间）
        3. 混合检索（学科关键词命中时使用过滤检索）
        4. 父子文档映射（子块 → 完整文档）
        5. 按路由类型生成回答（支持流式）

        Args:
            question: 用户问题
            stream: 是否使用流式输出

        Returns:
            生成的回答或生成器
        """
        if not all([self.retrieval_module, self.generation_module]):
            raise ValueError("请先构建知识库")

        print(f"\n❓ 用户问题: {question}")

        # 1. LLM查询路由
        print("🎯 正在分析查询...")
        route_info = self.generation_module.query_router(question)
        route = route_info['route']
        urgency = route_info['urgency']
        expertise = route_info['expertise']
        is_emergency = (route == 'emergency' or urgency == 'critical')

        print(f"   路由类型: {route}")
        print(f"   紧急程度: {urgency}")
        print(f"   专业级别: {expertise}")

        # 显示提取到的医学实体
        entities = route_info.get('entities', {})
        entity_summary = {k: v for k, v in entities.items() if v}
        if entity_summary:
            print(f"   医学实体: {entity_summary}")

        # 2. 智能查询重写（紧急情况跳过，直接检索争取时间）
        if is_emergency:
            print("🚨 检测到紧急情况，跳过查询重写直接处理！")
            search_query = question
        else:
            print("🤖 智能分析查询...")
            search_query = self.generation_module.query_rewrite(question)
            if search_query != question:
                print(f"📝 查询已重写: '{search_query}'")

        # 3. 检索相关子块（根据策略和过滤条件选择检索方式）
        print("🔍 检索相关文档...")
        strategy = self.config.retrieval_strategy

        if strategy == "embedding_route":
            # 嵌入路由策略：自带学科路由，忽略外部过滤
            relevant_chunks = self.retrieval_module.search(search_query)
        else:
            # hybrid / colbert 策略：检查是否需要学科过滤
            filters = self._extract_filters_from_query(question)
            if filters:
                print(f"   应用学科过滤: {filters}")
                relevant_chunks = self.retrieval_module.filtered_search(search_query, filters)
            else:
                # 根据策略调用对应的检索方法
                if strategy == "hybrid":
                    relevant_chunks = self.retrieval_module.hybrid_search(search_query)
                elif strategy == "colbert":
                    relevant_chunks, _ = self.retrieval_module.retrieve_and_rerank(search_query)

        # 显示检索到的子块信息
        if relevant_chunks:
            chunk_info = []
            for chunk in relevant_chunks:
                doc_name = chunk.metadata.get('doc_name', '未知文档')
                category = chunk.metadata.get('category', '')
                chunk_info.append(f"{doc_name}[{category}]" if category else doc_name)
            print(f"找到 {len(relevant_chunks)} 个相关文档块: {', '.join(chunk_info)}")
            # 显示每个文档块的内容预览
            for i, chunk in enumerate(relevant_chunks, 1):
                content_preview = chunk.page_content[:100].replace('\n', ' ')
                print(f"   文档块{i}: {content_preview}...")
        else:
            print("找到 0 个相关文档块")

        # 4. 紧急情况下即使没有检索到资料也必须输出急救指导
        if not relevant_chunks:
            if is_emergency:
                print("⚠️ 未检索到相关资料，仍将输出通用急救指导...")
                return self.generation_module.generate_answer(
                    question, [], route_info, stream=stream
                )
            return "抱歉，没有找到相关的医学资料。请尝试更换关键词，或咨询专业医生。"

        # 5. 父子文档映射（子块 → 前后窗口拼接的父上下文）
        #    旧版 get_parent_documents 返回的父文档 page_content 为空，
        #    导致 LLM 收到空上下文；改为按窗口扩展，content 真实非空。
        print("扩展父上下文窗口...")
        relevant_docs = self.data_module.expand_to_parent_windows(
            relevant_chunks,
            self.merged_vectorstore,
            window_size=self.config.parent_window_size,
        )

        if relevant_docs:
            win_info = [
                f"{d.metadata.get('doc_name', '未知')}[窗口{d.metadata.get('window', '')},{len(d.page_content)}字]"
                for d in relevant_docs
            ]
            print(f"生成 {len(relevant_docs)} 个父上下文: {', '.join(win_info)}")
        else:
            print("父上下文为空")

        # 6. 按路由类型生成回答
        print("✍️ 生成回答...")
        return self.generation_module.generate_answer(
            question, relevant_docs, route_info, stream=stream
        )

    def _extract_filters_from_query(self, query: str) -> dict:
        """
        从用户问题中提取学科过滤条件（关键词匹配）

        TODO: 当前使用关键词匹配，无法处理语义相近但字面不同的情况。
        例如"感冒了怎么办"不会命中"内科学"，但语义上属于内科学范畴。
        可改为让 LLM 判断学科归属（在 query_router 中增加 subject 维度），
        利用语义理解自动应用学科过滤，提高检索精准度。
        """
        filters = {}
        # 完整学科名匹配
        for cat in DataPreparationModule.CATEGORY_MAPPING.values():
            if cat in query:
                filters['category'] = cat
                break

        # 简称匹配（"内科" 命中 "内科学"）
        if 'category' not in filters:
            for short, full in [('内科', '内科学'), ('外科', '外科学'),
                                ('儿科', '儿科学'), ('药理', '药理学')]:
                if short in query:
                    filters['category'] = full
                    break

        return filters

    def run_interactive(self):
        """运行交互式问答"""
        print("=" * 60)
        print("🏥  医学RAG系统 - 交互式问答  🏥")
        print("=" * 60)
        print("💡 基于医学教材的智能问答，紧急情况会优先给出急救指导")

        # 初始化系统
        self.initialize_system()

        # 构建知识库
        self.build_knowledge_base()

        print("\n交互式问答 (输入'退出'结束):")

        while True:
            try:
                user_input = input("\n您的问题: ").strip()
                if user_input.lower() in ['退出', 'quit', 'exit', '']:
                    break

                # 询问是否使用流式输出
                stream_choice = input("是否使用流式输出? (y/n, 默认y): ").strip().lower()
                use_stream = stream_choice != 'n'

                print("\n回答:")
                print("-" * 40)
                try:
                    if use_stream:
                        # 流式输出
                        for chunk in self.ask_question(user_input, stream=True):
                            print(chunk, end="", flush=True)
                    else:
                        # 普通输出
                        answer = self.ask_question(user_input, stream=False)
                        print(answer)
                except Exception as stream_error:
                    print(f"\n生成回答时出错: {stream_error}")
                print("\n" + "-" * 40)

            except KeyboardInterrupt:
                print("\n检测到退出信号...")
                break
            except Exception as e:
                print(f"处理问题时出错: {e}")

        print("\n感谢使用医学RAG系统！")


def main():
    """主函数"""
    try:
        # 创建RAG系统
        rag_system = MedicalRAGSystem()

        # 运行交互式问答
        rag_system.run_interactive()

    except Exception as e:
        logger.error(f"系统运行出错: {e}")
        print(f"系统错误: {e}")

if __name__ == "__main__":
    main()
