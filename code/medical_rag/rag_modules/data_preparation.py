"""
医学RAG系统 - 数据准备模块
负责：PDF加载 → 元数据标记 → Unstructured分块 → 父子文档映射
"""

import os
import logging
import hashlib
import uuid
from pathlib import Path
from typing import List, Dict, Any

# 禁用 Unstructured 的复杂 PDF 检测，强制使用 fast 策略
os.environ["UNSTRUCTURED_IO_THRESHOLD"] = "0"

# 设置 HuggingFace 镜像（解决国内访问问题）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DataPreparationModule:
    """医学RAG数据准备模块"""

    # 学科分类映射（文件夹名 → 中文标签）
    CATEGORY_MAPPING = {
        'internal': '内科学',
        'surgery': '外科学',
        'pediatrics': '儿科学',
        'pharmacology': '药理学',
    }

    def __init__(self, data_path: str):
        """
        初始化数据准备模块

        Args:
            data_path: PDF数据根目录，子文件夹名作为学科分类
        """
        self.data_path = Path(data_path)
        self.documents: List[Document] = []   # 父文档列表（每个PDF一个）
        self.chunks: List[Document] = []      # 子块列表（Unstructured分块产物）
        self.parent_child_map: Dict[str, str] = {}  # child_chunk_id → parent_id

    # ------------------------------------------------------------------
    # 1. 加载 PDF 文档，每个 PDF 视为一个父文档
    # ------------------------------------------------------------------
    def load_documents(self) -> List[Document]:
        """
        扫描 data_path 下所有 PDF 文件，每个文件创建一个父文档

        Returns:
            父文档列表
        """
        logger.info(f"正在从 {self.data_path} 加载 PDF 文档...")

        documents = []
        for pdf_file in self.data_path.rglob("*.pdf"):
            try:
                # 计算相对路径（用于生成确定性 ID）
                try:
                    relative_path = pdf_file.resolve().relative_to(
                        self.data_path.resolve()
                    ).as_posix()
                except ValueError:
                    relative_path = pdf_file.as_posix()

                # 从父文件夹名提取学科分类
                folder_name = pdf_file.parent.name  # 英文文件夹名
                category = self._extract_category(pdf_file)  # 中文分类名

                # 文档名称 = 文件名（不含扩展名）
                doc_name = pdf_file.stem

                # 确定性唯一 ID（基于相对路径的 MD5）
                parent_id = hashlib.md5(relative_path.encode("utf-8")).hexdigest()

                doc = Document(
                    page_content="",  # 内容暂空，分块时再通过 Unstructured 提取
                    metadata={
                        "source": str(pdf_file),
                        "parent_id": parent_id,
                        "doc_type": "parent",
                        "category": category,
                        "folder_name": folder_name,  # 英文文件夹名，用于索引目录
                        "doc_name": doc_name,
                        "file_path": relative_path,
                    },
                )

                self._enhance_metadata(doc)
                documents.append(doc)
                logger.info(f"已加载: {doc_name} (分类: {category})")

            except Exception as e:
                logger.warning(f"加载 {pdf_file} 失败: {e}")

        self.documents = documents
        logger.info(f"共加载 {len(documents)} 个 PDF 父文档")
        return documents

    def _extract_category(self, pdf_file: Path) -> str:
        """从父文件夹名提取学科分类"""
        parent_folder = pdf_file.parent.name
        return self.CATEGORY_MAPPING.get(parent_folder, parent_folder)

    def _extract_with_pymupdf(self, pdf_path: str) -> str:
        """
        使用 PyMuPDF 提取 PDF 文本（备用方案）

        Args:
            pdf_path: PDF 文件路径

        Returns:
            提取的文本内容
        """
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        text_parts = []

        for page in doc:
            page_text = page.get_text("text")
            if page_text:
                text_parts.append(page_text.strip())

        doc.close()
        return "\n".join(text_parts)

    def _enhance_metadata(self, doc: Document):
        """
        增强元数据（预留扩展点）

        可在此处添加：页数统计、章节数量、关键词提取等
        """
        # 当前基础元数据已足够，后续可扩展
        pass

    # ------------------------------------------------------------------
    # 2. PDF 提取 + LangChain 分块 + 建立父子关系
    # ------------------------------------------------------------------
    def chunk_documents(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        """
        使用 Unstructured (fast) 提取文字 + LangChain RecursiveCharacterTextSplitter 分块
        如果 Unstructured 失败，自动回退到 pypdf

        流程：
          1. partition_pdf (fast) 提取 PDF 纯文字
          2. 如果失败或内容为空，使用 pypdf 提取
          3. RecursiveCharacterTextSplitter 按段落/句子边界分块
          4. 每个子块建立与父文档的映射关系

        Args:
            chunk_size: 每个子块的最大字符数
            chunk_overlap: 子块之间的重叠字符数

        Returns:
            子块列表
        """
        from unstructured.partition.pdf import partition_pdf
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        if not self.documents:
            raise ValueError("请先调用 load_documents() 加载文档")

        logger.info("开始提取文字并分块...")

        all_chunks = []

        for doc_idx, parent_doc in enumerate(self.documents):
            pdf_path = parent_doc.metadata["source"]
            parent_id = parent_doc.metadata["parent_id"]
            doc_name = parent_doc.metadata["doc_name"]

            logger.info(f"[{doc_idx + 1}/{len(self.documents)}] 正在解析: {doc_name}")

            # ---- Unstructured 提取文字（fast 策略，基于 pdfminer） ----
            full_text = ""
            try:
                elements = partition_pdf(
                    filename=pdf_path,
                    strategy="fast",       # fast: 基于 pdfminer，直接提取文字，速度快
                    languages=["chi_sim", "eng"],
                )
                # 拼接所有元素为完整文本
                full_text = "\n".join(str(elem).strip() for elem in elements if str(elem).strip())
                logger.info(f"  Unstructured 提取到 {len(full_text)} 个字符")
            except Exception as e:
                logger.warning(f"  Unstructured 提取失败: {e}")

            # ---- 如果 Unstructured 失败或内容为空，使用 PyMuPDF 提取 ----
            if not full_text or len(full_text) < 100:
                logger.info(f"  尝试使用 PyMuPDF 提取...")
                try:
                    full_text = self._extract_with_pymupdf(pdf_path)
                    logger.info(f"  PyMuPDF 提取到 {len(full_text)} 个字符")
                except Exception as e:
                    logger.warning(f"  PyMuPDF 提取也失败: {e}，跳过该文档")
                    continue

            if not full_text:
                logger.warning(f"  文档内容为空，跳过")
                continue

            # ---- RecursiveCharacterTextSplitter 按段落/句子边界分块 ----
            text_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", "。", "；", "，", " ", ""],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
            )

            text_chunks = text_splitter.split_text(full_text)
            logger.info(f"  分块为 {len(text_chunks)} 个子块")

            # ---- 将每个 chunk 转为 LangChain Document（子文档） ----
            for i, chunk_text in enumerate(text_chunks):
                chunk_text = chunk_text.strip()
                if not chunk_text:
                    continue

                chunk_id = str(uuid.uuid4())

                child_doc = Document(
                    page_content=chunk_text,
                    metadata={
                        "chunk_id": chunk_id,
                        "parent_id": parent_id,
                        "doc_type": "child",
                        "chunk_index": i,
                        "chunk_size": len(chunk_text),
                        # 继承父文档元数据
                        "category": parent_doc.metadata["category"],
                        "folder_name": parent_doc.metadata["folder_name"],
                        "doc_name": doc_name,
                        "source": pdf_path,
                    },
                )

                # 建立父子映射
                self.parent_child_map[chunk_id] = parent_id
                all_chunks.append(child_doc)

            logger.info(f"  完成，累计子块: {len(all_chunks)}")

        self.chunks = all_chunks
        logger.info(f"分块完成，共生成 {len(all_chunks)} 个子块")
        return all_chunks

    # ------------------------------------------------------------------
    # 3. 辅助方法
    # ------------------------------------------------------------------
    def get_parent_documents(self, child_chunks: List[Document]) -> List[Document]:
        """
        根据子块获取对应的父文档（智能去重 + 按相关性排序）

        Args:
            child_chunks: 检索到的子块列表

        Returns:
            去重后的父文档列表，按命中子块数降序排列
        """
        parent_relevance: Dict[str, int] = {}
        parent_docs_map: Dict[str, Document] = {}

        for chunk in child_chunks:
            pid = chunk.metadata.get("parent_id")
            if not pid:
                continue
            parent_relevance[pid] = parent_relevance.get(pid, 0) + 1
            if pid not in parent_docs_map:
                for doc in self.documents:
                    if doc.metadata.get("parent_id") == pid:
                        parent_docs_map[pid] = doc
                        break

        # 按命中次数降序排列
        sorted_ids = sorted(parent_relevance, key=parent_relevance.get, reverse=True)
        return [parent_docs_map[pid] for pid in sorted_ids if pid in parent_docs_map]

    def expand_to_parent_windows(
        self,
        child_chunks: List[Document],
        vectorstore,
        window_size: int = 5,
    ) -> List[Document]:
        """
        父子文档映射（上下文窗口版）—— 把检索命中的子块扩展为"父上下文"

        与 get_parent_documents 的区别：
          - 旧版父文档 = 整本 PDF，page_content 为空（load_documents 写死 ""，
            chunk_documents 也从不回填），导致送入 LLM 的上下文为空。
          - 本版父上下文 = 命中子块前后各 window_size 个相邻子块拼接而成，
            内容真实、相关性高（small-to-big / 句子窗口检索）。

        窗口数据直接来自已加载的 FAISS 索引 docstore（其中保留了每个子块的
        chunk_index / parent_id / 正文），因此无需重建索引。

        重叠去重：同一文档内，若多个命中子块的窗口相互重叠或相邻，合并为一个
        区间，避免把重复内容多次送给 LLM。

        Args:
            child_chunks: 检索命中的子块（按相关性排序）
            vectorstore: 已加载/合并的 FAISS 向量存储（用于取 docstore 还原窗口）
            window_size: 前后各扩展的子块数；0 表示只用命中子块本身

        Returns:
            父上下文 Document 列表，按命中子块的相关性顺序排列，page_content 非空
        """
        if not child_chunks:
            return []

        # 1. 从 docstore 还原 {parent_id: {chunk_index: Document}}
        parent_chunks: Dict[str, Dict[int, Document]] = {}
        try:
            doc_ids = list(vectorstore.index_to_docstore_id.values())
        except AttributeError:
            doc_ids = []
        for did in doc_ids:
            try:
                d = vectorstore.docstore.search(did)
            except Exception:
                continue
            pid = d.metadata.get("parent_id")
            ci = d.metadata.get("chunk_index")
            if pid is None or ci is None:
                continue
            parent_chunks.setdefault(pid, {})[ci] = d

        # 2. 为每个命中子块计算窗口区间，按 parent_id 归组
        #    wins: pid -> list of (start, end, rank, meta)
        wins_by_parent: Dict[str, list] = {}
        for rank, ch in enumerate(child_chunks):
            pid = ch.metadata.get("parent_id")
            ci = ch.metadata.get("chunk_index")
            meta = ch.metadata
            if pid is None or ci is None:
                # 缺少定位信息，退化为只用子块自身
                wins_by_parent.setdefault("__self__", []).append(
                    (rank, rank, rank, meta, ch.page_content)
                )
                continue
            if window_size <= 0:
                start = end = ci
            else:
                start = max(0, ci - window_size)
                end = ci + window_size
            wins_by_parent.setdefault(pid, []).append((start, end, rank, meta, None))

        # 3. 合并重叠/相邻区间，拼接窗口正文
        result: List[Document] = []
        for pid, wins in wins_by_parent.items():
            if pid == "__self__":
                # 无 parent_id 的退化情况：直接用子块自身正文
                for start, end, rank, meta, text in wins:
                    result.append(self._make_window_doc(meta, rank, text or "", f"self"))
                continue

            chunks_map = parent_chunks.get(pid, {})
            # 按区间起点排序后合并
            wins.sort(key=lambda x: (x[0], x[2]))
            merged: list = []  # (start, end, best_rank, meta)
            for start, end, rank, meta, _ in wins:
                if merged and start <= merged[-1][1] + 1:
                    # 与上一区间重叠或相邻 → 合并
                    merged[-1] = (
                        merged[-1][0],
                        max(merged[-1][1], end),
                        min(merged[-1][2], rank),
                        merged[-1][3],
                    )
                else:
                    merged.append((start, end, rank, meta))

            for start, end, rank, meta in merged:
                texts = []
                for ci in range(start, end + 1):
                    d = chunks_map.get(ci)
                    if d is not None and d.page_content:
                        texts.append(d.page_content)
                window_text = "\n".join(texts).strip()
                if not window_text:
                    continue
                result.append(
                    self._make_window_doc(meta, rank, window_text, f"{start}-{end}")
                )

        # 4. 按命中子块的相关性顺序排列
        result.sort(key=lambda doc: doc.metadata.get("_rank", 10**6))
        return result

    @staticmethod
    def _make_window_doc(meta: Dict[str, Any], rank: int, text: str, window: str) -> Document:
        """构造父上下文 Document（保留展示用元数据 + 内部排序用 _rank）"""
        return Document(
            page_content=text,
            metadata={
                "doc_name": meta.get("doc_name", ""),
                "category": meta.get("category", ""),
                "parent_id": meta.get("parent_id", ""),
                "window": window,
                "_rank": rank,
            },
        )

    def get_statistics(self) -> Dict[str, Any]:
        """获取数据统计信息"""
        if not self.documents:
            return {}

        categories = {}
        for doc in self.documents:
            cat = doc.metadata.get("category", "未知")
            categories[cat] = categories.get(cat, 0) + 1

        return {
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunks),
            "categories": categories,
            "avg_chunk_size": (
                sum(c.metadata.get("chunk_size", 0) for c in self.chunks) / len(self.chunks)
                if self.chunks else 0
            ),
        }
