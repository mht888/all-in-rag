"""
医学RAG系统 - 数据准备模块
负责：PDF加载 → 元数据标记 → Unstructured分块 → 父子文档映射
"""

import logging
import hashlib
import uuid
from pathlib import Path
from typing import List, Dict, Any

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
                category = self._extract_category(pdf_file)

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

    def _enhance_metadata(self, doc: Document):
        """
        增强元数据（预留扩展点）

        可在此处添加：页数统计、章节数量、关键词提取等
        """
        # 当前基础元数据已足够，后续可扩展
        pass

    # ------------------------------------------------------------------
    # 2. Unstructured 分块 + 建立父子关系
    # ------------------------------------------------------------------
    def chunk_documents(self, max_characters: int = 1000, overlap: int = 200) -> List[Document]:
        """
        使用 Unstructured 对每个父文档进行分块，并建立父子关系

        流程（参考 C2 的 01_unstructured_example.py）：
          1. partition_pdf 提取 PDF 元素（标题/段落/表格等）
          2. 使用原生 chunk_by_title 按标题元素切分章节
          3. 每个子块建立与父文档的映射关系

        Args:
            max_characters: 每个子块的最大字符数
            overlap: 子块之间的重叠字符数

        Returns:
            子块列表
        """
        from unstructured.partition.pdf import partition_pdf
        from unstructured.chunking.title import chunk_by_title

        if not self.documents:
            raise ValueError("请先调用 load_documents() 加载文档")

        logger.info("开始使用 Unstructured 进行分块...")

        all_chunks = []

        for doc_idx, parent_doc in enumerate(self.documents):
            pdf_path = parent_doc.metadata["source"]
            parent_id = parent_doc.metadata["parent_id"]
            doc_name = parent_doc.metadata["doc_name"]

            logger.info(f"[{doc_idx + 1}/{len(self.documents)}] 正在解析: {doc_name}")

            # ---- Unstructured 解析 PDF（使用 hi_res 策略识别结构） ----
            try:
                elements = partition_pdf(
                    filename=pdf_path,
                    strategy="hi_res",     # hi_res: 版面分析，识别标题/表格/图片等元素类型
                    languages=["chi_sim", "eng"],
                )
                logger.info(f"  解析得到 {len(elements)} 个元素")
            except Exception as e:
                logger.warning(f"  Unstructured 解析失败: {e}，跳过该文档")
                continue

            # ---- 使用原生 chunk_by_title 按标题边界分块 ----
            try:
                chunked_elements = chunk_by_title(
                    elements,
                    max_characters=max_characters,
                    overlap=overlap,
                )
                logger.info(f"  分块为 {len(chunked_elements)} 个子块")
            except Exception as e:
                logger.warning(f"  chunk_by_title 分块失败: {e}，将整篇文档作为单个子块")
                chunked_elements = elements

            # ---- 将每个 chunk 转为 LangChain Document（子文档） ----
            for i, chunk_elem in enumerate(chunked_elements):
                chunk_text = str(chunk_elem).strip()
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
