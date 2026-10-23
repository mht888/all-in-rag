"""
FAISS + BM25 混合检索模块
结合向量检索和 BM25 关键词检索，使用 RRF 融合排序
"""

import logging
import hashlib
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)


class HybridSearchModule:
    """FAISS + BM25 混合检索模块"""

    def __init__(
        self,
        vectorstore: FAISS,
        chunks: List[Document],
        top_k: int = 3,
        rrf_k: int = 60,
    ):
        """
        初始化混合检索模块

        Args:
            vectorstore: FAISS 向量存储
            chunks: 文档块列表
            top_k: 最终返回的文档数
            rrf_k: RRF 算法的 k 参数
        """
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.top_k = top_k
        self.rrf_k = rrf_k

        # 初始化检索器
        self._setup_retrievers()

    def _setup_retrievers(self):
        """设置向量检索器和 BM25 检索器"""
        logger.info("初始化检索器...")

        # 向量检索器
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k * 2},  # 多检索一些用于融合
        )

        # BM25 检索器
        self.bm25_retriever = BM25Retriever.from_documents(
            self.chunks,
            k=self.top_k * 2,
        )

        logger.info("检索器初始化完成")

    def hybrid_search(self, query: str) -> List[Document]:
        """
        混合检索

        Args:
            query: 查询文本

        Returns:
            融合排序后的文档列表
        """
        logger.info(f"开始混合检索: '{query[:30]}...'")

        # 1. 向量检索
        vector_docs = self.vector_retriever.invoke(query)
        logger.info(f"向量检索返回 {len(vector_docs)} 个文档")

        # 2. BM25 检索
        bm25_docs = self.bm25_retriever.invoke(query)
        logger.info(f"BM25 检索返回 {len(bm25_docs)} 个文档")

        # 3. RRF 融合
        reranked_docs = self._rrf_rerank(vector_docs, bm25_docs)

        # 4. 取 top-k
        final_docs = reranked_docs[: self.top_k]

        logger.info(f"RRF 融合后返回 top-{len(final_docs)} 文档")

        return final_docs

    def _rrf_rerank(
        self,
        vector_docs: List[Document],
        bm25_docs: List[Document],
    ) -> List[Document]:
        """
        使用 RRF (Reciprocal Rank Fusion) 算法重排文档

        Args:
            vector_docs: 向量检索结果
            bm25_docs: BM25 检索结果

        Returns:
            重排后的文档列表
        """
        doc_scores = {}
        doc_objects = {}

        # 计算向量检索结果的 RRF 分数
        for rank, doc in enumerate(vector_docs):
            # 使用文档内容的确定性哈希作为唯一标识
            doc_id = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
            doc_objects[doc_id] = doc

            # RRF 公式: 1 / (k + rank + 1)
            rrf_score = 1.0 / (self.rrf_k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

            logger.debug(
                f"向量检索 - 文档{rank + 1}: RRF分数 = {rrf_score:.4f}"
            )

        # 计算 BM25 检索结果的 RRF 分数
        for rank, doc in enumerate(bm25_docs):
            doc_id = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
            doc_objects[doc_id] = doc

            rrf_score = 1.0 / (self.rrf_k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

            logger.debug(
                f"BM25检索 - 文档{rank + 1}: RRF分数 = {rrf_score:.4f}"
            )

        # 按最终 RRF 分数排序
        sorted_docs = sorted(
            doc_scores.items(), key=lambda x: x[1], reverse=True
        )

        # 构建最终结果
        reranked_docs = []
        for doc_id, final_score in sorted_docs:
            if doc_id in doc_objects:
                doc = doc_objects[doc_id]
                # 将 RRF 分数添加到文档元数据中
                doc.metadata["rrf_score"] = final_score
                reranked_docs.append(doc)
                logger.debug(
                    f"最终排序 - 文档: {doc.page_content[:50]}... "
                    f"最终RRF分数: {final_score:.4f}"
                )

        logger.info(
            f"RRF重排完成: 向量检索{len(vector_docs)}个文档, "
            f"BM25检索{len(bm25_docs)}个文档, "
            f"合并后{len(reranked_docs)}个文档"
        )

        return reranked_docs
