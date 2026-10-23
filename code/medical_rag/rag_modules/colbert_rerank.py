"""
ColBERT 重排模块
使用 bge-reranker-v2-m3 对检索结果进行重排序
"""

import logging
from typing import List, Tuple
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from langchain_core.documents import Document
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor

logger = logging.getLogger(__name__)


class ColBERTReranker(BaseDocumentCompressor):
    """ColBERT 重排器 - 使用 bge-reranker-v2-m3"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        """
        初始化重排器

        Args:
            model_name: 重排模型名称
        """
        super().__init__()

        logger.info(f"加载重排模型: {model_name}")

        # 设置 HuggingFace 镜像
        import os
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

        # 加载模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

        # 检测设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        logger.info(f"重排模型加载完成，使用设备: {self.device}")

    def compress_documents(
        self,
        documents: List[Document],
        query: str,
        callbacks=None,
    ) -> List[Document]:
        """
        对文档进行重排序

        Args:
            documents: 待重排的文档列表
            query: 查询文本
            callbacks: 回调函数

        Returns:
            重排后的文档列表
        """
        if len(documents) == 0:
            return documents

        logger.info(f"开始重排 {len(documents)} 个文档...")

        # 准备输入
        pairs = []
        for doc in documents:
            pairs.append([query, doc.page_content])

        # 编码
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
            return_token_type_ids=False,
        )

        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 计算分数
        with torch.no_grad():
            scores = self.model(**inputs).logits.squeeze(-1)

        # 转换为 Python 列表
        scores = scores.cpu().tolist()

        # 按分数排序
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # 返回排序后的文档
        reranked_docs = [doc for doc, _ in scored_docs]

        logger.info(
            f"重排完成，最高分: {scores[0]:.3f}, "
            f"最低分: {scores[-1]:.3f}"
        )

        return reranked_docs


class ColbertRerankModule:
    """ColBERT 重排模块"""

    def __init__(
        self,
        vectorstore,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        initial_top_k: int = 20,
        final_top_k: int = 3,
    ):
        """
        初始化 ColBERT 重排模块

        Args:
            vectorstore: FAISS 向量存储
            model_name: 重排模型名称
            initial_top_k: 初始检索返回的文档数
            final_top_k: 重排后返回的文档数
        """
        self.vectorstore = vectorstore
        self.initial_top_k = initial_top_k
        self.final_top_k = final_top_k

        # 初始化重排器
        self.reranker = ColBERTReranker(model_name=model_name)

    def retrieve_and_rerank(self, query: str) -> Tuple[List[Document], List[float]]:
        """
        检索并重排

        Args:
            query: 查询文本

        Returns:
            (重排后的文档列表, 对应的分数列表)
        """
        logger.info(f"开始检索: '{query[:30]}...'")

        # 1. 初始检索
        initial_docs = self.vectorstore.similarity_search(
            query, k=self.initial_top_k
        )
        logger.info(f"初始检索返回 {len(initial_docs)} 个文档")

        # 2. 重排
        reranked_docs = self.reranker.compress_documents(initial_docs, query)

        # 3. 取 top-k
        final_docs = reranked_docs[: self.final_top_k]

        # 计算分数（用于展示）
        pairs = [[query, doc.page_content] for doc in final_docs]
        inputs = self.reranker.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
            return_token_type_ids=False,
        )
        inputs = {k: v.to(self.reranker.device) for k, v in inputs.items()}

        with torch.no_grad():
            scores = self.reranker.model(**inputs).logits.squeeze(-1)

        scores = scores.cpu().tolist()

        logger.info(f"重排后返回 top-{len(final_docs)} 文档")

        return final_docs, scores
