"""
医学RAG系统 - 索引构建模块
负责：向量化 + FAISS 索引构建 + 持久化保存/加载
"""

import os
import logging
from typing import List
from pathlib import Path

# 设置 HuggingFace 镜像（解决国内访问问题）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class IndexConstructionModule:
    """索引构建模块 - 负责向量化和索引构建"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        index_save_path: str = "./vector_index",
        device: str = "cuda",
    ):
        """
        初始化索引构建模块

        Args:
            model_name: 嵌入模型名称，默认 bge-m3
            index_save_path: 索引保存路径
            device: 运行设备，'cpu' 或 'cuda'
        """
        self.model_name = model_name
        self.index_save_path = index_save_path
        self.device = device
        self.embeddings = None
        self.vectorstore = None
        self.setup_embeddings()

    def setup_embeddings(self):
        """初始化嵌入模型"""
        logger.info(f"正在初始化嵌入模型: {self.model_name}")

        # 检查 CUDA 可用性
        import torch
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA 不可用，自动切换到 CPU")
            self.device = "cpu"

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True},
        )

        logger.info(f"嵌入模型初始化完成，使用设备: {self.device}")

    def build_vector_index(self, chunks: List[Document]) -> FAISS:
        """
        构建 FAISS 向量索引

        Args:
            chunks: 文档块列表

        Returns:
            FAISS 向量存储对象
        """
        logger.info("正在构建 FAISS 向量索引...")

        if not chunks:
            raise ValueError("文档块列表不能为空")

        self.vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings,
        )

        logger.info(f"向量索引构建完成，包含 {len(chunks)} 个向量")
        return self.vectorstore

    def save_index(self, category: str = None):
        """
        保存向量索引到磁盘

        Args:
            category: 学科分类，如果提供则保存到子目录
        """
        if not self.vectorstore:
            raise ValueError("请先构建向量索引")

        if category:
            save_path = Path(self.index_save_path) / category
        else:
            save_path = Path(self.index_save_path)

        save_path.mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(save_path)
        logger.info(f"向量索引已保存到: {save_path}")

    def load_index(self, category: str = None):
        """
        从磁盘加载向量索引

        Args:
            category: 学科分类，如果提供则从子目录加载

        Returns:
            加载的向量存储对象，加载失败返回 None
        """
        if not self.embeddings:
            self.setup_embeddings()

        if category:
            load_path = Path(self.index_save_path) / category
        else:
            load_path = Path(self.index_save_path)

        if not load_path.exists():
            logger.info(f"索引路径不存在: {load_path}，将构建新索引")
            return None

        try:
            self.vectorstore = FAISS.load_local(
                load_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info(f"向量索引已从 {load_path} 加载")
            return self.vectorstore
        except Exception as e:
            logger.warning(f"加载向量索引失败: {e}，将构建新索引")
            return None

    def index_exists(self, category: str = None) -> bool:
        """
        检查索引是否存在且有效（文件夹存在且包含索引文件）

        Args:
            category: 学科分类

        Returns:
            索引是否存在且有效
        """
        if category:
            check_path = Path(self.index_save_path) / category
        else:
            check_path = Path(self.index_save_path)

        # 检查文件夹是否存在
        if not check_path.exists():
            return False

        # 检查文件夹是否为空
        if not any(check_path.iterdir()):
            return False

        # 检查是否包含必要的索引文件
        index_faiss = check_path / "index.faiss"
        index_pkl = check_path / "index.pkl"

        return index_faiss.exists() and index_pkl.exists()

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        相似度搜索

        Args:
            query: 查询文本
            k: 返回结果数量

        Returns:
            相似文档列表
        """
        if not self.vectorstore:
            raise ValueError("请先构建或加载向量索引")

        return self.vectorstore.similarity_search(query, k=k)

    def similarity_search_with_score(self, query: str, k: int = 5):
        """
        带分数的相似度搜索

        Args:
            query: 查询文本
            k: 返回结果数量

        Returns:
            (文档, 分数) 列表，分数越小越相似
        """
        if not self.vectorstore:
            raise ValueError("请先构建或加载向量索引")

        return self.vectorstore.similarity_search_with_score(query, k=k)

    def metadata_filtered_search(
        self, query: str, filters: dict, k: int = 5
    ) -> List[Document]:
        """
        带元数据过滤的相似度搜索

        Args:
            query: 查询文本
            filters: 过滤条件，如 {"category": "内科学"}
            k: 返回结果数量

        Returns:
            相似文档列表
        """
        if not self.vectorstore:
            raise ValueError("请先构建或加载向量索引")

        # 构建 FAISS 过滤函数
        def filter_func(metadata: dict) -> bool:
            for key, value in filters.items():
                if metadata.get(key) != value:
                    return False
            return True

        return self.vectorstore.similarity_search(
            query, k=k, filter=filter_func
        )
