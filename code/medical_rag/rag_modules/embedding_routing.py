"""
嵌入路由模块
根据用户查询自动路由到对应学科，在学科范围内进行检索
"""

import logging
from typing import List, Dict, Any
import numpy as np

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.utils.math import cosine_similarity

logger = logging.getLogger(__name__)


class EmbeddingRoutingModule:
    """嵌入路由模块 - 根据学科分类路由到学科内检索"""

    # 学科路由配置（只保留描述和文件夹映射，移除 prompt）
    ROUTE_CONFIG = {
        "内科学": {
            "description": "处理心血管、呼吸、消化、内分泌、风湿免疫等内科疾病的诊断与治疗。包括高血压、冠心病、糖尿病、肺炎、胃炎、肝炎、肾炎等内科常见病。",
            "folder": "internal",
        },
        "外科学": {
            "description": "处理手术、创伤、肿瘤、畸形等外科疾病的治疗。包括骨折、阑尾炎、胆囊炎、疝气、乳腺疾病、甲状腺疾病等外科常见病。",
            "folder": "surgery",
        },
        "儿科学": {
            "description": "处理儿童生长发育、儿科疾病、新生儿疾病、儿童保健等。包括小儿感冒、肺炎、腹泻、营养不良、发育迟缓、疫苗接种等儿科常见病。",
            "folder": "pediatrics",
        },
        "药理学": {
            "description": "处理药物作用机制、用药指导、药物不良反应、药物相互作用、合理用药等。包括常见药物的用法用量、禁忌症、注意事项等。",
            "folder": "pharmacology",
        },
    }

    def __init__(
        self,
        vectorstore: FAISS,
        chunks: List[Document],
        embedding_model: str = "BAAI/bge-m3",
        top_k: int = 3,
    ):
        """
        初始化嵌入路由模块

        Args:
            vectorstore: FAISS 向量存储（未使用，保留接口兼容）
            chunks: 文档块列表
            embedding_model: 嵌入模型名称
            top_k: 检索返回的文档数
        """
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.top_k = top_k

        # 初始化嵌入模型
        logger.info(f"初始化嵌入模型: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # 预计算路由描述的向量
        self._setup_route_embeddings()

    def _setup_route_embeddings(self):
        """预计算路由描述的向量"""
        logger.info("预计算路由描述向量...")

        self.route_names = list(self.ROUTE_CONFIG.keys())
        self.route_descriptions = [
            self.ROUTE_CONFIG[name]["description"] for name in self.route_names
        ]

        # 编码路由描述
        self.route_embeddings = self.embeddings.embed_documents(self.route_descriptions)
        logger.info(f"已创建 {len(self.route_names)} 个学科路由")

    def route_query(self, query: str) -> str:
        """
        路由用户查询到最匹配的学科

        Args:
            query: 用户查询

        Returns:
            最匹配的学科名称
        """
        # 编码用户查询
        query_embedding = self.embeddings.embed_query(query)

        # 计算与各学科的余弦相似度
        similarity_scores = cosine_similarity(
            [query_embedding], self.route_embeddings
        )[0]

        # 选择最相似的学科
        chosen_index = np.argmax(similarity_scores)
        chosen_route = self.route_names[chosen_index]

        logger.info(
            f"路由决策: '{query[:30]}...' → {chosen_route} "
            f"(相似度: {similarity_scores[chosen_index]:.3f})"
        )

        return chosen_route

    def search(self, query: str, top_k: int = None) -> List[Document]:
        """
        路由到学科后，在学科范围内检索

        Args:
            query: 用户查询
            top_k: 检索返回的文档数（默认使用配置中的 top_k）

        Returns:
            检索到的文档列表
        """
        if top_k is None:
            top_k = self.top_k

        # 1. 路由到学科
        route_name = self.route_query(query)

        # 2. 获取该学科的所有 chunk
        route_chunks = [
            chunk
            for chunk in self.chunks
            if chunk.metadata.get("category") == route_name
        ]

        # 3. 构建该学科的向量存储并检索
        if not route_chunks:
            logger.warning(f"学科 {route_name} 没有文档块")
            return []

        route_vectorstore = FAISS.from_documents(route_chunks, self.embeddings)
        docs = route_vectorstore.similarity_search(query, k=top_k)

        logger.info(f"学科 {route_name} 内检索返回 {len(docs)} 个文档")
        return docs

    def filtered_search(self, query: str, filters: Dict[str, Any]) -> List[Document]:
        """
        带元数据过滤的检索（此模块自带学科路由，忽略外部 filters）

        Args:
            query: 查询文本
            filters: 元数据过滤条件（此模块忽略，使用内置学科路由）

        Returns:
            检索到的文档列表
        """
        logger.info(f"嵌入路由模块忽略外部过滤条件，使用内置学科路由")
        return self.search(query)
