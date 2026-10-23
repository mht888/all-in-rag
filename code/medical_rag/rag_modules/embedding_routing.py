"""
嵌入路由模块
根据用户查询自动路由到对应学科的独立检索链
"""

import os
import logging
from typing import List, Dict, Any
import numpy as np

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utils.math import cosine_similarity

logger = logging.getLogger(__name__)


class EmbeddingRoutingModule:
    """嵌入路由模块 - 根据学科分类路由到独立检索链"""

    # 学科路由配置
    ROUTE_CONFIG = {
        "内科学": {
            "description": "处理心血管、呼吸、消化、内分泌、风湿免疫等内科疾病的诊断与治疗。包括高血压、冠心病、糖尿病、肺炎、胃炎、肝炎、肾炎等内科常见病。",
            "folder": "internal",
            "role": "内科医生",
            "prompt": "你是一位经验丰富的内科医生。请根据以下医学资料，用专业但易懂的语言回答患者的问题。回答要准确、全面，必要时提醒患者及时就医。\n\n医学资料：\n{context}\n\n患者问题：{question}\n\n回答：",
        },
        "外科学": {
            "description": "处理手术、创伤、肿瘤、畸形等外科疾病的治疗。包括骨折、阑尾炎、胆囊炎、疝气、乳腺疾病、甲状腺疾病等外科常见病。",
            "folder": "surgery",
            "role": "外科医生",
            "prompt": "你是一位经验丰富的外科医生。请根据以下医学资料，用专业但易懂的语言回答患者的问题。回答要准确、全面，必要时提醒患者及时就医。\n\n医学资料：\n{context}\n\n患者问题：{question}\n\n回答：",
        },
        "儿科学": {
            "description": "处理儿童生长发育、儿科疾病、新生儿疾病、儿童保健等。包括小儿感冒、肺炎、腹泻、营养不良、发育迟缓、疫苗接种等儿科常见病。",
            "folder": "pediatrics",
            "role": "儿科医生",
            "prompt": "你是一位经验丰富的儿科医生。请根据以下医学资料，用专业但易懂的语言回答家长的问题。回答要准确、全面，必要时提醒家长及时带孩子就医。\n\n医学资料：\n{context}\n\n患者问题：{question}\n\n回答：",
        },
        "药理学": {
            "description": "处理药物作用机制、用药指导、药物不良反应、药物相互作用、合理用药等。包括常见药物的用法用量、禁忌症、注意事项等。",
            "folder": "pharmacology",
            "role": "临床药师",
            "prompt": "你是一位经验丰富的临床药师。请根据以下药学资料，用专业但易懂的语言回答患者的用药问题。回答要准确、全面，强调遵医嘱用药的重要性。\n\n药学资料：\n{context}\n\n患者问题：{question}\n\n回答：",
        },
    }

    def __init__(
        self,
        vectorstore: FAISS,
        chunks: List[Document],
        embedding_model: str = "BAAI/bge-m3",
        llm=None,
    ):
        """
        初始化嵌入路由模块

        Args:
            vectorstore: FAISS 向量存储
            chunks: 文档块列表
            embedding_model: 嵌入模型名称
            llm: LLM 实例
        """
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.llm = llm

        # 初始化嵌入模型
        logger.info(f"初始化嵌入模型: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # 预计算路由描述的向量
        self._setup_route_embeddings()

        # 为每个学科创建独立的检索链
        self._setup_route_chains()

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

    def _setup_route_chains(self):
        """为每个学科创建独立的检索链"""
        logger.info("创建学科检索链...")

        self.route_chains = {}
        for route_name, config in self.ROUTE_CONFIG.items():
            prompt = PromptTemplate(
                template=config["prompt"],
                input_variables=["context", "question"],
            )
            chain = prompt | self.llm | StrOutputParser()
            self.route_chains[route_name] = chain

        logger.info(f"已创建 {len(self.route_chains)} 个学科检索链")

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

    def retrieve_and_generate(
        self, query: str, top_k: int = 3
    ) -> Dict[str, Any]:
        """
        路由、检索、生成完整流程

        Args:
            query: 用户查询
            top_k: 检索返回的文档数

        Returns:
            包含路由结果、检索文档、生成回答的字典
        """
        # 1. 路由到学科
        route_name = self.route_query(query)

        # 2. 在该学科范围内检索
        # 获取该学科的所有 chunk
        route_folder = self.ROUTE_CONFIG[route_name]["folder"]
        route_chunks = [
            chunk
            for chunk in self.chunks
            if chunk.metadata.get("category") == route_name
        ]

        # 构建该学科的向量存储
        if not route_chunks:
            logger.warning(f"学科 {route_name} 没有文档块")
            return {
                "route": route_name,
                "documents": [],
                "answer": f"抱歉，{route_name}学科暂无相关资料。",
            }

        route_vectorstore = FAISS.from_documents(route_chunks, self.embeddings)

        # 检索
        docs = route_vectorstore.similarity_search(query, k=top_k)

        # 3. 生成回答
        context = "\n\n".join([doc.page_content for doc in docs])
        chain = self.route_chains[route_name]
        answer = chain.invoke({"context": context, "question": query})

        return {
            "route": route_name,
            "documents": docs,
            "answer": answer,
        }
