"""
医学RAG系统配置文件
"""

from dataclasses import asdict, dataclass
from typing import Dict, Any

@dataclass
class RAGConfig:
    """医学RAG系统配置类"""

    # 路径配置
    data_path: str = "../../data/medical_rag"
    index_save_path: str = "./vector_index"

    # 模型配置
    embedding_model: str = "BAAI/bge-m3"
    llm_model: str = "glm-5.3"
    device: str = "cuda"  # 嵌入模型运行设备: 'cuda' 或 'cpu'（CUDA 不可用时自动回退 cpu）

    # 分块配置
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # 检索策略配置（切换检索方案只需改这一个参数）
    # 可选: "hybrid"          - FAISS + BM25 + RRF 混合检索
    #       "colbert"         - FAISS 初始检索 + BGE 重排器精排
    #       "embedding_route" - 嵌入相似度路由到学科，学科内检索
    retrieval_strategy: str = "colbert"

    # 检索配置
    top_k: int = 3
    rrf_k: int = 60

    # 父子文档（上下文窗口）配置
    # 检索命中子块后，向前后各扩展 parent_window_size 个子块拼成"父上下文"喂给 LLM
    # （small-to-big / 句子窗口检索）。设为 0 则退化为只用命中的子块本身。
    parent_window_size: int = 5
    # 送入 LLM 的上下文最大字符数（窗口拼接后可能较长，需配合调大）
    context_max_length: int = 24000

    # ColBERT 重排策略参数（retrieval_strategy="colbert" 时生效）
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    colbert_initial_top_k: int = 20

    # 生成配置
    temperature: float = 0.1
    max_tokens: int = 4096

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RAGConfig':
        """从字典创建配置对象"""
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

# 默认配置实例
DEFAULT_CONFIG = RAGConfig()
