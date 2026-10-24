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
    llm_model: str = "qwen3.7-plus"
    device: str = "cuda"  # 嵌入模型运行设备: 'cpu' 或 'cuda'

    # 分块配置
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # 检索配置
    top_k: int = 3
    rrf_k: int = 60

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
