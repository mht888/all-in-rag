"""
测试数据准备模块
"""

import sys
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from rag_modules.data_preparation import DataPreparationModule

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def main():
    data_path = "../../data/medical_rag"

    # 1. 初始化 & 加载文档
    module = DataPreparationModule(data_path)
    parent_docs = module.load_documents()

    print("\n" + "=" * 60)
    print("父文档信息:")
    print("=" * 60)
    for doc in parent_docs:
        print(f"  名称: {doc.metadata['doc_name']}")
        print(f"  分类: {doc.metadata['category']}")
        print(f"  ID:   {doc.metadata['parent_id']}")
        print(f"  路径: {doc.metadata['source']}")
        print("-" * 40)

    # 2. PDF 提取 + 分块
    chunks = module.chunk_documents(chunk_size=1000, chunk_overlap=200)

    print("\n" + "=" * 60)
    print("子块信息 (前 5 个):")
    print("=" * 60)
    for i, chunk in enumerate(chunks[:5]):
        print(f"  [{i+1}] chunk_id: {chunk.metadata['chunk_id'][:12]}...")
        print(f"      parent_id: {chunk.metadata['parent_id'][:12]}...")
        print(f"      category: {chunk.metadata['category']}")
        print(f"      doc_name: {chunk.metadata['doc_name']}")
        print(f"      chunk_index: {chunk.metadata['chunk_index']}")
        print(f"      chunk_size: {chunk.metadata['chunk_size']}")
        print(f"      内容预览: {chunk.page_content[:200]}...")
        print("-" * 40)

    # 3. 统计信息
    stats = module.get_statistics()
    print("\n" + "=" * 60)
    print("统计信息:")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # 4. 验证父子映射
    print("\n" + "=" * 60)
    print("父子映射验证:")
    print("=" * 60)
    print(f"  父子映射条数: {len(module.parent_child_map)}")

    # 取第一个子块，验证能找到父文档
    if chunks:
        test_chunk = chunks[0]
        parents = module.get_parent_documents([test_chunk])
        print(f"  子块 '{test_chunk.metadata['doc_name']}' -> 父文档: {parents[0].metadata['doc_name']}")


if __name__ == "__main__":
    main()
