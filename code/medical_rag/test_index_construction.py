"""
测试索引构建模块
"""

import sys
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from rag_modules.data_preparation import DataPreparationModule
from rag_modules.index_construction import IndexConstructionModule

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def main():
    data_path = "../../data/medical_rag/internal"
    index_save_path = "./vector_index"

    # 1. 数据准备
    print("=" * 60)
    print("步骤 1: 数据准备")
    print("=" * 60)

    data_module = DataPreparationModule(data_path)
    data_module.load_documents()
    chunks = data_module.chunk_documents(chunk_size=1000, chunk_overlap=200)

    stats = data_module.get_statistics()
    print(f"文档数: {stats['total_documents']}")
    print(f"分块数: {stats['total_chunks']}")
    print(f"学科分类: {stats['categories']}")

    # 2. 构建索引
    print("\n" + "=" * 60)
    print("步骤 2: 构建向量索引 (bge-m3)")
    print("=" * 60)

    index_module = IndexConstructionModule(
        model_name="BAAI/bge-m3",
        index_save_path=index_save_path,
        device="cuda",  # 无 GPU 改为 "cpu"
    )

    index_module.build_vector_index(chunks)
    index_module.save_index()

    print(f"索引已保存到: {index_save_path}")

    # 3. 测试检索
    print("\n" + "=" * 60)
    print("步骤 3: 测试检索")
    print("=" * 60)

    test_queries = [
        "急性上呼吸道感染的治疗",
        "高血压的诊断标准",
        "糖尿病的临床表现",
    ]

    for query in test_queries:
        print(f"\n查询: {query}")
        print("-" * 40)

        results = index_module.similarity_search_with_score(query, k=3)

        for i, (doc, score) in enumerate(results):
            print(f"  [{i+1}] 相似度分数: {score:.4f}")
            print(f"      学科: {doc.metadata['category']}")
            print(f"      文档: {doc.metadata['doc_name']}")
            print(f"      块索引: {doc.metadata['chunk_index']}")
            print(f"      内容预览: {doc.page_content[:100]}...")
            print()

    # 4. 测试元数据过滤检索
    print("=" * 60)
    print("步骤 4: 测试元数据过滤检索")
    print("=" * 60)

    query = "感冒的治疗"
    filters = {"category": "内科学"}

    print(f"\n查询: {query}")
    print(f"过滤条件: {filters}")
    print("-" * 40)

    filtered_results = index_module.metadata_filtered_search(
        query, filters, k=3
    )

    for i, doc in enumerate(filtered_results):
        print(f"  [{i+1}] {doc.metadata['category']} - {doc.metadata['doc_name']}")
        print(f"      内容预览: {doc.page_content[:100]}...")
        print()

    print("=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
