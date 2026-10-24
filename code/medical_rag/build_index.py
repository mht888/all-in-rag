"""
医学RAG系统 - 独立索引构建脚本
按学科分类构建向量索引，已存在的学科会跳过
"""

import os
import sys
from pathlib import Path

# 修复Windows终端编码问题
if sys.platform == 'win32':
    os.system('chcp 65001 >nul 2>&1')

# 添加模块路径
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
from config import DEFAULT_CONFIG
from rag_modules import DataPreparationModule, IndexConstructionModule

# 加载环境变量
load_dotenv()

# 中文 category 到英文文件夹名的映射（避免 Windows 中文路径编码问题）
CATEGORY_FOLDER_MAP = {
    '内科学': 'internal',
    '外科学': 'surgery',
    '儿科学': 'pediatrics',
    '药理学': 'pharmacology',
}

def build_index():
    """按学科分类构建索引"""
    print("=" * 60)
    print("🔨 医学RAG系统 - 索引构建工具")
    print("=" * 60)
    
    config = DEFAULT_CONFIG
    
    # 1. 初始化模块
    print("\n📦 初始化模块...")
    data_module = DataPreparationModule(config.data_path)
    index_module = IndexConstructionModule(
        model_name=config.embedding_model,
        index_save_path=config.index_save_path,
        device=config.device
    )
    
    # 2. 加载文档
    print("\n📄 加载文档...")
    documents = data_module.load_documents()
    
    if not documents:
        print("❌ 未找到任何PDF文档")
        return
    
    # 3. 按学科分组
    docs_by_category = {}
    for doc in documents:
        category = doc.metadata.get('category', '未知')
        if category not in docs_by_category:
            docs_by_category[category] = []
        docs_by_category[category].append(doc)
    
    print(f"\n📊 发现 {len(docs_by_category)} 个学科:")
    for category, docs in docs_by_category.items():
        print(f"   - {category}: {len(docs)} 个文档")
    
    # 4. 按学科构建索引
    print("\n🔨 开始构建索引...")
    
    for category, docs in docs_by_category.items():
        # 使用英文文件夹名（避免 Windows 中文路径编码问题）
        folder_name = CATEGORY_FOLDER_MAP.get(category, category)
        
        print(f"\n{'='*40}")
        print(f"处理学科: {category} (文件夹: {folder_name})")
        print(f"{'='*40}")
        
        # 检查索引是否已存在
        if index_module.index_exists(folder_name):
            print(f"✅ 索引已存在，跳过构建")
            print(f"   路径: {config.index_save_path}/{folder_name}")
            continue
        
        print(f"📝 索引不存在，开始构建...")
        
        # 为该学科的文档创建临时模块进行分块
        temp_module = DataPreparationModule(config.data_path)
        temp_module.documents = docs
        
        # 分块
        print(f"  正在分块...")
        chunks = temp_module.chunk_documents(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        print(f"  ✅ 分块完成: {len(chunks)} 个子块")
        
        # 构建索引
        print(f"  正在构建向量索引...")
        vectorstore = index_module.build_vector_index(chunks)
        
        # 保存索引（使用英文文件夹名）
        print(f"  正在保存索引...")
        index_module.save_index(folder_name)
        print(f"  ✅ 索引已保存到: {config.index_save_path}/{folder_name}")
    
    # 5. 显示统计信息
    print("\n" + "=" * 60)
    print("📊 构建完成统计:")
    print("=" * 60)
    stats = data_module.get_statistics()
    print(f"   文档总数: {stats['total_documents']}")
    print(f"   文本块数: {stats['total_chunks']}")
    print(f"   学科分布: {list(stats['categories'].keys())}")
    print(f"   平均块大小: {stats['avg_chunk_size']:.0f} 字符")
    print(f"\n✅ 所有索引构建完成！")
    print(f"   索引路径: {config.index_save_path}")

if __name__ == "__main__":
    try:
        build_index()
    except Exception as e:
        print(f"\n❌ 构建失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
