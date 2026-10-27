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
    
    # 3. 按文件夹分组（直接用英文文件夹名）
    docs_by_folder = {}
    for doc in documents:
        folder_name = doc.metadata.get('folder_name', 'unknown')
        if folder_name not in docs_by_folder:
            docs_by_folder[folder_name] = []
        docs_by_folder[folder_name].append(doc)
    
    print(f"\n📊 发现 {len(docs_by_folder)} 个学科文件夹:")
    for folder_name, docs in docs_by_folder.items():
        category = docs[0].metadata.get('category', '未知') if docs else '未知'
        print(f"   - {folder_name} ({category}): {len(docs)} 个文档")
    
    # 4. 按文件夹构建索引
    print("\n🔨 开始构建索引...")
    
    for folder_name, docs in docs_by_folder.items():
        category = docs[0].metadata.get('category', '未知') if docs else '未知'
        
        print(f"\n{'='*40}")
        print(f"处理学科: {category} (文件夹: {folder_name})")
        print(f"{'='*40}")
        
        # 检查索引是否已存在（检查文件夹存在、不为空、且有 faiss+pkl 文件）
        if index_module.index_exists(folder_name):
            print(f"✅ 索引已存在且有效，跳过构建")
            print(f"   路径: {config.index_save_path}/{folder_name}")
            continue
        
        print(f"📝 索引不存在或无效，开始构建...")
        
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
        
        # 保存索引（直接用文件夹名）
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
