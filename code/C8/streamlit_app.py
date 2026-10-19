"""
尝尝咸淡RAG系统 - Streamlit Web界面
"""

import sys
import logging
from pathlib import Path

import streamlit as st

# 添加模块路径
sys.path.append(str(Path(__file__).parent))

from main import RecipeRAGSystem

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 页面配置
st.set_page_config(
    page_title="尝尝咸淡RAG系统",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_system():
    """加载并缓存RAG系统（只加载一次）"""
    with st.spinner("🚀 正在初始化RAG系统，首次加载可能需要几分钟..."):
        rag_system = RecipeRAGSystem()
        rag_system.initialize_system()

    with st.spinner("📚 正在构建知识库..."):
        rag_system.build_knowledge_base()

    return rag_system


def main():
    # 标题
    st.markdown('<p class="main-header">🍽️ 尝尝咸淡RAG系统</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">解决你的选择困难症，告别"今天吃什么"的世纪难题！</p>', unsafe_allow_html=True)

    # 加载系统
    try:
        rag_system = load_system()
    except Exception as e:
        st.error(f"系统初始化失败: {e}")
        st.info("请检查：\n1. 数据路径是否正确\n2. MOONSHOT_API_KEY 环境变量是否已设置")
        return

    # 侧边栏
    with st.sidebar:
        st.header("📊 系统信息")

        # 获取统计信息
        stats = rag_system.data_module.get_statistics()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("文档总数", stats.get('total_documents', 0))
        with col2:
            st.metric("文本块数", stats.get('total_chunks', 0))

        # 菜品分类
        st.subheader("🏷️ 菜品分类")
        categories = stats.get('categories', {})
        if categories:
            for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                st.caption(f"{cat}: {count}道")

        # 难度分布
        st.subheader("⭐ 难度分布")
        difficulties = stats.get('difficulties', {})
        if difficulties:
            for diff, count in sorted(difficulties.items()):
                st.caption(f"{diff}: {count}道")

        # 使用示例
        st.divider()
        st.header("💡 使用示例")
        st.markdown("""
        **菜品推荐：**
        - 推荐几个简单的素菜
        - 有哪些难度★★的菜品？

        **做法查询：**
        - 宫保鸡丁怎么做？
        - 红烧肉的具体步骤

        **食材查询：**
        - 西红柿炒鸡蛋需要什么食材？
        """)

    # 主界面 - 聊天区域
    st.markdown("---")

    # 初始化聊天历史
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 显示聊天历史
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else None):
            st.markdown(message["content"])

    # 快捷问题按钮
    st.markdown("#### 🎯 快捷问题")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🥗 推荐简单素菜", use_container_width=True):
            st.session_state.pending_question = "推荐几个简单的素菜"
            st.rerun()
    with col2:
        if st.button("🍗 宫保鸡丁做法", use_container_width=True):
            st.session_state.pending_question = "宫保鸡丁怎么做"
            st.rerun()
    with col3:
        if st.button("🥩 红烧肉食材", use_container_width=True):
            st.session_state.pending_question = "红烧肉需要什么食材"
            st.rerun()
    with col4:
        if st.button("⭐ 简单菜品推荐", use_container_width=True):
            st.session_state.pending_question = "有哪些难度非常简单的菜品"
            st.rerun()

    # 处理快捷问题
    pending = st.session_state.pop("pending_question", None)

    # 聊天输入
    question = st.chat_input("请输入你的问题，例如：宫保鸡丁怎么做？")

    # 优先使用快捷问题，其次使用输入框
    if pending:
        question = pending

    if question:
        # 显示用户消息
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        # 生成回答
        with st.spinner("🤖 正在思考中..."):
            try:
                answer = rag_system.ask_question(question, stream=False)
            except Exception as e:
                answer = f"❌ 生成回答时出错: {e}"

        with st.chat_message("assistant", avatar="🤖"):
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

    # 清空对话按钮
    if st.session_state.messages:
        if st.button("🗑️ 清空对话", type="secondary"):
            st.session_state.messages = []
            st.rerun()


if __name__ == "__main__":
    main()
