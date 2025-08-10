import streamlit as st
from agent import create_rag_chain

st.set_page_config(page_title="LabAgent-实验助手",page_icon="🧪")
st.title("🧪 LabAgent：AI 实验平台技术助手")
st.sidebar.header("📌 项目说明")
st.sidebar.markdown("""
- **模型**：通义千问 Qwen-Turbo  
- **知识库**：实验手册（Chroma 向量库）  
- **功能**：技术问答、参数查询、迁移指导
""")

# 可选：加个背景色
st.markdown(
    """
    <style>
    .stApp { background-color: #f8f9fa; }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown("基于 Langchain + Qwen + Chrome智能系统")

@st.cache_resource
def load_chain():
    return create_rag_chain()

chain = load_chain()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "你好！我是LabAgent,可以回答你关于模型迁移、训练参数等问题"}

    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("输入你的问题..."):
    st.session_state.messages.append({"role":"human","content":prompt})
    st.chat_message("human").write(prompt)
    with st.spinner("思考中..."):
        response = chain.invoke({"input":prompt})
        answer = response["answer"]
    
    st.session_state.messages.append({"role":"assistant","content":answer})
    st.chat_message("assistant").write(answer)