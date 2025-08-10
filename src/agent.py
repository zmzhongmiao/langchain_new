from langchain_community.chat_models.tongyi import ChatTongyi
import os
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import DashScopeEmbeddings
from dotenv import load_dotenv
load_dotenv(override=True)
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

def load_vector():
    # 加载向量数据库
        embeddings = DashScopeEmbeddings(model='text-embedding-v4',dashscope_api_key=DASHSCOPE_API_KEY)
        vectorstore = Chroma(persist_directory="vectorstore", embedding_function=embeddings)
        retriever = vectorstore.as_retriever()
        return retriever
def create_rag_chain():
    # 使用轻量LLM（如TinyLlama或Zephyr-7B）
    llm = ChatTongyi(
          model="qwen-turbo",
          top_p=0.9,
          temperature=0.3
    )
    retriever = load_vector()
    # 提示词模板
    system_prompt = """
    你是一个AI实验平台的技术助手，名叫 LabAgent。
    你的任务是根据提供的实验手册内容，准确回答用户关于模型结构、训练参数、迁移细节等问题。
    请遵循以下规则：

    1. 回答必须基于检索到的上下文，不要编造信息。
    2. 如果不知道答案，明确说“根据现有文档无法确定”。
    3. 使用专业术语，但解释清晰。
    4. 对比原TF实现与PyTorch重构的差异时，要具体。

    上下文：
    {context}

    问题：{input}
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt)
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

if __name__ == "__main__":
    chain = create_rag_chain()

    response = chain.invoke({"input": "FaceNet迁移时用了什么优化器？"})
    print("回答：", response["answer"])