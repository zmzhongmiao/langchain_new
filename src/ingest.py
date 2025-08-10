from langchain_community.document_loaders import DirectoryLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
import os
from dotenv import load_dotenv
import markdown
from pathlib import Path
load_dotenv(override=True)

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

def md2txt():
    # 手动把 .md 转成 .txt
    for md_file in Path("data/").glob("*.md"):
        with open(md_file, 'r', encoding='utf-8') as f:
            md_text = f.read()
        html_text = markdown.markdown(md_text)
        # 去掉 HTML 标签，只留文本
        from bs4 import BeautifulSoup
        text = BeautifulSoup(html_text, 'html.parser').get_text()
        
        txt_file = md_file.with_suffix(".txt")
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(text)

def ingest_docs():
    md2txt()
    text_loader_kwargs = {"autodetect_encoding": True}
    loader = DirectoryLoader("data/", glob="*.txt",loader_cls=TextLoader,loader_kwargs=text_loader_kwargs)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
          chunk_size=500,
          chunk_overlap=50,
          separators=["\n\n", "\n", "。", ".", " ", ""]
        )
    splits = text_splitter.split_documents(docs)
    embeddings = DashScopeEmbeddings(model='text-embedding-v4',dashscope_api_key=DASHSCOPE_API_KEY)
    
    vectorstore = Chroma.from_documents(splits, embeddings, persist_directory="vectorstore")
    print("✅ 文档已向量化并保存到 vectorstore/")

if __name__ == "__main__":
    ingest_docs()