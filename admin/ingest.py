import streamlit as st
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app import PINECONE_API_KEY, INDEX_NAME, OPENAI_API_KEY

def run_data_ingestion():
    with st.spinner("Processing document, chunking text, and generating vector embeddings..."):
        try:
            # 1. Load the file from your GitHub data folder
            loader = PyPDFLoader("data/company_policy.pdf")
            raw_docs = loader.load()
            
            # 2. Slice text into blocks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            documents = text_splitter.split_documents(raw_docs)
            
            # 3. Connect to Pinecone Index
            pc = Pinecone(api_key=PINECONE_API_KEY)
            index = pc.Index(INDEX_NAME)
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
            
            # 4. Format and Upload Vectors
            vectors_to_upsert = []
            for i, doc in enumerate(documents):
                vector = embeddings.embed_query(doc.page_content)
                vectors_to_upsert.append({
                    "id": f"chunk_{i}",
                    "values": vector,
                    "metadata": {"text": doc.page_content}
                })
            index.upsert(vectors=vectors_to_upsert)
            st.sidebar.success("🎉 Ingestion Complete! Data is now live in Pinecone.")
        except Exception as e:
            st.sidebar.error(f"Ingestion failed: {str(e)}")
