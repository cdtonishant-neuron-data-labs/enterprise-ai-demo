import os
import streamlit as st
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

st.set_page_config(page_title="AI Data Insights Demo", layout="wide")
st.title("🤖 Secure AI & Live Enterprise Data Pipeline")

# Retrieve hidden secure keys from Streamlit Cloud Secrets management
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
INDEX_NAME = "enterprise-knowledge"

user_query = st.text_input("Ask our AI Agent anything about corporate data policies:", placeholder="e.g., What is our policy on international expense payouts?")

if user_query:
    with st.spinner("Querying vector database and synthesizing answer..."):
        # Connect to cloud database infrastructure
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(INDEX_NAME)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

        # Vectorize the text query
        query_vector = embeddings.embed_query(user_query)
        search_results = index.query(vector=query_vector, top_k=2, include_metadata=True)

        retrieved_context = ""
        for match in search_results['matches']:
            retrieved_context += match['metadata']['text'] + "\n\n"

        system_prompt = f"You are a corporate legal compliance assistant. Answer the user strictly using the provided context.\n\nCONTEXT:\n{retrieved_context}"

        llm = ChatOpenAI(model="gpt-4o", temperature=0.0, api_key=OPENAI_API_KEY)
        response = llm.invoke([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_query}])

        st.write("### 📋 AI Assistant Response:")
        st.info(response.content)

st.write("---")
st.write("### 📊 Live Executive BI Insights Dashboard")
# Paste your actual Power BI embed iframe URL inside the quotes below
power_bi_url = "https://powerbi.com"
st.components.v1.iframe(power_bi_url, height=600, scrolling=True)
