import streamlit as st
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Ensure these keys/constants are imported or handled properly 
# (e.g., from a config file or st.secrets)
from config import PINECONE_API_KEY, INDEX_NAME, OPENAI_API_KEY 

def render_hr_compliance_demo():
    st.title("🤖 Secure AI & Live Enterprise Data Pipeline")
    
    user_query = st.text_input(
        "Ask our AI Agent anything about corporate data policies:", 
        placeholder="e.g., What is our policy on international expense payouts?"
    )
    
    if user_query:
        with st.spinner("Querying vector database and synthesizing answer..."):
            pc = Pinecone(api_key=PINECONE_API_KEY)
            index = pc.Index(INDEX_NAME)
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
            
            query_vector = embeddings.embed_query(user_query)
            search_results = index.query(vector=query_vector, top_k=2, include_metadata=True)
            
            retrieved_context = ""
            for match in search_results['matches']:
                retrieved_context += match['metadata']['text'] + "\n\n"
                
            system_prompt = f"You are a corporate legal compliance assistant. Answer the user strictly using the provided context.\n\nCONTEXT:\n{retrieved_context}"
            
            llm = ChatOpenAI(model="gpt-4o", temperature=0.0, api_key=OPENAI_API_KEY)
            response = llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ])
            
            st.write("### 📋 AI Assistant Response:")
            st.info(response.content)
