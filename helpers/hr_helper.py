import streamlit as st
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# accept keys as parameters:
def render_hr_compliance_demo(openai_key, pinecone_key, index_name):
    st.title("🤖 Secure AI & Live Enterprise Data Pipeline")
    
    user_query = st.text_input(
        "Ask our AI Agent anything about corporate data policies:", 
        placeholder="e.g., What is our policy on international expense payouts?"
    )
    
    if user_query:
        with st.spinner("Querying vector database and synthesizing answer..."):
            # Use the variables passed into the function parameters
            pc = Pinecone(api_key=pinecone_key)
            index = pc.Index(index_name)
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_key)
            
            query_vector = embeddings.embed_query(user_query)
            search_results = index.query(vector=query_vector, top_k=2, include_metadata=True)
            
            retrieved_context = ""
            for match in search_results['matches']:
                retrieved_context += match['metadata']['text'] + "\n\n"
                
            system_prompt = f"You are a corporate legal compliance assistant. Answer the user strictly using the provided context.\n\nCONTEXT:\n{retrieved_context}"
            
            llm = ChatOpenAI(model="gpt-4o", temperature=0.0, api_key=openai_key)
            response = llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ])
            
            st.write("### 📋 AI Assistant Response:")
            st.info(response.content)
