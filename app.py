import os
import streamlit as st
import streamlit.components.v1 as components
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

st.set_page_config(page_title="AI Data Insights Demo", layout="wide")
st.title("🤖 Secure AI & Live Enterprise Data Pipeline")

# Retrieve hidden secure keys from Streamlit Cloud Secrets management
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
INDEX_NAME = "enterprise-knowledge"


# Temporary Admin button to load data into your empty Pinecone Index
# if st.sidebar.button("⚙️ Admin: Run Data Ingestion Pipeline"):
 #   with st.spinner("Processing document, chunking text, and generating vector embeddings..."):
  #      try:
            # Re-using the ingestion logic from our architectural blueprint
   #         from langchain_community.document_loaders import PyPDFLoader
    #        from langchain_text_splitters import RecursiveCharacterTextSplitter
            
            # 1. Load the file from your GitHub data folder
     #       loader = PyPDFLoader("data/company_policy.pdf")
      #      raw_docs = loader.load()
            
            # 2. Slice text into blocks
       #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        #    documents = text_splitter.split_documents(raw_docs)
            
            # 3. Connect to Pinecone Index
         #   pc = Pinecone(api_key=PINECONE_API_KEY)
          #  index = pc.Index(INDEX_NAME)
           # embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
            
            # 4. Format and Upload Vectors
           # vectors_to_upsert = []
           # for i, doc in enumerate(documents):
            #    vector = embeddings.embed_query(doc.page_content)
             #   vectors_to_upsert.append({
              #      "id": f"chunk_{i}",
               #     "values": vector,
                #    "metadata": {"text": doc.page_content}
               # })
            
           # index.upsert(vectors=vectors_to_upsert)
           # st.sidebar.success("🎉 Ingestion Complete! Data is now live in Pinecone.")
            
        # except Exception as e:
          #  st.sidebar.error(f"Ingestion failed: {str(e)}")

import streamlit as st

st.set_page_config(page_title="Enterprise AI Solutions", layout="wide")

# Navigation Sidebar
st.sidebar.title("💼 Enterprise AI Portfolio")
demo_selection = st.sidebar.radio(
    "Select a Live Business Use Case:",
    [
        "1. HR & Compliance Assistant (Current)",
        "2. Automated Financial Auditor",
        "3. E-commerce Customer Sentiment Pipeline",
        "4. B2B Sales Intel & Lead Enrichment"
    ]
)

if demo_selection == "1. HR & Compliance Assistant (Current)":
    st.title("🤖 Secure AI & Live Enterprise Data Pipeline")
    render_hr_compliance_demo()

elif demo_selection == "2. Automated Financial Auditor":
    render_financial_auditor_demo()

elif demo_selection == "3. E-commerce Customer Sentiment Pipeline":
    render_sentiment_pipeline_demo()

elif demo_selection == "4. B2B Sales Intel & Lead Enrichment":
    render_sales_intel_demo()


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
components.iframe(power_bi_url, height=600, scrolling=True)
