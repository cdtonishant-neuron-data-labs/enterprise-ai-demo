import os
import streamlit as st
import streamlit.components.v1 as components
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from helpers.hr_helper import render_hr_compliance_demo
from admin.ingest import run_data_ingestion
# from helpers.finance_helper import render_financial_auditor_demo

st.set_page_config(page_title="AI Data Insights Demo", layout="wide")
st.title("🤖 Secure AI & Live Enterprise Data Pipeline")

# Retrieve hidden secure keys from Streamlit Cloud Secrets management
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
INDEX_NAME = "enterprise-knowledge"


# Temporary Admin button to load data into your empty Pinecone Index 
if st.sidebar.button("⚙️ Admin: Run Data Ingestion Pipeline"): 
    run_data_ingestion(openai_key=OPENAI_API_KEY, 
        pinecone_key=PINECONE_API_KEY, 
        index_name=INDEX_NAME
    )

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
    render_hr_compliance_demo(
        openai_key=OPENAI_API_KEY, 
        pinecone_key=PINECONE_API_KEY, 
        index_name=INDEX_NAME
    )

elif demo_selection == "2. Automated Financial Auditor":
    render_financial_auditor_demo()

elif demo_selection == "3. E-commerce Customer Sentiment Pipeline":
    render_sentiment_pipeline_demo()

elif demo_selection == "4. B2B Sales Intel & Lead Enrichment":
    render_sales_intel_demo()

st.write("---")
st.write("### 📊 Live Executive BI Insights Dashboard")
# Paste your actual Power BI embed iframe URL inside the quotes below
power_bi_url = "https://microsoft.com"
components.iframe(power_bi_url, height=600, scrolling=True)
