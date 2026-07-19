from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import streamlit as st
import re

def clean_ai_response(text):
    # This regex looks for single letters wrapped in asterisks with spaces
    cleaned = re.sub(r'\*([a-zA-Z])\*\s?', r'\1', text)
    return cleaned
    
def render_hr_compliance_demo(openai_key, pinecone_key, index_name):
    st.title("🤖 Secure AI & Live Enterprise Data Pipeline")
    
    # 1. Initialize chat history in session state if it doesn't exist yet
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 2. Create a 2-column layout to show chat input and chat history side-by-side
    col1, col2 = st.columns([1, 1]) # Adjust ratios (e.g., [2, 3]) if you want one side wider

    with col1:
        st.subheader("💬 Ask a New Question")
        user_query = st.text_area(
            "Enter your policy query:", 
            placeholder="e.g., What is our policy on international expense payouts?",
            height=150,
            key="hr_user_query_input" # Unique key for session management
        )
        submit_button = st.button("Ask AI Agent")

    with col2:
        st.subheader("📜 Conversation History")
        # Container with fixed scroll height to cleanly display logs
        with st.container(height=400):
            if not st.session_state.chat_history:
                st.caption("No conversation history yet. Ask a question on the left!")
            else:
                # Render the history in reverse order (newest at the top) or standard order
                for chat in reversed(st.session_state.chat_history):
                    st.markdown(f"**👤 You:** {chat['question']}")
                    st.info(f"**🤖 AI:** {chat['answer']}")
                    st.markdown("---")

    # 3. Process the logic only when the user explicitly clicks the button
    if submit_button and user_query:
        with st.spinner("Querying vector database and synthesizing answer..."):
            pc = Pinecone(api_key=pinecone_key)
            index = pc.Index(index_name)
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_key)
            
            query_vector = embeddings.embed_query(user_query)
            search_results = index.query(vector=query_vector, top_k=2, include_metadata=True)
            
            retrieved_context = ""
            for match in search_results['matches']:
                retrieved_context += match['metadata']['text'] + "\n\n"
                
            system_prompt = f"You are a corporate legal compliance assistant. Answer the user strictly using the provided context.Respond in clean, standard Markdown prose. Do not use mathematical styling, italics, or bolding on individual characters or whole sentences unless emphasizing a key term.\n\nCONTEXT:\n{retrieved_context}"
            
            llm = ChatOpenAI(model="gpt-4o", temperature=0.0, api_key=openai_key)
            response = llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ])

            cleaned_text = clean_ai_response(response.content)
            
            # 4. Save the Q&A pair to history and force a quick rerun to update the UI
            st.session_state.chat_history.append({
                "question": user_query,
                "answer": cleaned_text
            })
            st.rerun()
