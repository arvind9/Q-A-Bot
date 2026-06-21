import streamlit as st
from query import query_rag_pipeline

st.set_page_config(page_title="BookExpert AI Assistant", page_icon="📚", layout="centered")

st.title("📚 BookExpert Document Q&A Bot")
st.subheader("AI-Powered Research Assistant with RAG")

st.markdown("---")

user_query = st.text_input("Ask a question about your documents:", placeholder="e.g., What is the total revenue listed in the report?")

if user_query:
    with st.spinner("Searching documents and generating grounded answer..."):
        try:
            result = query_rag_pipeline(user_query)
            
            st.markdown("### 🤖 Answer")
            st.write(result["answer"])
            
            st.markdown("### 📄 Source Citations")
            for citation in result["citations"]:
                st.caption(f"📍 {citation}")
                
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.info("Make sure you have placed documents in the 'data' folder and run ingestion first!")