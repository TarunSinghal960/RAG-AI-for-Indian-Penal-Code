import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/ask"

st.set_page_config(page_title="RAG AI App")

st.title("📚 RAG AI Assistant (Gemini + FAISS)")

query = st.text_input("Ask something from your documents:")

if st.button("Submit"):
    if query:
        response = requests.post(API_URL, json={"query": query})
        answer = response.json()["response"]

        st.subheader("Answer:")
        st.write(answer)