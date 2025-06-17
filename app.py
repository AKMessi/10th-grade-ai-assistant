import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import streamlit as st

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Load ChromaDB + Create Retriever

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever()

# Create RAG Chain

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type="stuff"
)

# if __name__ == "__main__":
#     query = input("Ask a question from the textbooks: \n")
#     result = qa_chain(query)

#     print("\n 📚 Answer:")
#     print(result["result"])

#     print("\n 📄 Sources:")
#     for doc in result["source_documents"]:
#         print(f"- {doc.metadata.get('source', 'Unknown')} - {doc.page_content[:150]}...")

# ---------------------------
# Streamlit UI Starts Here
# ---------------------------

st.set_page_config(page_title="📘 10th Grade AI Tutor", page_icon="🤖")
st.title("📘 10th Grade AI Assistant")
st.markdown("Ask me anything from your 10th-grade textbooks. I only answer using textbook content. ✅")

query = st.text_input("🔎 Ask your question:", placeholder="e.g. What is Newton's second law?")
ask_button = st.button("🧠 Get Answer")

if ask_button and query:
    with st.spinner("Thinking..."):
        result = qa_chain(query)

        st.success("✅ Answer:")
        st.write(result["result"])

        st.markdown("### 📄 Source Passages")
        for i, doc in enumerate(result["source_documents"]):
            with st.expander(f"🔹 Source {i+1}"):
                st.markdown(doc.page_content)
                st.caption(f"📁 Source: {doc.metadata.get('source', 'Unknown')}")
