# ingest.py – Step 1.1: Load PDF textbooks

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain.vectorstores import Chroma

# Load API key from .env
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Folder containing your 9 subject PDFs
PDF_DIR = "D:/AKMESSI/CODING/AI/Learning AI Agents/Projects/Textbook AI Assistant/data"

# Loading function

def load_pdfs_from_directory(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory, filename)
            print(f"📘 Loading {filename}...")
            loader = PyPDFLoader(pdf_path)
            # Each page becomes a Document; we'll merge these later
            pdf_docs = loader.load()
            documents.extend(pdf_docs)
    return documents

# Splitting function

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    split_docs = splitter.split_documents(documents)
    return split_docs

# Embedding model

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Storing in Vector DB 

def store_embeddings(chunks, embedding_model, persist_dir="chroma_db"):
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir
    )
    vectorstore.persist()
    print(f"💾 Stored {len(chunks)} chunks in ChromaDB at '{persist_dir}")

if __name__ == "__main__":
    all_documents = load_pdfs_from_directory(PDF_DIR)
    print(f"✅ Loaded {len(all_documents)} pages from {len(os.listdir(PDF_DIR))} textbooks.")

    # Split into chunks
    chunked_documents = split_documents(all_documents)
    print(f"✂️ Split into {len(chunked_documents)} chunks.")

    # Store embeddings
    store_embeddings(chunked_documents, embedding_model)