"""
DocInsight - A simple RAG application using LlamaIndex and Streamlit.
"""
import os
import tempfile
from typing import List

import streamlit as st
from dotenv import load_dotenv

from utils.document_processor import DocumentProcessor
from utils.index_manager import IndexManager
from utils.query_engine import QueryEngine

# Load environment variables
load_dotenv()

# Check if OpenAI API key is set
if os.getenv("OPENAI_API_KEY") is None:
    st.error("Please set your OpenAI API key in the .env file or as an environment variable.")
    st.stop()

# App title and description
st.title("DocInsight")
st.subheader("Upload documents and ask questions")
st.markdown("""
This application allows you to upload PDF and text files, and then ask questions about their contents.
The system uses LlamaIndex and the OpenAI API to provide relevant answers based on your documents.
""")

# Initialize session state
if "processor" not in st.session_state:
    st.session_state.processor = DocumentProcessor()
if "index_manager" not in st.session_state:
    st.session_state.index_manager = IndexManager()
if "query_engine" not in st.session_state:
    st.session_state.query_engine = QueryEngine()
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False

# File upload section
st.header("Upload Documents")
uploaded_files = st.file_uploader(
    "Upload PDF or text files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

def save_uploaded_files(files) -> List[str]:
    """Save uploaded files to a temporary directory and return their paths."""
    temp_dir = tempfile.mkdtemp()
    file_paths = []
    
    for file in files:
        file_path = os.path.join(temp_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        file_paths.append(file_path)
    
    return file_paths

# Process documents button
if uploaded_files and st.button("Process Documents"):
    with st.spinner("Processing documents..."):
        # Save uploaded files
        file_paths = save_uploaded_files(uploaded_files)
        
        # Load and process documents
        documents = st.session_state.processor.load_documents(file_paths)
        processed_documents = st.session_state.processor.process_documents(documents)
        
        # Create index
        index = st.session_state.index_manager.create_index(processed_documents)
        
        # Set index in query engine
        st.session_state.query_engine.set_index(index)
        
        # Update session state
        st.session_state.documents_processed = True
        
        st.success(f"Processed {len(uploaded_files)} documents successfully!")

# Query section
st.header("Ask Questions")
query = st.text_input("Enter your question")

if query and st.button("Ask"):
    if st.session_state.documents_processed:
        with st.spinner("Generating answer..."):
            response = st.session_state.query_engine.query(query)
            
            if response:
                st.subheader("Answer")
                st.write(response.response)
                
                st.subheader("Sources")
                for idx, source_node in enumerate(response.source_nodes, 1):
                    st.markdown(f"**Source {idx}**: {source_node.metadata.get('file_name', 'Unknown')}")
                    st.markdown(f"**Excerpt**: {source_node.text[:300]}...")
            else:
                st.error("Failed to generate a response. Please make sure documents are processed.")
    else:
        st.error("Please process documents first!")

# Add information on the sidebar
with st.sidebar:
    st.title("About DocInsight")
    st.markdown("""
    **DocInsight** is a simple RAG (Retrieval Augmented Generation) application built with:
    - LlamaIndex
    - Streamlit
    - OpenAI API
    - ChromaDB
    """)
    
    st.subheader("How it works")
    st.markdown("""
    1. Upload PDF or text documents
    2. Click "Process Documents" to index them
    3. Ask questions about the document contents
    4. Get answers with relevant source references
    """) 