from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
from dotenv import load_dotenv
import torch
import os

def load_pdf_file(path):
    loader = DirectoryLoader(
        path, 
        glob="**/*.pdf", 
        loader_cls=PyPDFLoader
    )
    
    documents = loader.load()
    return documents

def filter_to_minimal_docs(docs: List[Document]) ->List[Document]:
    minimal_docs: List[Document] = []
    for doc in docs:
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": doc.metadata.get("source", "")}
            )
        )
    return minimal_docs

def split_text(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        length_function=len
    )
    texts_chunks = text_splitter.split_documents(docs)
    return texts_chunks

def download_huggingface_embeddings():
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    )
    return embeddings