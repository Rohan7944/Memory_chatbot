from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

import os

from utils.chat_manager import prepare_vectordb_search_prompt

# Configuration
OLLAMA_MODEL = 'llama3.2:1b'
ollama_embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
vectordb_path = "data/faiss_index"

def get_vectordb_search_results(user_question: str):
    # Get vector db search prompt
    prompt = prepare_vectordb_search_prompt(user_question)
    
    embeddings = ollama_embeddings
    
    if os.path.exists(vectordb_path):
        vectorstore_faiss = FAISS.load_local(vectordb_path, embeddings, allow_dangerous_deserialization=True)
        results = vectorstore_faiss.similarity_search(prompt)
        r = ""
        for doc in results:
            r += f"{doc.page_content}\n"
        print(f"VERBOSE: Vector DB search results created")
        return r
    else:
        return None

def update_vector_store(chat_summary):
    # Create a text data source for vector store documents and ingesting data
    summary_text = "Given below are the summaries of conversation between a user and a chatbot stored in database-\n\n"
    for index, value in enumerate(chat_summary):
        summary_text += f"Summary of conversation number {index}: {value}\n"
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_text(summary_text)
    
    # Create vector embeddings and vector store
    embeddings = ollama_embeddings
    vectorstore_faiss = FAISS.from_texts(docs, embeddings)
    vectorstore_faiss.save_local(vectordb_path)
    
    print(f"VERBOSE: Vector store updated")