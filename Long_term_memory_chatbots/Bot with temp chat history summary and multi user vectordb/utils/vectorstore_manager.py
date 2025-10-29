from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

from utils.chat_manager import prepare_vectordb_search_prompt

import os
from dotenv import load_dotenv
load_dotenv()

# Configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
ollama_embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
vectordb_path = os.getenv("VECTORDB_PATH")

def get_vectordb_search_results(user_question: str, user_id: str):
    # Get vector db search prompt
    prompt = prepare_vectordb_search_prompt(user_question)
    
    embeddings = ollama_embeddings
    if user_id:
        path = vectordb_path + "user/" + user_id
    else:
        path = vectordb_path + "general/knowledge_store"
    
    if os.path.exists(path):
        vectorstore_faiss = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
        results = vectorstore_faiss.similarity_search(prompt)
        r = ""
        for doc in results:
            r += f"{doc.page_content}\n"
        if user_id:
            print(f"VERBOSE: User Vector DB search results created")
        else:
            print(f"VERBOSE: General Vector DB search results created")
        return r
    else:
        return None

def update_vector_store(new_summary: str, user_id: str):
    summary_text = f"Latest chat summary-\n{new_summary}\n"

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    new_texts = text_splitter.split_text(summary_text)
    embeddings = ollama_embeddings
    if user_id:
        path = vectordb_path + "user/" + user_id
    else:
        path = vectordb_path + "general/knowledge_store"
    
    if os.path.exists(path):
        db = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
        db.add_texts(new_texts)
        db.save_local(path)
    else:
        db = FAISS.from_texts(new_texts, embeddings)
        db.save_local(path)
    
    if user_id:
        print(f"VERBOSE: Vector store updated for user: {user_id}")
    else:
        print(f"VERBOSE: Vector store updated for knowledge store")