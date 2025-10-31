from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

from utils.prompt_manager import prepare_vectordb_search_prompt

import os
from dotenv import load_dotenv
load_dotenv()

# Configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
ollama_embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
vectordb_path = os.getenv("VECTORDB_PATH")

def get_vectordb_search_results(user_question: str, user_id: str):
    """
    Performs a similarity search in the appropriate vector database (user or general)
    based on the provided user_id and returns only the search results text.
    """
    # === Step 1: Prepare semantic search prompt ===
    prompt = prepare_vectordb_search_prompt(user_question)
    embeddings = ollama_embeddings

    # === Step 2: Determine database path ===
    if user_id:
        path = os.path.join(vectordb_path, "user", user_id)
    else:
        path = os.path.join(vectordb_path, "general", "knowledge_store")

    # === Step 3: Check if the database exists ===
    if not os.path.exists(path):
        return None

    # === Step 4: Load the FAISS index ===
    vectorstore_faiss = FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # === Step 5: Perform similarity search ===
    results = vectorstore_faiss.similarity_search(prompt)

    # === Step 6: Concatenate page contents ===
    if results:
        search_results = "\n".join(doc.page_content for doc in results)
    else:
        search_results = None

    # === Step 7: Print verbose logs ===
    if user_id:
        print("VERBOSE: User Vector DB search results fetched")
    else:
        print("VERBOSE: General Vector DB search results fetched")

    return search_results


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