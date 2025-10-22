from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA

from utils.chat_manager import prepare_vectordb_search_prompt

# Configuration
OLLAMA_MODEL = 'llama3.2:1b'
ollama_embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)

def get_vectordb_search_results(user_question: str):
    # Get vector db search prompt
    prompt = prepare_vectordb_search_prompt(user_question)
    
    embeddings = ollama_embeddings
    vectorstore_faiss = FAISS.load_local("data/faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    qa = RetrievalQA.from_chain_type(
        llm = OllamaLLM(model=OLLAMA_MODEL),
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k",3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    answer = qa({"query":user_question})
    
    print(f"VERBOSE: Vector DB search results created: {answer['result']}")
    
    return answer['result']

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
    vectorstore_faiss.save_local("data/faiss_vector_store")
    
    print(f"VERBOSE: Vector store updated")