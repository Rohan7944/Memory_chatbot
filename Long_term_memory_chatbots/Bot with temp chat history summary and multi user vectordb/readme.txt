Step 1 - Create environment variables -

OLLAMA_MODEL = 'llama3.2:1b' # change model name as per your system
VECTORDB_PATH = "data/vector_stores/"
TEMP_MEMORY_DB_NAME = 'chatbot_memory.db'
SAVED_CHAT_CONVO = 5
SAVED_CHAT_SUMMARIES = 10
MAX_SUMMARIZATION_ITERATIONS=3

Step 2 - Under utils/token_counter.py, add model name and context windows under the list - MODEL_CONTEXT_WINDOWS