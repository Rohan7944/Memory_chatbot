import ollama

# Configuration
OLLAMA_MODEL = 'llama3.2:1b'

def prepare_vectordb_search_prompt(user_question) -> str:
    return f"Do not make up an answer. Answer the given question strictly based on the following user query: {user_question}"
    
def prepare_summary_prompt(previous_summary, question: str, answer: str) -> str:
    summary_prompt = """Summarize the following conversation while preserving key details and 
    conversations's tone:\n\n"""

    if previous_summary:
        summary_prompt += f"Previous chat summary(s) in a list:\n{previous_summary}\n\n"
        
    summary_prompt += f"Here is the recent chat conversation:"
    summary_prompt += f"\nUser: {question}\nAssistant: {answer}\n\n"

    summary_prompt += "Provide a concise summary while keeping important details."
    
    print(f"VERBOSE: Prepared summary prompt")
    
    return summary_prompt

def get_llm_response(prompt: str, question: str) -> str:
    # Prepare messages to pass into LLM
    messages = []
    if question:
        messages.append({'role': 'system', 'content': prompt})
        messages.append({'role': 'user', 'content': question})
    else:
        messages.append({'role': 'user', 'content': prompt})
    
    # Get response from LLM
    response = ollama.chat(model=OLLAMA_MODEL, messages=messages)
    
    print(f"VERBOSE: Generated response from LLM")
    
    return response['message']['content']
    
def prepare_chat_system_prompt(vectordb_results,chat_history, chat_summary) -> str:
    # Function to return prompt for chat conversations
    prompt = """You are a professional assistant. Your job is to answer user questions but in a brief manner within 100 words
    such that no important information isn't left out and you will receive a user query."""
    
    if vectordb_results:
        prompt += "\n\n"
        prompt += f"""Here are the similarity search results retrieved from a chat history vector database 
        based on user query:\n{vectordb_results}"""
    
    if chat_summary:
        prompt += "\n\n"
        prompt += f"Here are the summaries of the last few conversations in a list:\n{chat_summary}"
        
    if chat_history:
        prompt += "\n\n"
        prompt += f"Here are the last few conversation between you and the user in a list:\n{chat_history}"
        
    print(f"VERBOSE: Prepared chat system prompt")
    
    return prompt