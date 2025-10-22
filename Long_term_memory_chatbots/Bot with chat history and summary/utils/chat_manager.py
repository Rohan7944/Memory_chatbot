import ollama

# Configuration
OLLAMA_MODEL = 'llama3.2:1b'

def prepare_summary_prompt(previous_summary, question: str, answer: str) -> str:
    summary_prompt = "Summarize the following conversation:\n\n"

    if previous_summary:
        summary_prompt += f"Previous chat summary:\n{previous_summary}\n\n"
        
    summary_prompt += f"Here is the recent chat conversation:"
    summary_prompt += f"\nUser: {question}\nAssistant: {answer}\n\n"

    summary_prompt += "Provide a concise summary while keeping important details."
    
    print(f"VERBOSE: Prepared new chat summary")
    
    return summary_prompt

def get_llm_response(prompt: str, question: str) -> str:
    # Prepare messages to pass into LLM
    messages = []
    messages.append({'role': 'system', 'content': prompt})
    if question:
        messages.append({'role': 'user', 'content': question})
    
    # Get response from LLM
    response = ollama.chat(model=OLLAMA_MODEL, messages=messages)
    
    print(f"VERBOSE: Generated response from LLM")
    
    return response['message']['content']
    
def prepare_chat_system_prompt(chat_history, chat_summary) -> str:
    # Function to return prompt for chat conversations
    prompt = """You are a professional assistant. Your job is to answer user questions but in a brief manner within 100 words
    such that no important information isn't left out."""
    
    if chat_summary:
        prompt += "\n\n"
        prompt += f"Here are the summaries of the previous conversation history in a list:\n{chat_summary}"
        
    if chat_history:
        prompt += "\n\n"
        prompt += f"Here are the last 5 conversation between you and the user in a list:\n{chat_history}"
        
    print(f"VERBOSE: Prepared chat system prompt")
    
    return prompt