import ollama

import os
from dotenv import load_dotenv
load_dotenv()

# Configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

def get_older_summaries(chat_history: list | None, previous_chat_summary: list | None) -> list:
    """
    Returns chat summaries that do not have corresponding chat history entries.
    Safe to call even if both inputs are empty or None.
    Both lists must be in ascending time order (oldest → newest).
    """
    if not previous_chat_summary:
        return []
    if not chat_history:
        return previous_chat_summary

    extra_count = len(previous_chat_summary) - len(chat_history)
    return previous_chat_summary[:extra_count] if extra_count > 0 else []

def summarize_within_token_limit(
    data,
    remaining_tokens: int,
    question: str
) -> str:
    """
    Summarizes given data strictly within the remaining token budget.
    Returns only the summary text — no extra explanations, headings, or formatting.
    """
    if isinstance(data, list):
        content = "\n".join(str(item) for item in data)
    else:
        content = str(data)

    prompt = (
        f"You will be provided with some data below.\n\n"
        f"Your task: produce a compact, information-rich summary capturing the key points relevant to the question.\n"
        f"STRICT CONSTRAINTS:\n"
        f"1. The summary must fit within {remaining_tokens} tokens (~{remaining_tokens * 4} characters).\n"
        f"2. Only return the summary text — do NOT include explanations, titles, lists, or any other content.\n"
        f"3. Be concise and exclude redundancy. Prioritize the key facts most relevant to the user's question.\n\n"
        f"User question (for context): {question}\n\n"
        f"Data:\n{content}"
    )

    # This uses your existing get_llm_response() definition
    summary = get_llm_response(prompt, question)
    print("VERBOSE: Generated response from LLM (used for summarization)")

    return summary.strip()

def build_messageslist(prompt: str, question: str) -> list:
    # Function to build messages that are passed on to LLM
    messages = []
    
    if question:
        messages.append({'role': 'system', 'content': prompt})
        messages.append({'role': 'user', 'content': question})
    else:
        messages.append({'role': 'user', 'content': prompt})
    
    return messages

def prepare_basic_chat_system_prompt() -> str:
    return """You are a professional assistant. Your job is to answer user questions but in a brief manner within 100 words 
    such that no important information isn't left out and you will receive a user query."""

def get_llm_response(prompt: str, question: str) -> str:
    # Get messages to pass into LLM
    messages = build_messageslist(prompt, question)
    
    # Get response from LLM
    response = ollama.chat(model=OLLAMA_MODEL, messages=messages)
    
    return response['message']['content']
    
# def prepare_chat_system_prompt(general_vectordb_results, user_vectordb_results, chat_summary) -> str:
#     # Function to return prompt for chat conversations
#     prompt = prepare_basic_chat_system_prompt()
    
#     if general_vectordb_results:
#         prompt += "\n\n"
#         prompt += f"""Here are the similarity search results retrieved from an all user(general) chat summary vector database 
#         based on user query:\n{general_vectordb_results}"""
    
#     if user_vectordb_results:
#         prompt += "\n\n"
#         prompt += f"""Here are the similarity search results retrieved from users chat summary vector database 
#         based on user query:\n{user_vectordb_results}"""
    
#     if chat_summary:
#         prompt += "\n\n"
#         prompt += f"""Here are the summaries of the last {len(chat_summary)} conversations sorted by time in ascending order
#         in a tuple:\n{chat_summary}"""
        
#     print(f"VERBOSE: Prepared chat system prompt")
    
#     return prompt