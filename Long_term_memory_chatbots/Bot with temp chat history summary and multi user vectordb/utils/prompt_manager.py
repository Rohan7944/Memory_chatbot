
def prepare_vectordb_search_prompt(user_question: str) -> str:
    """
    Prepare an optimized prompt for similarity search in the vector database.
    Converts the user's natural question into a clear, context-rich query 
    that improves embedding relevance and retrieval accuracy.
    """
    prompt = (
        "You are preparing a semantic search query for a vector database. "
        "Rephrase the following user question into a concise, factual query "
        "that best captures its meaning and intent. Remove conversational tone "
        "and filler words. Focus on the key topics, entities, and actions.\n\n"
        f"User question: {user_question}\n\n"
        "Optimized search query:"
    )
    return prompt

def prepare_summary_prompt(previous_summary: str, question: str, answer: str, type: str) -> str:
    """
    Prepare a prompt to summarize a chat conversation between user and LLM.
    The prompt instructs the model to return strictly the summary text.
    
    Args:
        previous_summary (str): Optional previous summary to include.
        question (str): User's message.
        answer (str): Assistant's response.
        type (str): 'user' for user-facing summary, else general summary removing PII.

    Returns:
        str: The prompt text to use for generating a summary.
    """
    
    prompt = "Summarize the following conversation, preserving key details and the conversation's tone:\n\n"
    
    if previous_summary:
        prompt += f"Previous summary:\n{previous_summary}\n\n"
    
    prompt += f"Conversation:\nUser: {question}\nAssistant: {answer}\n\n"
    
    if type == "user":
        prompt += "Provide a concise summary including all important details. Respond **only** with the summary text."
    else:
        prompt += ("Provide a concise summary including important details, but remove any personally identifiable "
                   "information (PII) or user-specific data. Respond **only** with the summary text.")
    
    return prompt