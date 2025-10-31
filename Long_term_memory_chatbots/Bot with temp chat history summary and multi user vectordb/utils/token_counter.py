import tiktoken

# You can add more models and their context windows here
MODEL_CONTEXT_WINDOWS = {
    "gemma3:1b": 4096,    # 4k token context for 1B Gemma3
    "llama3.2:1b": 4096,  # 4k token context for 1B LLaMA3.2
}

def get_model_context_window(model: str) -> int:
    """
    Retrieve the context window (max tokens) for a model.
    """
    context_window = MODEL_CONTEXT_WINDOWS.get(model)
    if context_window is None:
        print(f"⚠️ Context window not known for model '{model}'.")
    return context_window

def count_tokens(messages: list) -> int:
    """
    Count tokens for a list of messages using tiktoken.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    text = ""
    for m in messages:
        # Include role markers to mimic chat formatting
        text += f"<|{m['role']}|>\n{m['content']}\n"
    return len(enc.encode(text))

def is_contextwindow_full(model: str, messages: list) -> dict:
    """
    Return a dict with tokens used, remaining, and context window size.
    """
    total_tokens = count_tokens(messages)
    context_window = get_model_context_window(model)

    if context_window is None:
        return {"used": total_tokens, "remaining": None, "context_window": None}

    remaining = max(context_window - total_tokens, 0)
    return {
        "used": total_tokens,
        "remaining": remaining,
        "context_window": context_window
    }