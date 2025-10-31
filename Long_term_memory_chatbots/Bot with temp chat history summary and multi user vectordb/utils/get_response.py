import os
from utils.response_manager import (
    get_llm_response,
    build_messageslist,
    prepare_basic_chat_system_prompt,
    summarize_within_token_limit,
)
from utils.token_counter import is_contextwindow_full
from dotenv import load_dotenv

load_dotenv()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
MAX_ITERATIONS = int(os.getenv("MAX_SUMMARIZATION_ITERATIONS", 3))  # default to 3 if not set

def prepare_llm_response_with_resources(
    question: str,
    chat_history: list | None = None,
    chat_summary: list | None = None,
    user_vectordb_results: list | None = None,
    general_vectordb_results: list | None = None,
) -> str:
    """
    Builds a prompt progressively using available data resources.
    Summarizes and generates intermediate response only if context window is breached.
    Max iterations controlled by .env variable MAX_SUMMARIZATION_ITERATIONS.
    """

    prompt = prepare_basic_chat_system_prompt()
    print("VERBOSE: Initialized base system prompt")

    def get_remaining_tokens(temp_prompt: str) -> tuple[int, int]:
        messages = build_messageslist(temp_prompt, question)
        context_info = is_contextwindow_full(OLLAMA_MODEL, messages)
        return context_info.get("remaining", 0), context_info.get("used", 0)

    def inject_data_resource(prompt: str, resource: list | None, resource_name: str, intro_text: str) -> str:
        if not resource:
            print(f"VERBOSE: No data found for {resource_name}. Skipping injection.")
            return prompt

        data_slice = resource[-min(len(resource), 100):]

        iteration = 0
        while iteration < MAX_ITERATIONS:
            remaining_tokens, used_tokens = get_remaining_tokens(prompt)
            print(f"VERBOSE: Attempting to inject '{resource_name}' (Tokens used: {used_tokens}, Remaining: {remaining_tokens})")

            # Decide whether to summarize or inject directly
            estimated_tokens_needed = sum(len(str(d)) // 4 for d in data_slice)  # rough estimate
            if remaining_tokens < estimated_tokens_needed:
                print(f"VERBOSE: ⚠️ Resource '{resource_name}' may exceed context window. Summarizing...")
                resource_text = summarize_within_token_limit(
                    data=data_slice,
                    remaining_tokens=remaining_tokens,
                    question=question
                )
                print(f"VERBOSE: Resource '{resource_name}' summarized for injection.")
            else:
                resource_text = "\n".join(map(str, data_slice))

            temp_prompt = prompt + "\n\n" + f"{intro_text}\n" + resource_text
            remaining_after, used_after = get_remaining_tokens(temp_prompt)

            if remaining_after > 0:
                print(f"VERBOSE: Successfully injected '{resource_name}'. Tokens used: {used_after}, Remaining: {remaining_after}")
                return temp_prompt
            else:
                print(f"VERBOSE: ⚠️ Context window breached after adding '{resource_name}', generating intermediate response...")
                intermediate_response = get_llm_response(temp_prompt, question)
                print(f"VERBOSE: Intermediate response generated after context breach for '{resource_name}'.")
                prompt = prepare_basic_chat_system_prompt() + "\n\n" + f"Here is a summarized version of prior information:\n{intermediate_response}"
                iteration += 1
                print(f"VERBOSE: Rebuilt prompt after summarizing '{resource_name}', iteration {iteration}.")

        print(f"VERBOSE: Maximum summarization iterations reached for '{resource_name}'. Proceeding with current prompt.")
        return prompt

    # Inject resources
    prompt = inject_data_resource(
        prompt=prompt,
        resource=chat_history,
        resource_name="chat_history",
        intro_text="Here is the recent chat history between user and assistant:"
    )

    prompt = inject_data_resource(
        prompt=prompt,
        resource=chat_summary,
        resource_name="chat_summary",
        intro_text="Here is a summary of the most recent conversations:"
    )

    prompt = inject_data_resource(
        prompt=prompt,
        resource=user_vectordb_results,
        resource_name="user_vectordb_results",
        intro_text=(
            "Here are the similarity search results of the most recent user-specific information "
            "retrieved from the database based on user query:"
        )
    )

    prompt = inject_data_resource(
        prompt=prompt,
        resource=general_vectordb_results,
        resource_name="general_vectordb_results",
        intro_text=(
            "Here are the similarity search results of the most recent general knowledge information "
            "retrieved from the database based on user query:"
        )
    )

    print("VERBOSE: Generating final LLM response...")
    final_response = get_llm_response(prompt, question)
    print("VERBOSE: LLM response successfully generated.")

    return final_response