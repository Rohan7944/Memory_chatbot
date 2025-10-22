import streamlit as st
from utils.sql_manager import init_db,save_chat_responses,get_chat_history,get_chat_summary_record,save_chat_summary_record
from utils.chat_manager import prepare_chat_system_prompt,get_llm_response,prepare_summary_prompt
from utils.vectorstore_manager import update_vector_store, get_vectordb_search_results

# Configuration
OLLAMA_MODEL = 'llama3.2:1b'

def generate_response(question):
    # Get vector db search results
    vectordb_results = get_vectordb_search_results(question)
    
    # Get chat history
    chat_history = get_chat_history()
    
    # Get record of previous chat summary
    previous_chat_summary = get_chat_summary_record()
    
    # Prepare chat system prompt using vector db results and last 5 chat histories and summaries
    system_prompt = prepare_chat_system_prompt(vectordb_results = vectordb_results,
                                               chat_history     = chat_history[-5:], 
                                               chat_summary     = previous_chat_summary[-5:])
    
    # Get response from LLM
    response = get_llm_response(system_prompt, question)
    
    # Update the databases
    save_chat_responses(question, response)
    summary_prompt = prepare_summary_prompt(previous_chat_summary[-1:], question, response)
    new_summary = get_llm_response(summary_prompt,None)
    save_chat_summary_record(new_summary)
    update_vector_store(previous_chat_summary)
    
    return response

# Initialize DB
init_db()

# Streamlit App UI
st.set_page_config(page_title="Chat with Ollama", layout="centered")
st.title("Chatbot with Ollama")
st.markdown(f"Model: `{OLLAMA_MODEL}`")

# Session state init
if "chat_history" not in st.session_state:
    st.session_state.chat_history = get_chat_history()

# Clear session chat button
if st.button("Clear Displayed Chat"):
    st.session_state.chat_history = []

# Display chat messages from session
for user_msg, bot_msg in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(user_msg)
    with st.chat_message("assistant"):
        st.markdown(bot_msg)

# User input
if question := st.chat_input("Say something..."):
    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner("Thinking..."):
        response = generate_response(question)
        st.session_state.chat_history.append((question, response))

    with st.chat_message("assistant"):
        st.markdown(response)