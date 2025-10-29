import streamlit as st
from utils.sql_manager import init_db,save_chat_responses,get_chat_history,get_chat_summary_record,save_chat_summary_record
from utils.chat_manager import prepare_chat_system_prompt,get_llm_response,prepare_summary_prompt
from utils.vectorstore_manager import update_vector_store, get_vectordb_search_results

import os
from dotenv import load_dotenv
load_dotenv()

# Configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

def generate_response(question, user_id):
    # Get general vector db search results
    general_vectordb_results = get_vectordb_search_results(user_question = question, 
                                                           user_id       = None)
    
    # Get user vector db search results
    user_vectordb_results = get_vectordb_search_results(user_question = question, 
                                                        user_id       = user_id)
    
    # Get chat history
    chat_history = get_chat_history(user_id)
    
    # Get record of previous chat summary
    previous_chat_summary = get_chat_summary_record(user_id)
    
    # Prepare chat system prompt using vector db results and chat histories and summaries
    system_prompt = prepare_chat_system_prompt(general_vectordb_results = general_vectordb_results,
                                               user_vectordb_results    = user_vectordb_results,
                                               chat_history             = chat_history, 
                                               chat_summary             = previous_chat_summary)
    
    # Get response from LLM
    response = get_llm_response(prompt   = system_prompt, 
                                question = question)
    
    # Update the databases
    save_chat_responses(question, response, user_id)
    user_summary_prompt = prepare_summary_prompt(previous_summary = previous_chat_summary[-1:],
                                                 question         = question, 
                                                 answer           = response,
                                                 type             = "user")
    new_summary = get_llm_response(prompt   = user_summary_prompt,
                                   question = None)
    save_chat_summary_record(chat_summary = new_summary,
                             user_id      = user_id)
    update_vector_store(new_summary = new_summary, 
                        user_id     = user_id)
    general_summary_prompt = prepare_summary_prompt(previous_summary = previous_chat_summary[-1:],
                                                    question         = question, 
                                                    answer           = response,
                                                    type             = "general")
    general_summary = get_llm_response(prompt   = general_summary_prompt,
                                       question = None)
    update_vector_store(new_summary = general_summary,
                        user_id     = None)
    
    return response

# Streamlit App UI
st.set_page_config(page_title="Chat with Ollama", layout="centered")
st.title("Chatbot with Ollama")

# --- Step 1: Ask for User ID before chat starts ---
if "user_id" not in st.session_state:
    st.session_state.user_id = None

if st.session_state.user_id is None:
    st.markdown("### Please enter your User ID to start chatting:")
    user_id_input = st.text_input("User ID")

    if st.button("Start Chat") and user_id_input.strip():
        st.session_state.user_id = user_id_input.strip()
        st.success(f"Welcome, **{st.session_state.user_id}**! You can now start chatting.")
        st.rerun()  # refresh to show chat interface
else:
    # Initialize DB
    init_db()
    
    current_id = st.session_state.user_id
    st.markdown(f"**User ID:** `{current_id}`")
    st.markdown(f"Model: `{OLLAMA_MODEL}`")

    # --- Step 2: Initialize chat history ---
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = get_chat_history(current_id)

    # --- Step 3: Clear chat ---
    if st.button("Clear Displayed Chat"):
        st.session_state.chat_history = []

    # --- Step 4: Display chat history ---
    for user_msg, bot_msg in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(user_msg)
        with st.chat_message("assistant"):
            st.markdown(bot_msg)

    # --- Step 5: Chat input ---
    if question := st.chat_input("Say something..."):
        with st.chat_message("user"):
            st.markdown(question)

        with st.spinner("Thinking..."):
            response = generate_response(question, current_id)
            st.session_state.chat_history.append((question, response))

        with st.chat_message("assistant"):
            st.markdown(response)