import re

import streamlit as st
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# Function to clean out thought process of llm with "think" tag
def clean_text(text: str):
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned_text.strip()

st.title("Agent with Memory")

# Saving the messages in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Adding memory to session state so that it is only create once
if "memory" not in st.session_state:
    memory = MemorySaver()
    st.session_state.memory = memory

for message in st.session_state["messages"]:
    st.chat_message(message["role"]).write(message["content"])

model = ChatOllama(model="llama3.2:1b")

chat_agent = create_react_agent(
    model=model,
    tools=[],
    name="chat_agent",
    checkpointer=st.session_state.memory # passing memory to chat agent
)

question = st.chat_input()

if question:
    # Append question to session state
    st.session_state["messages"].append({"role": "user", "content": question})
    
    # Show user messages in UI
    st.chat_message("user").write(question)
    
    # Calling the model
    result = chat_agent.invoke(
        {
            "messages":  [
                {
                    "role": "user",
                    "content": question
                }
            ]
        },
        # To identify which conversation is being referred. Need to pass specific thread ID
        # for a real time deployment
        config={"configurable": {"thread_id": "1"}}
    )

    # Extracting content from last message into llm response
    # after cleaning out thought process of llm
    response = clean_text(result["messages"][-1].content)
    
    # Append llm response to session state
    st.session_state["messages"].append({"role": "assistant", "content": response})
    
    # Show llm response in UI with a different role
    st.chat_message("assistant").write(response)