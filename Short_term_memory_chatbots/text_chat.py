from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

llm = OllamaLLM(model="llama3.2:1b")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Answer questions based on the provided context"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]
)

chain = prompt | llm

chat_history = []

def get_response(user_input: str, history: list) -> str:
    
    response = chain.invoke({"input": user_input, "chat_history": history})
    
    return response

print("Chat bot initialized")

while True:
    user_message = input("You: ")
    if user_message.lower() == 'exit':
        break
    
    ai_response = get_response(user_message, chat_history)
    print(f"Bot: {ai_response}")
    
    chat_history.append(HumanMessage(content=user_message))
    chat_history.append(AIMessage(content=ai_response))