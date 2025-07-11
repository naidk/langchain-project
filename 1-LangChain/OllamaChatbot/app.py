import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama


import os
from dotenv import load_dotenv
load_dotenv()


## langsmith tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] ="Q&A Chatbot with OLLAMA"

## Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user queries."),
    ("user", "Question: {question}")
])

# Function to generate response
def generate_response(question, model_name):
    llm = Ollama(model=model_name)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question': question})
    return answer

# Streamlit UI
st.title("🤖 Q&A Chatbot with Ollama")

# Sidebar Settings
st.sidebar.title("Settings")
model_name = st.sidebar.selectbox("Select an Ollama Model", ["gemma:2b"])
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

# Main App Interface
st.write("Go ahead and ask any question!")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input, model_name)
    st.write(f"**Assistant:** {response}")
else:
    st.info("💬 Waiting for your question...")