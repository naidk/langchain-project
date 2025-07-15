import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os

# Set environment variables from Streamlit secrets
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with OPENAI"
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user queries."),
    ("user", "Question: {question}")
])

# Response Generator Function
def generate_response(question, llm_model, temperature, max_tokens):
    llm = ChatOpenAI(model=llm_model, temperature=temperature, max_tokens=max_tokens)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    return chain.invoke({'question': question})

# App Title
st.title("🤖 Enhanced Q&A Chatbot with OpenAI")

# Sidebar Settings
st.sidebar.title("⚙️ Settings")
llm_model = st.sidebar.selectbox("Choose an OpenAI Model", ["gpt-4o", "gpt-4-turbo", "gpt-4"])
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
max_tokens = st.sidebar.slider("Max Tokens", 50, 300, 150)

# Main Chat Interface
st.write("💬 Go ahead and ask any question:")
user_input = st.text_input("You:")

if user_input:
    try:
        response = generate_response(user_input, llm_model, temperature, max_tokens)
        st.write(f"**Assistant:** {response}")
    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
else:
    st.info("💡 Waiting for your question...")
