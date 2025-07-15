import streamlit as st
import os
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

# Load local .env (only for local development)
load_dotenv()

# ✅ Use Streamlit secrets for LangSmith tracking
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with OLLAMA"

# ------------------ Prompt Template ------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user queries."),
    ("user", "Question: {question}")
])

# ------------------ Generate Response Function ------------------
def generate_response(question, model_name, temperature, max_tokens):
    try:
        llm = Ollama(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser
        return chain.invoke({'question': question})
    except Exception as e:
        return f"⚠️ Error while generating response: {str(e)}"

# ------------------ Streamlit UI Setup ------------------
st.set_page_config(page_title="Ollama Q&A Chatbot", page_icon="🤖")
st.title("🤖 Q&A Chatbot with Ollama")
st.markdown("Ask any question and get an answer using a local Ollama-powered LLM.")

# Sidebar Settings
st.sidebar.title("⚙️ Settings")
model_name = st.sidebar.selectbox("Select an Ollama Model", ["gemma:2b", "llama2", "mistral", "codellama"])
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
max_tokens = st.sidebar.slider("Max Tokens", 50, 300, 150)

# Chat Input
st.write("💬 Go ahead and ask your question below:")
user_input = st.text_input("You:")

if user_input:
    with st.spinner("🔄 Generating answer..."):
        response = generate_response(user_input, model_name, temperature, max_tokens)
        st.success("✅ Response:")
        st.write(f"**Assistant:** {response}")
else:
    st.info("💡 Waiting for your question...")
