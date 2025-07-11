import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate





import os
from dotenv import load_dotenv
load_dotenv()


## langsmith tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] ="Q&A Chatbot with OPENAI"

## Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user queries."),
    ("user", "Question: {question}")
])

def generate_response(question,api_key,llm,temperature,max_tokens):
     # Set the environment variable for LangChain compatibility
    os.environ["OPENAI_API_KEY"] = api_key
    llm = ChatOpenAI(model=llm, temperature=temperature, max_tokens=max_tokens)
    output_parser=StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question':question})
    return answer


### title of the app
st.title("Enhanced Q&A chatbot with openAi")

### sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your OpenAi Api key:",type="password")


##drop down to select various openapi models
llm =st.sidebar.selectbox("Select an OpenAi models",["gpt-4o","gpt-4-turbo","gpt-4"])


## adjust response parameter
temperature = st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens = st.sidebar.slider("Max Tokens",min_value=50,max_value=300,value=150)


## main interface for user input

st.write("Go head and ask any question")
user_input = st.text_input("You:")

if user_input:
    if api_key:
        response = generate_response(user_input, api_key, llm, temperature, max_tokens)
        st.write(f"**Assistant:** {response}")
    else:
        st.warning("⚠️ Please enter your OpenAI API key.")
else:
    st.info("💬 Waiting for your question...")