from dotenv import load_dotenv
import os
from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser




load_dotenv()

##Langchain Tracking
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"]= os.getenv("LANGCHAIN_PROJECT")


## prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant.please responds questions asked by the user."),
        ("user","Question: {question}"),
        
        ]
)

## streamlit framework
st.title("Langchain demo with gemma2b")
input_text = st.text_input("What question you have in your mind?")

## ollama gemma2b setup
llm =Ollama(model="gemma:2b")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser
if input_text:
   st.write(chain.invoke({"question": input_text}))
