import streamlit as st
from pathlib import Path
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
import mysql.connector
from langchain_groq import ChatGroq

# ---------------------------
# Streamlit UI Setup
# ---------------------------
st.set_page_config(page_title="Chat with SQL", page_icon=":robot:")
st.title("Chat with SQL Database :robot:")
st.markdown("This app allows you to interact with a SQL database using natural language queries.")

st.sidebar.header("Configuration")

# Database selection
LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"
radio_opt = ["USE SQLITe3 database - student.db", "Connect to MySQL database"]
selected_opt = st.sidebar.radio("Select Database Type which you want to chat", options=radio_opt)

# MySQL inputs if selected
if radio_opt.index(selected_opt) == 1:
    db_url = MYSQL
    mysql_host = st.sidebar.text_input("Provide MYSQL HOST")
    mysql_user = st.sidebar.text_input("Provide MYSQL USER")
    mysql_password = st.sidebar.text_input("Provide MYSQL PASSWORD", type="password")
    mysql_db = st.sidebar.text_input("Provide MYSQL DB NAME")
else:
    db_url = LOCALDB

# API key input
api_key = st.sidebar.text_input("Enter your Groq API Key", type="password")
if not api_key:
    st.warning("Please enter your Groq API Key to proceed.")
    st.stop()

# ---------------------------
# Define LLM with Groq
# ---------------------------
llm = ChatGroq(
    groq_api_key=api_key,
    model_name="llama3-70b-8192",
    streaming=True,
)

# ---------------------------
# Database Configuration
# ---------------------------
def configure_db(db_url, mysql_host=None, mysql_user=None, mysql_password=None, mysql_db=None):
    if db_url == MYSQL:
        return SQLDatabase(create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"))
    else:
        return SQLDatabase.from_uri("sqlite:///student.db")

# Connect DB
if db_url == MYSQL:
    db = configure_db(db_url, mysql_host, mysql_user, mysql_password, mysql_db)
else:
    db = configure_db(db_url)

# ---------------------------
# Create SQL Agent
# ---------------------------
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

# ---------------------------
# Chat History Initialization
# ---------------------------
if "messages" not in st.session_state or st.sidebar.button("Clear messages history"):
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hello, how can I help you?"
    }]

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---------------------------
# User Input & Response
# ---------------------------
user_query = st.chat_input(placeholder="Ask anything from a database ...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = agent.run(user_query, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
