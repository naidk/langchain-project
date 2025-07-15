import streamlit as st
from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler

# Load environment variables if running locally
load_dotenv()

# Set up API wrappers
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

search = DuckDuckGoSearchRun(name="Search")

# Page UI
st.set_page_config(page_title="Groq LLM Search Agent", page_icon="🔍")
st.title("🔍 Search Engine with Groq LLM + LangChain")
st.markdown("Ask anything — I’ll search Arxiv, Wikipedia, or the Web using Groq + LangChain.")

# Get Groq API Key securely
groq_api_key = st.secrets.get("GROQ_API_KEY", "")

if not groq_api_key:
    st.error("❗ Please set your GROQ_API_KEY in Streamlit Cloud secrets to continue.")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hi! I'm built with Groq and LangChain. I can search Arxiv, Wikipedia, or the Web. Ask me anything!"
        }
    ]

# Display chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
user_input = st.chat_input("What would you like to know?")
if user_input:
    # Add user message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Setup Groq LLM
    llm = ChatGroq(model="llama3-70b-8192", api_key=groq_api_key, streaming=True)
    tools = [arxiv, wiki, search]

    # Create LangChain agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
    )

    # Get response from agent
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        try:
            response = agent.run(user_input, callbacks=[st_cb])
        except Exception as e:
            response = f"⚠️ Error occurred: {type(e).__name__} - {str(e)}"
        st.session_state["messages"].append({"role": "assistant", "content": response})
        st.write(response)
