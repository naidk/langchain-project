import os
import streamlit as st
from dotenv import load_dotenv

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load env variables (for local testing)
load_dotenv()

# Set Hugging Face token securely
os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Streamlit UI setup
st.set_page_config(page_title="PDF Chatbot with RAG", layout="wide")
st.title("📄 RAG Q&A PDF Chatbot with Chat History")
st.markdown("Upload a PDF, ask questions, and retain full conversational context.")

# Get Groq API key from secrets
groq_api_key = st.secrets.get("GROQ_API_KEY")

if not groq_api_key:
    st.warning("🔐 Please add your GROQ_API_KEY in Streamlit secrets to proceed.")
    st.stop()

# Initialize Groq LLM
llm = ChatGroq(groq_api_key=groq_api_key, model="llama3-70b-8192")

# Session ID and chat history
session_id = st.text_input("🆔 Enter Session ID", value="default_session")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}
if "last_user_question" not in st.session_state:
    st.session_state.last_user_question = None

# PDF Upload
uploaded_files = st.file_uploader("📎 Upload PDF file(s)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())
        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()
        documents.extend(docs)

    # Split and embed
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)
    vector_store = Chroma.from_documents(documents=split_docs, embedding=embeddings)
    retriever = vector_store.as_retriever()

    # Contextual question reformulation prompt
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the user's latest question, "
                   "reformulate it into a standalone question. Do not answer it. "
                   "Return as-is if already standalone."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # Final answer prompt
    question_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant for question-answering tasks. "
                   "Use the following retrieved context to answer the question. "
                   "If you don’t know, say 'I don’t know'. Keep the answer under 3 sentences.\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    question_chain = create_stuff_documents_chain(llm=llm, prompt=question_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_chain)

    # Chat history function
    def get_chat_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.chat_history:
            st.session_state.chat_history[session_id] = ChatMessageHistory()
        return st.session_state.chat_history[session_id]

    # Wrap with history
    conversation_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_chat_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    # User input
    user_input = st.text_input("💬 Ask a question based on your PDF:")
    session_history = get_chat_history(session_id)

    if user_input:
        if "last question" in user_input.lower():
            if st.session_state.last_user_question:
                st.success(f"🧠 Your last question was: '{st.session_state.last_user_question}'")
            else:
                st.info("🧠 You haven't asked a question yet.")
        else:
            st.session_state.last_user_question = user_input

            # Run the chain
            response = conversation_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            )

            st.success(f"🤖 Assistant: {response['answer']}")

    # Display history
    with st.expander("📜 Full Conversation History", expanded=True):
        if session_history.messages:
            for i, msg in enumerate(session_history.messages):
                role = "🧑 You" if msg.type == "human" else "🤖 Assistant"
                st.markdown(f"**{role}**: {msg.content}")
        else:
            st.info("No conversation history yet.")
else:
    st.warning("📂 Please upload at least one PDF file to continue.")
