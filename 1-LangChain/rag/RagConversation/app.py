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

# Load env variables
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Streamlit app setup
st.set_page_config(page_title="PDF Chatbot with RAG", layout="wide")
st.title("📄 RAG Q&A PDF Chatbot with Chat History")
st.markdown("Upload a PDF, ask questions, and retain full conversational context.")

# Input Groq API Key
api_key = st.text_input("🔐 Enter your Groq API Key:", type="password")

if api_key:
    llm = ChatGroq(groq_api_key=api_key, model="llama3-70b-8192")

    # Session & History Initialization
    session_id = st.text_input("🆔 Enter Session ID", value="default_session")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = {}
    if "last_user_question" not in st.session_state:
        st.session_state.last_user_question = None

    # Upload PDFs
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

        # Contextualization prompt
        contextualize_q_system_prompt = (
            "Given a chat history and the user's latest question, "
            "reformulate it into a standalone question. Do not answer it. "
            "Return as-is if already standalone."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # Answer prompt
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following retrieved context to answer the question. "
            "If you don’t know, say 'I don’t know'. Max 3 sentences.\n\n{context}"
        )
        question_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        question_chain = create_stuff_documents_chain(llm=llm, prompt=question_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_chain)

        # Chat history management
        def get_chat_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.chat_history:
                st.session_state.chat_history[session_id] = ChatMessageHistory()
            return st.session_state.chat_history[session_id]

        conversation_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_chat_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )


        # User Input
        user_input = st.text_input("💬 Ask a question:")
        session_history = get_chat_history(session_id)

        if user_input:
            if "last question" in user_input.lower():
                if st.session_state.last_user_question:
                    st.success(f"🧠 Your last question was: '{st.session_state.last_user_question}'")
                else:
                    st.info("🧠 You haven't asked a question yet.")
            else:
                # Save question manually for last-question recall
                st.session_state.last_user_question = user_input

                # Run RAG
                response = conversation_rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}},
                )

                st.success(f"🤖 Assistant: {response['answer']}")

        # Show full chat history
        with st.expander("📜 Full Conversation History", expanded=True):
            if session_history.messages:
                for i, msg in enumerate(session_history.messages):
                    role = "🧑 You" if msg.type == "human" else "🤖 Assistant"
                    st.markdown(f"**{role}**: {msg.content}")
            else:
                st.info("No conversation history yet.")
    else:
        st.warning("📂 Please upload at least one PDF file.")
else:
    st.warning("🔐 Please enter your Groq API key.")
