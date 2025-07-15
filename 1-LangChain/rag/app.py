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

# Load local .env only for local dev, not needed on Streamlit Cloud
load_dotenv()

# Set HF token from secrets
os.environ["HF_TOKEN"] = st.secrets.get("HF_TOKEN", "")

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Set up Streamlit UI
st.set_page_config(page_title="RAG Q&A with PDF", layout="wide")
st.title("📄 RAG Q&A: Conversational PDF Assistant")
st.markdown("Upload PDF(s), ask questions, and retain chat history between questions.")

# Get Groq API key securely from secrets
groq_api_key = st.secrets.get("GROQ_API_KEY")

if groq_api_key:
    llm = ChatGroq(groq_api_key=groq_api_key, model="llama3-70b-8192")

    # Session ID input
    session_id = st.text_input("🆔 Session ID", value="default_session")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = {}

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

        # Split PDFs into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(documents)

        # Create vector store + retriever
        vector_store = Chroma.from_documents(documents=split_docs, embedding=embeddings)
        retriever = vector_store.as_retriever()

        # History-aware retriever prompt
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the user's latest question, "
                       "reformulate it into a standalone question. Do not answer, just rephrase if necessary."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # QA prompt
        question_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant for question-answering tasks. "
                       "Use the following pieces of retrieved context to answer the question. "
                       "If you don't know the answer, say 'I don't know'. "
                       "Keep the answer under three sentences.\n\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        question_chain = create_stuff_documents_chain(llm=llm, prompt=question_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_chain)

        # Message history handler
        def get_chat_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.chat_history:
                st.session_state.chat_history[session_id] = ChatMessageHistory()
            return st.session_state.chat_history[session_id]

        # Wrap RAG chain with message history
        conversation_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_chat_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

        # Question input
        user_input = st.text_input("💬 Ask a question based on your PDF:")
        session_history = get_chat_history(session_id)

        if user_input:
            if "last question" in user_input.lower():
                previous_questions = [
                    msg.content for msg in session_history.messages if msg.type == "human"
                ]
                if previous_questions:
                    st.success(f"🧠 Your last question was: '{previous_questions[-1]}'")
                else:
                    st.info("🧠 You haven't asked any questions yet.")
            else:
                response = conversation_rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}},
                )
                st.success(f"🤖 Assistant: {response['answer']}")

            # Show history
            with st.expander("📜 Full Conversation History", expanded=True):
                for msg in session_history.messages:
                    role = "🧑 You" if msg.type == "human" else "🤖 Assistant"
                    st.markdown(f"**{role}:** {msg.content}")

    else:
        st.warning("📂 Please upload at least one PDF file to proceed.")
else:
    st.warning("🔐 Please set your GROQ_API_KEY in Streamlit secrets to run this app.")
