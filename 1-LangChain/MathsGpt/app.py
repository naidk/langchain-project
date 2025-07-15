import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler

# Streamlit setup
st.set_page_config(
    page_title="🧮 Text to Math Problem Solver & Data Assistant",
    page_icon="🧠"
)
st.title("🧮 Text to Math Problem Solver using Google Gemma 2")
st.markdown("This assistant uses Groq + LangChain to solve math problems and search Wikipedia.")

# Load API key securely
groq_api_key = st.secrets.get("GROQ_API_KEY", "")

if not groq_api_key:
    st.error("❗ Please set your GROQ_API_KEY in Streamlit Cloud → Secrets to run this app.")
    st.stop()

# Initialize LLM
llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

# Tools setup
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool to search Wikipedia for facts and information about any topic."
)

# Math tool
math_chain = LLMMathChain.from_llm(llm=llm)
calculator = Tool(
    name="Calculator",
    func=math_chain.run,
    description="Use this tool to answer math-related questions. Input must be a mathematical expression."
)

# Reasoning tool (LLMChain)
reasoning_prompt = """
You are an AI agent tasked with solving users' mathematical and logical questions. 
Logically arrive at the solution, and provide a step-by-step explanation in bullet points.

Question: {question}
Answer:
"""

reasoning_template = PromptTemplate(
    input_variables=["question"],
    template=reasoning_prompt
)

reasoning_chain = LLMChain(llm=llm, prompt=reasoning_template)

reasoning_tool = Tool(
    name="Reasoning Tool",
    func=reasoning_chain.run,
    description="A tool for answering reasoning and multi-step math questions."
)

# Initialize agent
assistant_agent = initialize_agent(
    tools=[wikipedia_tool, calculator, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a Math and Reasoning Assistant! Ask me any question involving numbers, logic, or research."}
    ]

# Display message history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
question = st.text_area("Enter your question:", value="I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. Each pack of blueberries contains 25 berries. How many total pieces of fruit do I have at the end?")

if st.button("🧠 Find My Answer"):
    if question:
        with st.chat_message("user"):
            st.write(question)
        st.session_state["messages"].append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = assistant_agent.run(question, callbacks=[st_cb])
            st.write(response)
        st.session_state["messages"].append({"role": "assistant", "content": response})
    else:
        st.warning("⚠️ Please enter a question to get an answer.")
