import streamlit as st
import validators
import urllib.request
import traceback

from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

# -------------------- Streamlit Setup --------------------
st.set_page_config(page_title="Text Summarization", page_icon="📝", layout="wide")
st.title("📝 Text Summarization with Groq")
st.subheader("Summarize YouTube or Website Content")

# -------------------- URL Input --------------------
generic_url = st.text_input("Enter YouTube or Website URL", label_visibility="collapsed")

# -------------------- Prompt Template --------------------
prompt_template = """
Provide a concise, clear summary of the following content in 300 words:
{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# -------------------- Headers (Edge/Chrome UA) --------------------
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0"
}

# -------------------- On Button Click --------------------
if st.button("🔍 Summarize the Content"):
    groq_api_key = st.secrets.get("GROQ_API_KEY", None)

    if not groq_api_key:
        st.error("❗ GROQ API Key not found in secrets.")
    elif not generic_url.strip():
        st.error("❗ Please provide a valid URL.")
    elif not validators.url(generic_url):
        st.error("❗ The URL is not valid.")
    else:
        try:
            st.info("⏳ Loading content, please wait...")

            docs = None

            # -------- YouTube Loader --------
            if "youtube.com" in generic_url or "youtu.be" in generic_url:
                try:
                    opener = urllib.request.build_opener()
                    opener.addheaders = [("User-Agent", headers["User-Agent"])]
                    urllib.request.install_opener(opener)

                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=False)
                    docs = loader.load()
                except Exception as yt_error:
                    st.error(f"❌ Failed to load YouTube video (may lack captions or be restricted): {yt_error}")

            # -------- Website Loader --------
            else:
                try:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers=headers
                    )
                    docs = loader.load()
                except Exception as web_error:
                    st.error(f"❌ Failed to load website content: {web_error}")

            # -------- Run LLM Chain if docs exist --------
            if not docs:
                st.warning("⚠️ No content found. The video may lack captions or the website blocks access.")
            else:
                llm = ChatGroq(model="llama3-70b-8192", api_key=groq_api_key)
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                summary = chain.run(docs)

                st.success("✅ Summary Generated:")
                st.write(summary)

        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {type(e).__name__}: {str(e)}")
            st.text(traceback.format_exc())
