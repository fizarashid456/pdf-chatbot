import streamlit as st
import os
import tempfile
import time
from dotenv import load_dotenv

load_dotenv()

def get_default_api_key() -> str:
    try:
        return st.secrets.get("GOOGLE_API_KEY", "")
    except Exception:
        return os.getenv("GOOGLE_API_KEY", "")

st.set_page_config(
    page_title="PDF Chatbot - Powered by Gemini AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .main-header h1 { font-size: 2.2rem; margin: 0; font-weight: 700; }
    .main-header p { font-size: 1rem; opacity: 0.8; margin-top: 0.5rem; }
    .chat-message {
        padding: 1rem 1.2rem;
        border-radius: 12px;
        margin-bottom: 0.8rem;
        line-height: 1.6;
    }
    .user-message {
        background: linear-gradient(135deg, #0f3460, #16213e);
        color: white;
        margin-left: 2rem;
        border-left: 4px solid #e94560;
    }
    .assistant-message {
        background: #1e1e2e;
        color: #e0e0e0;
        margin-right: 2rem;
        border-left: 4px solid #4ecca3;
    }
    .source-box {
        background: #0d0d1a;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 0.8rem;
        margin-top: 0.5rem;
        font-size: 0.85rem;
        color: #aaa;
    }
    .stats-card {
        background: linear-gradient(135deg, #1e1e2e, #16213e);
        border: 1px solid #4ecca3;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        color: white;
    }
    .stats-card .number { font-size: 2rem; font-weight: 700; color: #4ecca3; }
    .stats-card .label { font-size: 0.8rem; opacity: 0.7; }
    .upload-area {
        border: 2px dashed #4ecca3;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        background: #0d0d1a;
        color: #aaa;
    }
    .footer {
        text-align: center;
        color: #555;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding: 1rem;
        border-top: 1px solid #222;
    }
    .stButton > button { border-radius: 8px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    defaults = {
        "chat_history": [],
        "vectorstore": None,
        "pdf_processed": False,
        "doc_stats": {},
        "api_key_valid": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def validate_api_key(api_key: str) -> bool:
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        models = genai.list_models()
        list(models)
        return True
    except Exception:
        return False


def process_pdf(uploaded_file, chunk_size: int, chunk_overlap: int) -> tuple:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        num_pages = len(documents)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        chunks = splitter.split_documents(documents)
        num_chunks = len(chunks)

        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=f"pdf_{int(time.time())}",
        )

        return vectorstore, num_pages, num_chunks

    finally:
        os.unlink(tmp_path)


def get_qa_chain(vectorstore, api_key: str, temperature: float, num_sources: int):
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate

    os.environ["GOOGLE_API_KEY"] = api_key

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=temperature,
        google_api_key=api_key,
    )

    prompt_template = """You are a helpful AI assistant analyzing a PDF document.
Use ONLY the context below to answer the question. If the answer isn't in the context,
say "I couldn't find this information in the document."

Context:
{context}

Question: {question}

Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": num_sources}
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )
    return chain


def ask_question(chain, question: str) -> dict:
    try:
        result = chain.invoke({"query": question})
        if isinstance(result, dict):
            answer = result.get("result", str(result))
            sources = result.get("source_documents", [])
        else:
            answer = str(result)
            sources = []
        return {"answer": answer, "sources": sources}
    except Exception as e:
        return {"answer": f"Error generating response: {str(e)}", "sources": []}


# ─── MAIN APP ────────────────────────────────────────────────────────────────

initialize_session_state()

st.markdown("""
<div class="main-header">
    <h1>📄 PDF Chatbot</h1>
    <p>Upload any PDF and ask questions — powered by Google Gemini AI</p>
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("### 🔑 Google API Key")
    api_key_input = st.text_input(
        "Enter your Gemini API key",
        type="password",
        value=get_default_api_key(),
        placeholder="AIza...",
        help="Get your free key at https://aistudio.google.com/app/apikey"
    )

    if api_key_input:
        if st.button("✅ Validate Key", use_container_width=True):
            with st.spinner("Validating..."):
                if validate_api_key(api_key_input):
                    st.session_state.api_key_valid = True
                    st.success("API key is valid!")
                else:
                    st.session_state.api_key_valid = False
                    st.error("Invalid API key. Please check and try again.")
    else:
        st.info("👆 Enter your Google API key to get started.\n\n[Get a free key →](https://aistudio.google.com/app/apikey)")

    st.divider()

    st.markdown("### 🧠 Model Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05,
        help="Lower = more precise, Higher = more creative")
    num_sources = st.slider("Sources to retrieve", 1, 8, 4,
        help="Number of document chunks to use as context")
    show_sources = st.toggle("Show source references", value=False)

    st.divider()

    st.markdown("### 📐 Text Chunking")
    chunk_size = st.slider("Chunk size", 200, 2000, 800, 100)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, 100, 50)

    st.divider()

    if st.session_state.pdf_processed and st.session_state.doc_stats:
        stats = st.session_state.doc_stats
        st.markdown("### 📊 Document Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""<div class="stats-card">
                <div class="number">{stats.get('pages', 0)}</div>
                <div class="label">Pages</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="stats-card">
                <div class="number">{stats.get('chunks', 0)}</div>
                <div class="label">Chunks</div>
            </div>""", unsafe_allow_html=True)
        st.markdown(f"**📄 File:** {stats.get('filename', 'N/A')}")
        st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    with col2:
        if st.button("🔄 Reset All", use_container_width=True):
            for key in ["chat_history", "vectorstore", "pdf_processed", "doc_stats"]:
                st.session_state[key] = [] if key == "chat_history" else (
                    None if key == "vectorstore" else False if key == "pdf_processed" else {})
            st.rerun()

# ─── MAIN CONTENT ─────────────────────────────────────────────────────────────
col_main, col_right = st.columns([3, 1])

with col_main:
    if not api_key_input:
        st.warning("⚠️ Please enter your Google Gemini API key in the sidebar to get started.")
        st.stop()

    st.markdown("### 📂 Upload Your PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file", type=["pdf"],
        help="Upload any PDF document to start chatting",
        label_visibility="collapsed"
    )

    if uploaded_file and not st.session_state.pdf_processed:
        st.markdown(f"**Selected:** `{uploaded_file.name}` ({uploaded_file.size / 1024:.1f} KB)")

        if st.button("🚀 Process PDF", type="primary", use_container_width=True):
            with st.spinner("🔍 Loading and processing your PDF..."):
                progress = st.progress(0)
                status = st.empty()
                status.text("📖 Loading PDF pages...")
                progress.progress(20)
                status.text("✂️ Splitting into chunks...")
                progress.progress(40)

                try:
                    vectorstore, num_pages, num_chunks = process_pdf(
                        uploaded_file, chunk_size, chunk_overlap)
                    progress.progress(80)
                    status.text("🧠 Building vector index...")

                    st.session_state.vectorstore = vectorstore
                    st.session_state.pdf_processed = True
                    st.session_state.doc_stats = {
                        "pages": num_pages,
                        "chunks": num_chunks,
                        "filename": uploaded_file.name,
                    }
                    st.session_state.chat_history = []
                    progress.progress(100)
                    status.empty()
                    progress.empty()
                    st.success(f"✅ PDF processed! **{num_pages} pages**, **{num_chunks} chunks**.")
                    time.sleep(1)
                    st.rerun()

                except Exception as e:
                    progress.empty()
                    status.empty()
                    st.error(f"❌ Error processing PDF: {str(e)}")

    elif uploaded_file and st.session_state.pdf_processed:
        st.success(f"✅ **{st.session_state.doc_stats.get('filename')}** is ready!")

    if st.session_state.pdf_processed:
        st.divider()
        st.markdown("### 💬 Chat with Your PDF")

        examples = ["Summarize this document", "What are the main topics?",
                    "List the key findings", "What conclusions are drawn?"]
        chips_html = "".join(
            f'<span style="display:inline-block;background:#16213e;color:#4ecca3;border:1px solid #4ecca3;border-radius:20px;padding:0.3rem 0.8rem;margin:0.2rem;font-size:0.82rem">{q}</span>'
            for q in examples)
        st.markdown(f'<div>{chips_html}</div>', unsafe_allow_html=True)
        st.caption("Type your question below ↓")

        if st.session_state.chat_history:
            st.markdown("---")
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f"""<div class="chat-message user-message">
                        <b>🧑 You:</b><br>{msg["content"]}</div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class="chat-message assistant-message">
                        <b>🤖 Assistant:</b><br>{msg["content"]}</div>""", unsafe_allow_html=True)
                    if show_sources and msg.get("sources"):
                        with st.expander(f"📚 Sources ({len(msg['sources'])} chunks)"):
                            for i, src in enumerate(msg["sources"]):
                                page = src.metadata.get("page", "?")
                                preview = src.page_content[:300].replace("\n", " ")
                                st.markdown(f"""<div class="source-box">
                                    <b>Source {i+1} — Page {page}</b><br><i>{preview}...</i>
                                </div>""", unsafe_allow_html=True)

        with st.form("chat_form", clear_on_submit=True):
            col_input, col_btn = st.columns([5, 1])
            with col_input:
                user_question = st.text_input("Ask a question",
                    placeholder="e.g., What is the main argument of this document?",
                    label_visibility="collapsed")
            with col_btn:
                submitted = st.form_submit_button("Send 📨", use_container_width=True, type="primary")

        if submitted and user_question.strip():
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            with st.spinner("🤔 Thinking..."):
                try:
                    chain = get_qa_chain(st.session_state.vectorstore, api_key_input, temperature, num_sources)
                    response = ask_question(chain, user_question)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response["sources"],
                    })
                except Exception as e:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"Sorry, error: {str(e)}",
                        "sources": [],
                    })
            st.rerun()

    elif not uploaded_file:
        st.markdown("""<div class="upload-area">
            <h3>👆 Upload a PDF to get started</h3>
            <p>Supports research papers, books, reports, manuals, and more.</p>
        </div>""", unsafe_allow_html=True)

with col_right:
    st.markdown("### 📌 Tips")
    st.info("**Best practices:**\n\n• Ask specific questions\n• Request summaries\n• Ask for comparisons\n• Request explanations")
    st.markdown("### 🛠️ Model")
    st.markdown("**Gemini 1.5 Flash**\n\nFast, capable, and free-tier friendly.")
    if st.session_state.pdf_processed:
        st.markdown("### 📈 Session")
        num_q = sum(1 for m in st.session_state.chat_history if m["role"] == "user")
        st.metric("Questions asked", num_q)

st.markdown("""<div class="footer">
    Built with ❤️ using Streamlit · LangChain · Google Gemini AI · HuggingFace Embeddings<br>
    <a href="https://aistudio.google.com/app/apikey" target="_blank">Get your free Gemini API key</a>
</div>""", unsafe_allow_html=True)
