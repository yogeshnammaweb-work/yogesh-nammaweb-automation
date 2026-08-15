import streamlit as st
from pathlib import Path
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama


# ============================================================
# CONFIGURATION
# ============================================================

DOCUMENTS_PATH = Path("documents")
CHROMA_PATH = Path("data/chroma_db")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2:3b"

DOCUMENTS_PATH.mkdir(parents=True, exist_ok=True)
CHROMA_PATH.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NammaWeb AI",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# HTML-SAFE MARKDOWN HELPER
# ------------------------------------------------------------
# Streamlit's markdown renderer treats any line indented by
# 4+ spaces as a code block. Since our HTML snippets below are
# indented for readability, we patch st.markdown so that any
# call made with unsafe_allow_html=True first strips leading
# whitespace from every line, guaranteeing it renders as HTML
# instead of showing up as literal text/code.
# ============================================================

_original_markdown = st.markdown


def _html_safe_markdown(body, *args, **kwargs):
    if kwargs.get("unsafe_allow_html") and isinstance(body, str):
        body = "\n".join(line.lstrip() for line in body.split("\n"))
    return _original_markdown(body, *args, **kwargs)


st.markdown = _html_safe_markdown



# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

.stApp {
    background:
        radial-gradient(
            circle at 15% 10%,
            rgba(99, 102, 241, 0.10),
            transparent 30%
        ),
        radial-gradient(
            circle at 85% 90%,
            rgba(139, 92, 246, 0.08),
            transparent 30%
        ),
        #08090d;

    color: #f5f5f7;
}

.main .block-container {
    max-width: 1500px;
    padding-top: 2.2rem;
    padding-bottom: 5rem;
}

#MainMenu,
footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}


/* ========================================================
   SIDEBAR
======================================================== */

section[data-testid="stSidebar"] {
    background: #0b0c11;
    border-right: 1px solid rgba(255,255,255,0.07);
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.8rem;
}

.brand {
    font-size: 22px;
    font-weight: 800;
    letter-spacing: -0.7px;
    color: #f5f5f7;
}

.brand-icon {
    color: #a5b4fc;
}

.brand-sub {
    color: #707583;
    font-size: 11px;
    margin-top: 5px;
    margin-bottom: 25px;
}

.side-title {
    color: #666b79;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1.7px;
    margin-top: 25px;
    margin-bottom: 10px;
}

.side-document,
.system-card {
    background: #11131a;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 13px;
    padding: 14px;
    margin-bottom: 8px;
}

.side-document-name {
    font-size: 13px;
    font-weight: 700;
    color: #e7e8ed;
}

.side-document-info {
    color: #707583;
    font-size: 11px;
    margin-top: 6px;
}

.system-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.system-title {
    color: #e4e5ea;
    font-size: 12px;
    font-weight: 700;
}

.system-badge {
    color: #6ee7b7;
    font-size: 9px;
    font-weight: 800;
    letter-spacing: 0.7px;
}

.system-description {
    color: #707583;
    font-size: 10px;
    line-height: 1.5;
    margin-top: 5px;
}

.system-online {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #8b909d;
    font-size: 11px;
    margin-top: 14px;
}

.online-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #34d399;
    box-shadow: 0 0 10px rgba(52,211,153,0.7);
}


/* ========================================================
   HERO
======================================================== */

.eyebrow {
    color: #8187a0;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 2.2px;
    font-weight: 800;
}

.hero-title {
    font-size: 43px;
    line-height: 1.08;
    font-weight: 800;
    letter-spacing: -2px;
    margin-top: 8px;
    margin-bottom: 9px;
    color: #f5f5f7;
}

.hero-subtitle {
    color: #858a98;
    font-size: 14px;
    line-height: 1.6;
}


/* ========================================================
   DASHBOARD
======================================================== */

.dashboard-card {
    background: rgba(17,19,26,0.92);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 18px;
    margin-bottom: 16px;
}

.card-label {
    color: #7d8291;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1.7px;
    margin-bottom: 14px;
}

.document-name {
    color: #f0f1f4;
    font-size: 15px;
    font-weight: 700;
}

.document-description {
    color: #777c89;
    font-size: 11px;
    margin-top: 5px;
}

.metric-row {
    display: flex;
    gap: 9px;
    margin-top: 17px;
}

.metric {
    flex: 1;
    background: #0d0f14;
    border: 1px solid rgba(255,255,255,0.04);
    border-radius: 10px;
    padding: 11px;
    text-align: center;
}

.metric-number {
    color: #a5b4fc;
    font-size: 20px;
    font-weight: 800;
}

.metric-label {
    color: #666b79;
    font-size: 9px;
    font-weight: 700;
    margin-top: 3px;
    letter-spacing: 0.8px;
}


/* ========================================================
   CHAT
======================================================== */

.welcome-card {
    background:
        linear-gradient(
            145deg,
            rgba(99,102,241,0.09),
            rgba(255,255,255,0.025)
        );

    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 17px;
    padding: 21px;
    margin-top: 28px;
    margin-bottom: 20px;
}

.ai-label {
    color: #a5b4fc;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1.5px;
    margin-bottom: 11px;
}

.answer {
    color: #e4e5ea;
    font-size: 14px;
    line-height: 1.75;
}

.user-message {
    background: #171922;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 15px;
    padding: 15px 18px;
    margin: 18px 0 12px 12%;
    color: #e7e8ed;
    font-size: 14px;
    line-height: 1.6;
}

.ai-message {
    background:
        linear-gradient(
            145deg,
            rgba(99,102,241,0.08),
            rgba(255,255,255,0.025)
        );

    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 17px;
    padding: 20px;
    margin: 10px 0 20px 0;
}


/* ========================================================
   SUGGESTIONS
======================================================== */

.section-label {
    color: #73798a;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1.8px;
    margin-top: 20px;
    margin-bottom: 10px;
}

.suggestion-card {
    background: #11131a;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 13px;
    padding: 15px;
    min-height: 90px;
}

.suggestion-title {
    color: #e1e2e8;
    font-size: 12px;
    font-weight: 700;
}

.suggestion-description {
    color: #6f7482;
    font-size: 10px;
    line-height: 1.5;
    margin-top: 6px;
}


/* ========================================================
   RAG PIPELINE
======================================================== */

.flow-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 5px 0;
}

.flow-number {
    width: 30px;
    height: 30px;
    min-width: 30px;
    border-radius: 9px;
    background: #1c1f29;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #a5b4fc;
    font-size: 9px;
    font-weight: 800;
}

.flow-number-last {
    background: rgba(99,102,241,0.15);
}

.flow-title {
    color: #e1e2e8;
    font-size: 11px;
    font-weight: 600;
}

.flow-description {
    color: #686d7b;
    font-size: 9px;
    margin-top: 2px;
}

.flow-arrow {
    color: #414552;
    font-size: 13px;
    margin: 2px 0 2px 10px;
}


/* ========================================================
   STATUS
======================================================== */

.status-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}

.status-row:last-child {
    border-bottom: none;
}

.status-name {
    color: #9da1ad;
    font-size: 11px;
}

.status-ready {
    color: #6ee7b7;
    font-size: 9px;
    font-weight: 800;
}


/* ========================================================
   BUTTONS
======================================================== */

.stButton > button {
    background: #11131a;
    border: 1px solid rgba(255,255,255,0.07);
    color: #cfd2dc;
    border-radius: 10px;
    font-size: 11px;
    min-height: 38px;
}

.stButton > button:hover {
    background: #171923;
    border-color: rgba(129,140,248,0.4);
    color: white;
}


/* ========================================================
   FILE UPLOADER
======================================================== */

[data-testid="stFileUploader"] {
    background: #11131a;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 13px;
    padding: 8px;
}

[data-testid="stFileUploader"] section {
    background: transparent;
}

[data-testid="stFileUploaderDropzone"] {
    background: #0d0f14;
    border: 1px dashed rgba(165,180,252,0.25);
    border-radius: 10px;
}


/* ========================================================
   CHAT INPUT
======================================================== */

div[data-testid="stChatInput"] {
    margin-top: 20px;
}

div[data-testid="stChatInput"] textarea {
    background: #11131a !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    color: #f5f5f7 !important;
    border-radius: 15px !important;
}

div[data-testid="stChatInput"] textarea:focus {
    border-color: rgba(129,140,248,0.6) !important;
    box-shadow: 0 0 0 1px rgba(129,140,248,0.15) !important;
}


/* ========================================================
   SCROLLBAR
======================================================== */

::-webkit-scrollbar {
    width: 6px;
}

::-webkit-scrollbar-track {
    background: #08090d;
}

::-webkit-scrollbar-thumb {
    background: #292c36;
    border-radius: 10px;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "active_document" not in st.session_state:
    existing_pdfs = list(DOCUMENTS_PATH.glob("*.pdf"))

    if existing_pdfs:
        st.session_state.active_document = existing_pdfs[0]
    else:
        st.session_state.active_document = None

if "rag_version" not in st.session_state:
    st.session_state.rag_version = 0


# ============================================================
# RAG SYSTEM
# ============================================================

@st.cache_resource
def load_rag(pdf_path, rag_version):
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )

    chunks = splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_PATH),
    )

    llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0,
    )

    return vector_store, llm, pages, chunks


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">
            <span class="brand-icon">◈</span> NAMMAWEB AI
        </div>

        <div class="brand-sub">
            Retrieval Intelligence Workspace
        </div>
        """,
        unsafe_allow_html=True,
    )


    # ========================================================
    # DOCUMENT MANAGER
    # ========================================================

    st.markdown(
        '<div class="side-title">DOCUMENT MANAGER</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Add or replace PDF",
        type=["pdf"],
        help="Upload a PDF to use as the active knowledge base.",
    )

    if uploaded_file is not None:

        current_document_name = (
            st.session_state.active_document.name
            if st.session_state.active_document
            else None
        )

        if uploaded_file.name != current_document_name:

            new_document_path = DOCUMENTS_PATH / uploaded_file.name

            with open(new_document_path, "wb") as file:
                file.write(uploaded_file.getbuffer())

            # Remove old PDF files
            for old_pdf in DOCUMENTS_PATH.glob("*.pdf"):
                if old_pdf != new_document_path:
                    try:
                        old_pdf.unlink()
                    except Exception:
                        pass

            # Remove old Chroma database
            if CHROMA_PATH.exists():
                shutil.rmtree(CHROMA_PATH, ignore_errors=True)

            st.session_state.active_document = new_document_path
            st.session_state.messages = []
            st.session_state.rag_version += 1

            st.success(
                f"Active document changed to {uploaded_file.name}"
            )

            st.rerun()


    # ========================================================
    # LOAD ACTIVE DOCUMENT
    # ========================================================

    if st.session_state.active_document is not None:

        active_pdf = st.session_state.active_document

        with st.spinner("Loading document..."):

            vector_store, llm, pages, chunks = load_rag(
                str(active_pdf),
                st.session_state.rag_version,
            )

    else:

        vector_store = None
        llm = None
        pages = []
        chunks = []

        st.info(
            "Upload a PDF above to create your knowledge base."
        )


    # ========================================================
    # NEW CONVERSATION
    # ========================================================

    if st.button(
        "+ New conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()


    # ========================================================
    # WORKSPACE
    # ========================================================

    st.markdown(
        '<div class="side-title">WORKSPACE</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.active_document:

        st.markdown(
            f"""
            <div class="side-document">

                <div class="side-document-name">
                    📄 &nbsp;{st.session_state.active_document.name}
                </div>

                <div class="side-document-info">
                    {len(pages)} pages · {len(chunks)} semantic chunks
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            """
            <div class="side-document">

                <div class="side-document-name">
                    No document selected
                </div>

                <div class="side-document-info">
                    Upload a PDF to begin
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


    # ========================================================
    # AI SYSTEM
    # ========================================================

    st.markdown(
        '<div class="side-title">AI SYSTEM</div>',
        unsafe_allow_html=True,
    )

    system_status = (
        "READY"
        if vector_store is not None
        else "WAITING"
    )

    st.markdown(
        f"""
        <div class="system-card">

            <div class="system-header">

                <div class="system-title">
                    Retrieval Engine
                </div>

                <div class="system-badge">
                    {system_status}
                </div>

            </div>

            <div class="system-description">
                Local document retrieval powered by
                embeddings, ChromaDB and Llama.
            </div>

            <div class="system-online">

                <span class="online-dot"></span>

                {
                    "All systems operational"
                    if vector_store is not None
                    else "Waiting for document"
                }

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# MAIN LAYOUT
# ============================================================

main_col, right_col = st.columns(
    [3.2, 1.15],
    gap="large",
)


# ============================================================
# MAIN CONTENT
# ============================================================

with main_col:

    st.markdown(
        '<div class="eyebrow">DOCUMENT INTELLIGENCE</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="hero-title">Ask your knowledge base.</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.active_document:

        st.markdown(
            f"""
            <div class="hero-subtitle">
                Ask questions about
                <strong>{st.session_state.active_document.name}</strong>
                and get grounded answers powered by local retrieval and Llama.
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            """
            <div class="hero-subtitle">
                Upload a PDF from the sidebar to create your
                document knowledge base.
            </div>
            """,
            unsafe_allow_html=True,
        )


    # ========================================================
    # CHAT HISTORY
    # ========================================================

    if not st.session_state.messages:

        if vector_store is not None:

            st.markdown(
                """
                <div class="welcome-card">

                    <div class="ai-label">
                        NAMMAWEB AI
                    </div>

                    <div class="answer">
                        Your document is loaded and ready.
                        Ask a question and I'll retrieve the most
                        relevant information from the knowledge base.
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

    else:

        for message in st.session_state.messages:

            if message["role"] == "user":

                st.markdown(
                    f"""
                    <div class="user-message">
                        {message["content"]}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            else:

                st.markdown(
                    f"""
                    <div class="ai-message">

                        <div class="ai-label">
                            NAMMAWEB AI
                        </div>

                        <div class="answer">
                            {message["content"]}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )


    # ========================================================
    # SUGGESTIONS
    # ========================================================

    if vector_store is not None:

        st.markdown(
            '<div class="section-label">TRY ASKING</div>',
            unsafe_allow_html=True,
        )

        suggestion_col1, suggestion_col2 = st.columns(2)

        with suggestion_col1:

            st.markdown(
                """
                <div class="suggestion-card">

                    <div class="suggestion-title">
                        📚 Explore the document
                    </div>

                    <div class="suggestion-description">
                        Get an overview of the topics and
                        information covered in the document.
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

        with suggestion_col2:

            st.markdown(
                """
                <div class="suggestion-card">

                    <div class="suggestion-title">
                        🔎 Find specific information
                    </div>

                    <div class="suggestion-description">
                        Ask about a particular concept,
                        program, experiment or implementation.
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


    # ========================================================
    # CHAT INPUT
    # ========================================================

    query = st.chat_input(
        "Ask something about your document..."
    )


    # ========================================================
    # QUERY PROCESSING
    # ========================================================

    if query:

        if vector_store is None:

            st.error(
                "Please upload a PDF before asking a question."
            )

        else:

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": query,
                }
            )

            with st.spinner(
                "Searching the knowledge base..."
            ):

                docs = vector_store.similarity_search(
                    query,
                    k=4,
                )

                context = "\n\n".join(
                    [
                        f"Page {doc.metadata.get('page', 0) + 1}:\n"
                        f"{doc.page_content}"
                        for doc in docs
                    ]
                )

                prompt = f"""
You are NammaWeb AI, a document question-answering assistant.

Answer the user's question using ONLY the supplied document context.

If the answer is not available in the context, clearly say that
the information was not found in the document.

Be concise, professional and accurate.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{query}

ANSWER:
"""

                response = llm.invoke(prompt)

                answer = response.content

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )

            st.rerun()


# ============================================================
# RIGHT DASHBOARD
# ============================================================

with right_col:

    # ========================================================
    # KNOWLEDGE BASE
    # ========================================================

    st.markdown(
        f"""
        <div class="dashboard-card">

            <div class="card-label">
                KNOWLEDGE BASE
            </div>

            <div class="document-name">
                📄 &nbsp;
                {
                    st.session_state.active_document.name
                    if st.session_state.active_document
                    else "No document"
                }
            </div>

            <div class="document-description">
                Active PDF knowledge base
            </div>

            <div class="metric-row">

                <div class="metric">

                    <div class="metric-number">
                        {len(pages)}
                    </div>

                    <div class="metric-label">
                        PAGES
                    </div>

                </div>

                <div class="metric">

                    <div class="metric-number">
                        {len(chunks)}
                    </div>

                    <div class="metric-label">
                        CHUNKS
                    </div>

                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


    # ========================================================
    # PIPELINE
    # ========================================================

    st.markdown(
        """
        <div class="dashboard-card">

            <div class="card-label">
                RAG PIPELINE
            </div>

            <div class="flow-item">

                <div class="flow-number">
                    01
                </div>

                <div>
                    <div class="flow-title">
                        Document Loader
                    </div>

                    <div class="flow-description">
                        PDF extraction
                    </div>
                </div>

            </div>

            <div class="flow-arrow">
                ↓
            </div>

            <div class="flow-item">

                <div class="flow-number">
                    02
                </div>

                <div>
                    <div class="flow-title">
                        Text Chunking
                    </div>

                    <div class="flow-description">
                        Semantic splitting
                    </div>
                </div>

            </div>

            <div class="flow-arrow">
                ↓
            </div>

            <div class="flow-item">

                <div class="flow-number">
                    03
                </div>

                <div>
                    <div class="flow-title">
                        Embeddings
                    </div>

                    <div class="flow-description">
                        Vector representation
                    </div>
                </div>

            </div>

            <div class="flow-arrow">
                ↓
            </div>

            <div class="flow-item">

                <div class="flow-number">
                    04
                </div>

                <div>
                    <div class="flow-title">
                        ChromaDB
                    </div>

                    <div class="flow-description">
                        Vector retrieval
                    </div>
                </div>

            </div>

            <div class="flow-arrow">
                ↓
            </div>

            <div class="flow-item">

                <div class="flow-number flow-number-last">
                    05
                </div>

                <div>
                    <div class="flow-title">
                        Llama
                    </div>

                    <div class="flow-description">
                        Grounded generation
                    </div>
                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


    # ========================================================
    # SYSTEM STATUS
    # ========================================================

    st.markdown(
        f"""
        <div class="dashboard-card">

            <div class="card-label">
                SYSTEM STATUS
            </div>

            <div class="status-row">

                <div class="status-name">
                    Document Loader
                </div>

                <div class="status-ready">
                    {"READY" if vector_store else "WAITING"}
                </div>

            </div>

            <div class="status-row">

                <div class="status-name">
                    Embedding Model
                </div>

                <div class="status-ready">
                    {"READY" if vector_store else "WAITING"}
                </div>

            </div>

            <div class="status-row">

                <div class="status-name">
                    ChromaDB
                </div>

                <div class="status-ready">
                    {"READY" if vector_store else "WAITING"}
                </div>

            </div>

            <div class="status-row">

                <div class="status-name">
                    Llama 3.2
                </div>

                <div class="status-ready">
                    ONLINE
                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="
        text-align:center;
        color:#555a68;
        font-size:10px;
        margin-top:30px;
        letter-spacing:1px;
    ">
        NAMMAWEB AI · LOCAL RETRIEVAL INTELLIGENCE
    </div>
    """,
    unsafe_allow_html=True,
)