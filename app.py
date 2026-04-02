"""
Streamlit UI for OCR Document Analysis System
Run: streamlit run app.py
"""

import os
import torch
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from document_loader import extract_text, IMAGE_EXTS

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OCR Document Analyzer",
    page_icon="🔍",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert document analysis assistant.
Answer the user's question accurately based only on the provided document text.
Return a clear, structured answer. Use JSON when extracting specific fields.
If some text seems garbled, use context clues to interpret it correctly."""

SAMPLE_IMAGE = Path(__file__).parent / "Medical_report.jpg"

# All accepted upload extensions
ACCEPTED_TYPES = ["jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp",
                  "pdf", "docx", "txt", "md", "markdown",
                  "csv", "json", "xml", "html", "htm", "rst", "log"]


# ── Load EasyOCR once (cached across reruns) ──────────────────────────────────
@st.cache_resource(show_spinner="Loading EasyOCR model...")
def load_ocr_model():
    import easyocr
    gpu = torch.cuda.is_available()
    reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)
    return reader, gpu


@st.cache_resource(show_spinner=False)
def load_llm(provider: str):
    if provider == "gemini":
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
    return ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)


# ── Universal document extractor (cached per file bytes) ──────────────────────
@st.cache_data(show_spinner="Extracting text from document...")
def process_document(_ocr_model, file_bytes: bytes, filename: str) -> str:
    return extract_text(file_bytes, filename, _ocr_model)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    provider = st.selectbox(
        "LLM Provider",
        options=["openai", "gemini"],
        format_func=lambda x: "OpenAI (GPT-4o-mini)" if x == "openai" else "Google Gemini 2.0 Flash",
    )
    st.session_state["provider"] = provider

    st.divider()

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            api_key = st.text_input("OpenAI API Key", type="password",
                                    placeholder="sk-...")
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
        else:
            st.success("OpenAI API key loaded from .env")
    else:
        api_key = os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            api_key = st.text_input("Google API Key", type="password",
                                    placeholder="AIza...")
            if api_key:
                os.environ["GOOGLE_API_KEY"] = api_key
        else:
            st.success("Google API key loaded from .env")

    st.divider()

    ocr_model, gpu_available = load_ocr_model()
    hw = "GPU" if gpu_available else "CPU"
    st.info(f"EasyOCR running on **{hw}**")
    st.caption("EasyOCR is used for images and scanned PDF pages.")

    st.divider()
    st.markdown("**Supported formats**")
    st.caption("Images: JPG, PNG, BMP, TIFF, WEBP")
    st.caption("Documents: PDF, DOCX")
    st.caption("Text: TXT, MD, CSV, JSON, XML, HTML, RST")

    st.divider()
    if st.button("🗑️ Clear chat history"):
        st.session_state.messages = []
        st.rerun()

    if st.button("🔄 Reset (new document)"):
        for key in ["ocr_text", "messages", "image_name"]:
            st.session_state.pop(key, None)
        st.rerun()


# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("🔍 Document Analyzer")
_model_label = "Gemini 2.0 Flash" if st.session_state.get("provider") == "gemini" else "GPT-4o-mini"
st.caption(f"Upload any document → extract text → ask questions using {_model_label}")

# ── Session state init ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ocr_text" not in st.session_state:
    st.session_state.ocr_text = None
if "image_name" not in st.session_state:
    st.session_state.image_name = None

# ── File upload ───────────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    uploaded = st.file_uploader(
        "Upload document (image, PDF, DOCX, TXT, MD, CSV, JSON, …)",
        type=ACCEPTED_TYPES,
    )

with col2:
    st.write("")
    st.write("")
    use_sample = st.button("📄 Use sample Medical_report.jpg",
                           disabled=not SAMPLE_IMAGE.exists())

# ── Determine active file ──────────────────────────────────────────────────────
file_bytes = None
file_name = None

if uploaded:
    file_bytes = uploaded.read()
    file_name = uploaded.name
elif use_sample and SAMPLE_IMAGE.exists():
    file_bytes = SAMPLE_IMAGE.read_bytes()
    file_name = SAMPLE_IMAGE.name

# ── Process new document ───────────────────────────────────────────────────────
if file_bytes and file_name != st.session_state.image_name:
    st.session_state.messages = []
    st.session_state.image_name = file_name
    st.session_state.ocr_text = process_document(ocr_model, file_bytes, file_name)

# ── Show preview + extracted text ─────────────────────────────────────────────
if st.session_state.ocr_text:
    ext = Path(st.session_state.image_name).suffix.lower() if st.session_state.image_name else ""
    is_image = ext in IMAGE_EXTS

    if is_image:
        col_img, col_text = st.columns([1, 1])
        with col_img:
            st.subheader("📷 Preview")
            if file_bytes:
                st.image(file_bytes, use_container_width=True)
            elif SAMPLE_IMAGE.exists():
                st.image(str(SAMPLE_IMAGE), use_container_width=True)
        with col_text:
            st.subheader("📝 Extracted Text")
            lines = st.session_state.ocr_text.splitlines()
            st.caption(f"{len(lines)} lines extracted")
            with st.expander("View extracted text", expanded=True):
                st.text(st.session_state.ocr_text)
    else:
        st.subheader("📝 Extracted Text")
        lines = st.session_state.ocr_text.splitlines()
        st.caption(f"{len(lines)} lines extracted from **{st.session_state.image_name}**")
        with st.expander("View extracted text", expanded=True):
            st.text(st.session_state.ocr_text)

    st.divider()

    # ── Chat interface ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("💬 Ask Questions")

    # Render previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    _provider = st.session_state.get("provider", "openai")
    _key_env = "GOOGLE_API_KEY" if _provider == "gemini" else "OPENAI_API_KEY"
    _key_name = "Google API key" if _provider == "gemini" else "OpenAI API key"

    if not os.getenv(_key_env):
        st.warning(f"Enter your {_key_name} in the sidebar to ask questions.")
    else:
        if question := st.chat_input("Ask anything about this document..."):
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                llm = load_llm(_provider)
                messages = [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=(
                        f"Document text:\n\n{st.session_state.ocr_text}"
                        f"\n\nQuestion: {question}"
                    )),
                ]
                response = st.write_stream(
                    chunk.content
                    for chunk in llm.stream(messages)
                    if chunk.content
                )

            st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.info("👆 Upload a document (image, PDF, DOCX, TXT, MD, …) or click 'Use sample Medical_report.jpg' to get started.")
