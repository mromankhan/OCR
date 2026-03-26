"""
Streamlit UI for OCR Document Analysis System
Run: streamlit run app.py
"""

import os
import torch
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OCR Document Analyzer",
    page_icon="🔍",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert document analysis assistant.
Answer the user's question accurately based only on the provided OCR text.
Return a clear, structured answer. Use JSON when extracting specific fields.
If some text seems garbled, use context clues to interpret it correctly."""

SAMPLE_IMAGE = Path(__file__).parent / "Medical_report.jpg"


# ── Load EasyOCR once (cached across reruns) ──────────────────────────────────
@st.cache_resource(show_spinner="Loading EasyOCR model...")
def load_ocr_model():
    import easyocr
    gpu = torch.cuda.is_available()
    reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)
    return reader, gpu


@st.cache_resource(show_spinner=False)
def load_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)


# ── OCR helper (cached per file bytes so same upload isn't re-processed) ──────
@st.cache_data(show_spinner="Running OCR on image...")
def run_ocr(_model, image_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(image_bytes)
        tmp_path = f.name
    try:
        results = _model.readtext(tmp_path)
        text = "\n".join([item[1] for item in results]) if results else "No text detected."
    finally:
        os.unlink(tmp_path)
    return text


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        api_key = st.text_input("OpenAI API Key", type="password",
                                placeholder="sk-...")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
    else:
        st.success("API key loaded from .env")

    st.divider()

    ocr_model, gpu_available = load_ocr_model()
    hw = "GPU" if gpu_available else "CPU"
    st.info(f"EasyOCR running on **{hw}**")

    st.divider()
    if st.button("🗑️ Clear chat history"):
        st.session_state.messages = []
        st.rerun()

    if st.button("🔄 Reset (new image)"):
        for key in ["ocr_text", "messages", "image_name"]:
            st.session_state.pop(key, None)
        st.rerun()


# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("🔍 OCR Document Analyzer")
st.caption("Upload an image → extract text with EasyOCR → ask questions using GPT-4o-mini")

# ── Session state init ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ocr_text" not in st.session_state:
    st.session_state.ocr_text = None
if "image_name" not in st.session_state:
    st.session_state.image_name = None

# ── Image upload ──────────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    uploaded = st.file_uploader(
        "Upload image (JPG, PNG, BMP, TIFF)",
        type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
    )

with col2:
    st.write("")
    st.write("")
    use_sample = st.button("📄 Use sample Medical_report.jpg",
                           disabled=not SAMPLE_IMAGE.exists())

# ── Determine active image ────────────────────────────────────────────────────
image_bytes = None
image_name = None

if uploaded:
    image_bytes = uploaded.read()
    image_name = uploaded.name
elif use_sample and SAMPLE_IMAGE.exists():
    image_bytes = SAMPLE_IMAGE.read_bytes()
    image_name = SAMPLE_IMAGE.name

# ── Process new image ─────────────────────────────────────────────────────────
if image_bytes and image_name != st.session_state.image_name:
    st.session_state.messages = []          # reset chat on new image
    st.session_state.image_name = image_name
    st.session_state.ocr_text = run_ocr(ocr_model, image_bytes)

# ── Show image + OCR text ─────────────────────────────────────────────────────
if st.session_state.ocr_text:
    col_img, col_text = st.columns([1, 1])

    with col_img:
        st.subheader("📷 Image")
        if image_bytes:
            st.image(image_bytes, use_container_width=True)
        elif SAMPLE_IMAGE.exists():
            st.image(str(SAMPLE_IMAGE), use_container_width=True)

    with col_text:
        st.subheader("📝 Extracted Text")
        lines = st.session_state.ocr_text.splitlines()
        st.caption(f"{len(lines)} text regions found")
        with st.expander("View raw OCR text", expanded=True):
            st.text(st.session_state.ocr_text)

    st.divider()

    # ── Chat interface ────────────────────────────────────────────────────────
    st.subheader("💬 Ask Questions")

    # Render previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("Enter your OpenAI API key in the sidebar to ask questions.")
    else:
        if question := st.chat_input("Ask anything about this document..."):
            # Show user message
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            # Stream assistant response
            with st.chat_message("assistant"):
                llm = load_llm()
                messages = [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=(
                        f"OCR extracted text:\n\n{st.session_state.ocr_text}"
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
    st.info("👆 Upload an image or click 'Use sample Medical_report.jpg' to get started.")
