"""
OCR Document Processing System
================================
EasyOCR + OpenAI GPT-4o + LangGraph Agent
General Purpose - Works with images, PDFs, DOCX, TXT, MD, and more

Usage:
    uv run ocr_system.py                              # Interactive mode
    uv run ocr_system.py invoice.png                  # Process image
    uv run ocr_system.py report.pdf -q "Summary?"     # PDF one-shot
    uv run ocr_system.py notes.txt -q "Key points?"   # Text file
    uv run ocr_system.py doc.docx --quiet             # DOCX quiet mode
"""

import os
import sys
import argparse
from typing import List, Dict, Any
from pathlib import Path

from dotenv import load_dotenv

# ── Load .env for API keys ────────────────────────────────────────────────────
load_dotenv()

# ── EasyOCR ───────────────────────────────────────────────────────────────────
import easyocr
import torch

# ── LangChain / LLM providers ────────────────────────────────────────────────
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# ── Universal document loader ─────────────────────────────────────────────────
from document_loader import extract_text_from_path

# ── Global OCR model (loaded once at startup) ─────────────────────────────────
_gpu = torch.cuda.is_available()
print(f"\nLoading EasyOCR model (first run downloads weights ~100 MB)... [GPU: {'yes' if _gpu else 'no - CPU only'}]")
ocr_model = easyocr.Reader(["en"], gpu=_gpu, verbose=False)
print("EasyOCR ready!\n")


# ─────────────────────────────────────────────────────────────────────────────
# Document text extraction (cached per path)
# ─────────────────────────────────────────────────────────────────────────────

_doc_cache: Dict[str, str] = {}

def extract_document(file_path: str) -> str:
    """Extract text from any supported document type. Results are cached."""
    if file_path in _doc_cache:
        return _doc_cache[file_path]
    text = extract_text_from_path(file_path, ocr_model)
    _doc_cache[file_path] = text
    return text


# ─────────────────────────────────────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert document analysis assistant.
Answer the user's question accurately based only on the provided OCR text.
Return a clear, structured answer. Use JSON when extracting specific fields.
If some text seems garbled, use context clues to interpret it correctly."""

def create_llm(provider: str = "openai"):
    if provider == "gemini":
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_banner(provider: str = "openai"):
    model = "Gemini 2.0 Flash" if provider == "gemini" else "GPT-4o-mini"
    print("=" * 60)
    print("   DOCUMENT PROCESSING SYSTEM")
    print(f"   EasyOCR  +  {model}  |  PDF · DOCX · TXT · MD · Images")
    print("=" * 60)

def separator(title: str = ""):
    if title:
        pad = max((58 - len(title)) // 2, 1)
        print(f"\n{'─'*pad} {title} {'─'*pad}\n")
    else:
        print("─" * 60)

def run_query(llm, ocr_text: str, question: str, verbose: bool = True) -> str:
    """Send OCR text + question to LLM in a single call and return the answer."""
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"OCR extracted text:\n\n{ocr_text}\n\nQuestion: {question}"),
    ]

    if verbose:
        # Stream tokens for perceived speed
        answer = ""
        for chunk in llm.stream(messages):
            token = chunk.content
            print(token, end="", flush=True)
            answer += token
        print()
        return answer
    else:
        return llm.invoke(messages).content


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="General-purpose document Q&A: images, PDFs, DOCX, TXT, MD, …",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("document",   nargs="?", help="Path to any document file")
    parser.add_argument("--question", "-q",      help="Question (non-interactive)")
    parser.add_argument("--quiet",    "-s",      action="store_true",
                        help="Hide tool calls, show only final answer")
    parser.add_argument("--provider", "-p",      choices=["openai", "gemini"],
                        default="openai",
                        help="LLM provider: openai (default) or gemini")
    args = parser.parse_args()

    print_banner(args.provider)

    # ── API key check ─────────────────────────────────────────────────────────
    if args.provider == "gemini":
        if not os.getenv("GOOGLE_API_KEY"):
            print("\n[ERROR] GOOGLE_API_KEY not set.")
            print("  Edit the '.env' file and add:  GOOGLE_API_KEY=AIza...")
            sys.exit(1)
    else:
        if not os.getenv("OPENAI_API_KEY"):
            print("\n[ERROR] OPENAI_API_KEY not set.")
            print("  Edit the '.env' file and add:  OPENAI_API_KEY=sk-...")
            sys.exit(1)

    # ── Get document path ─────────────────────────────────────────────────────
    doc_path = args.document
    if not doc_path:
        doc_path = input("\nEnter document path (image, PDF, DOCX, TXT, MD, …): ").strip()

    doc_path = doc_path.strip("\"'")

    if not Path(doc_path).exists():
        print(f"\n[ERROR] File not found: '{doc_path}'")
        sys.exit(1)

    print(f"\nDocument : {Path(doc_path).resolve()}")

    # ── Extract text (OCR for images/scanned PDFs, direct for text files) ─────
    print("Extracting text...")
    ocr_text = extract_document(doc_path)
    print(f"Done — {len(ocr_text.splitlines())} lines extracted.")

    llm = create_llm(args.provider)
    print("LLM ready! Type your question. Type 'quit' to exit.")
    separator()

    # ── Question loop ─────────────────────────────────────────────────────────
    first = True
    while True:
        if args.question and first:
            question = args.question
        else:
            try:
                question = input("\nYour question: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

        if question.lower() in ("quit", "exit", "q", "bye"):
            print("Goodbye!")
            break

        if not question:
            continue

        separator("ANSWER")
        try:
            answer = run_query(llm, ocr_text, question, verbose=not args.quiet)
            if args.quiet:
                print(answer)
        except Exception as e:
            print(f"\n[ERROR] {e}")

        separator()
        first = False

        if args.question:
            break


if __name__ == "__main__":
    main()
