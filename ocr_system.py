"""
OCR Document Processing System
================================
EasyOCR + OpenAI GPT-4o + LangGraph Agent
General Purpose - Works with any image

Usage:
    uv run ocr_system.py                           # Interactive mode
    uv run ocr_system.py invoice.png               # Process specific image
    uv run ocr_system.py invoice.png -q "Total?"   # One-shot question
    uv run ocr_system.py receipt.jpg --quiet       # Hide agent reasoning
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

# ── LangChain / OpenAI ───────────────────────────────────────────────────────
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

# ── Global OCR model (loaded once at startup) ─────────────────────────────────
_gpu = torch.cuda.is_available()
print(f"\nLoading EasyOCR model (first run downloads weights ~100 MB)... [GPU: {'yes' if _gpu else 'no - CPU only'}]")
ocr_model = easyocr.Reader(["en"], gpu=_gpu, verbose=False)
print("EasyOCR ready!\n")


# ─────────────────────────────────────────────────────────────────────────────
# OCR (direct — no LangChain tool wrapper needed)
# ─────────────────────────────────────────────────────────────────────────────

# Cache: image_path → extracted text string (avoids re-running OCR on same image)
_ocr_cache: Dict[str, str] = {}

def run_ocr(image_path: str) -> str:
    """Run EasyOCR on image_path and return extracted text. Results are cached."""
    if image_path in _ocr_cache:
        return _ocr_cache[image_path]
    results = ocr_model.readtext(image_path)
    text = "\n".join([item[1] for item in results]) if results else "No text detected."
    _ocr_cache[image_path] = text
    return text


# ─────────────────────────────────────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert document analysis assistant.
Answer the user's question accurately based only on the provided OCR text.
Return a clear, structured answer. Use JSON when extracting specific fields.
If some text seems garbled, use context clues to interpret it correctly."""

def create_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_banner():
    print("=" * 60)
    print("   OCR DOCUMENT PROCESSING SYSTEM")
    print("   EasyOCR  +  GPT-4o-mini  +  LangGraph")
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
        description="General-purpose OCR: EasyOCR + GPT-4o",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("image",      nargs="?", help="Path to image file")
    parser.add_argument("--question", "-q",      help="Question (non-interactive)")
    parser.add_argument("--quiet",    "-s",      action="store_true",
                        help="Hide tool calls, show only final answer")
    args = parser.parse_args()

    print_banner()

    # ── API key check ─────────────────────────────────────────────────────────
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[ERROR] OPENAI_API_KEY not set.")
        print("  Edit the '.env' file and add:  OPENAI_API_KEY=sk-...")
        sys.exit(1)

    # ── Get image path ────────────────────────────────────────────────────────
    image_path = args.image
    if not image_path:
        image_path = input("\nEnter image path (e.g. invoice.png): ").strip()

    image_path = image_path.strip("\"'")

    if not Path(image_path).exists():
        print(f"\n[ERROR] File not found: '{image_path}'")
        sys.exit(1)

    print(f"\nDocument : {Path(image_path).resolve()}")

    # ── Run OCR once upfront ───────────────────────────────────────────────────
    print("Running OCR...")
    ocr_text = run_ocr(image_path)
    print(f"OCR done — {len(ocr_text.splitlines())} text regions extracted.")

    llm = create_llm()
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
