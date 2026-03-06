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

# ── LangGraph / LangChain / OpenAI ───────────────────────────────────────────
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# ── Global OCR model (loaded once at startup) ─────────────────────────────────
_gpu = torch.cuda.is_available()
print(f"\nLoading EasyOCR model (first run downloads weights ~100 MB)... [GPU: {'yes' if _gpu else 'no - CPU only'}]")
ocr_model = easyocr.Reader(["en"], gpu=_gpu, verbose=False)
print("EasyOCR ready!\n")


# ─────────────────────────────────────────────────────────────────────────────
# OCR Tool
# ─────────────────────────────────────────────────────────────────────────────

@tool
def ocr_read_document(image_path: str) -> List[Dict[str, Any]]:
    """
    Read an image file using EasyOCR and return all detected text.

    Returns a list of dicts, each containing:
      - 'text'       : recognized text string
      - 'bbox'       : bounding box [x_min, y_min, x_max, y_max] in pixels
      - 'confidence' : float 0-1 (how confident the model is)
    """
    try:
        results = ocr_model.readtext(image_path)

        extracted = []
        for bbox_points, text, confidence in results:
            # bbox_points = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
            xs = [p[0] for p in bbox_points]
            ys = [p[1] for p in bbox_points]
            extracted.append({
                "text":       text,
                "bbox":       [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                "confidence": round(float(confidence), 3),
            })

        return extracted or [{"info": "No text detected in image."}]

    except Exception as e:
        return [{"error": f"OCR failed: {e}"}]


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert document analysis assistant.

You have one tool:
  ocr_read_document – extracts all text + bounding boxes from any image.

Workflow:
1. Call ocr_read_document with the image path to get all text.
2. Reason over the extracted text to answer the user's question accurately.
3. Return a clear, structured answer. Use JSON when extracting specific fields.
4. If some text seems garbled, use context clues to interpret it correctly.
"""

def create_agent():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return create_react_agent(llm, [ocr_read_document])


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

def run_agent(agent, image_path: str, question: str, verbose: bool = True) -> str:
    """Invoke the agent and return the final answer."""
    user_msg = (
        f"Document to analyze: '{image_path}'\n\n"
        f"User's question: {question}"
    )
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ]

    if verbose:
        final_content = ""
        for chunk in agent.stream({"messages": messages}):
            for node, update in chunk.items():
                if node == "tools":
                    for msg in update.get("messages", []):
                        tool_name = getattr(msg, "name", "tool")
                        print(f"\n  [Tool: {tool_name}]")
                        content = str(msg.content)
                        print(f"  {content[:500]}{'...' if len(content) > 500 else ''}")
                elif node == "agent":
                    msgs = update.get("messages", [])
                    if msgs:
                        final_content = msgs[-1].content
        return final_content
    else:
        result = agent.invoke({"messages": messages})
        return result["messages"][-1].content


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
    print("Building agent...")
    agent = create_agent()
    print("Agent ready!")
    print("Type your question. Type 'quit' to exit.")
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

        separator("AGENT RESPONSE")
        try:
            answer = run_agent(agent, image_path, question, verbose=not args.quiet)
            print("\n FINAL ANSWER:\n")
            print(answer)
        except Exception as e:
            print(f"\n[ERROR] {e}")

        separator()
        first = False

        if args.question:
            break


if __name__ == "__main__":
    main()
