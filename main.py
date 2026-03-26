from dotenv import load_dotenv
load_dotenv()

import torch
import easyocr
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# 1. Initialize OCR (GPU if available)
_gpu = torch.cuda.is_available()
print(f"Loading EasyOCR... [GPU: {'yes' if _gpu else 'no - CPU'}]")
reader = easyocr.Reader(['en'], gpu=_gpu, verbose=False)
print("EasyOCR ready!\n")

# 2. Get image path and run OCR ONCE upfront
image_path = input("Image path: ").strip().strip("\"'")

print("Running OCR...")
results = reader.readtext(image_path)
ocr_text = "\n".join([item[1] for item in results])
print(f"Done — {len(results)} text regions found.\n")

# 3. Single LLM call per question (no ReAct loop)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

while True:
    question = input("Question (or 'quit'): ").strip()
    if question.lower() in ("quit", "exit", "q", "bye", ""):
        print("Goodbye!")
        break

    messages = [
        SystemMessage(content=(
            "You are an expert document analysis assistant. "
            "Answer the user's question accurately based only on the provided OCR text. "
            "If some text seems garbled, use context clues to interpret it correctly."
        )),
        HumanMessage(content=f"OCR extracted text:\n\n{ocr_text}\n\nQuestion: {question}")
    ]

    response = llm.invoke(messages)
    print("\nAnswer:", response.content, "\n")
