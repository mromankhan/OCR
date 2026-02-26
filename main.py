from dotenv import load_dotenv
load_dotenv()

import easyocr
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# 1. Initialize OCR
reader = easyocr.Reader(['en'], gpu=False, verbose=False)

# 2. Create OCR Tool
@tool
def ocr_read_document(image_path: str) -> str:
    """Reads an image from the given path and returns extracted text using OCR."""
    try:
        results = reader.readtext(image_path)
        text = "\n".join([item[1] for item in results])
        return text
    except Exception as e:
        return f"Error reading image: {e}"

# 3. Create the Agent
tools = [ocr_read_document]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=1)

agent = create_react_agent(llm, tools)

# 4. Run the Agent
image_path = input("Image path: ")
question   = input("Question  : ")

task = f"Please process the document at '{image_path}' using the OCR tool and answer: {question}"

response = agent.invoke({
    "messages": [HumanMessage(content=task)]
})

print("\nAnswer:", response["messages"][-1].content)
