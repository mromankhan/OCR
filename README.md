  ---
  What Is This Project?

  An AI-powered OCR Document Analysis Agent built as a lab exercise series (Labs 1-3).

  It combines:
  - EasyOCR — extracts text + bounding boxes from any image
  - OpenAI GPT-4o — reasons over the extracted text to answer questions
  - LangGraph (ReAct agent) — orchestrates the tool-calling loop between OCR and the LLM

  Two scripts:

  ┌───────────────┬───────────────────────────────────────────────────────────────────────────┐
  │     File      │                                Description                                │
  ├───────────────┼───────────────────────────────────────────────────────────────────────────┤
  │ main.py       │ Simple version — interactive prompts, GPT-4o-mini                         │
  ├───────────────┼───────────────────────────────────────────────────────────────────────────┤
  │ ocr_system.py │ Full-featured CLI — args, streaming verbose output, question loop, GPT-4o │
  └───────────────┴───────────────────────────────────────────────────────────────────────────┘

  The workflow:
  1. You give it an image (invoice, receipt, medical report, etc.)
  2. The agent calls the ocr_read_document tool → EasyOCR extracts raw text
  3. The LLM reasons over the text → answers your question

  ---
  How to Test It

  Prerequisites:

  1. Set up your OpenAI API key — create a .env file in the project root:
  OPENAI_API_KEY=sk-...
  2. Install dependencies (uses uv):
  uv sync

  Run the full-featured version:
  # Interactive mode (it asks for image path & questions in a loop)
  uv run ocr_system.py

  # Pass image directly, ask one question
  uv run ocr_system.py Medical_report.jpg -q "What is the patient's name?"

  # Quiet mode (hide tool calls, show only final answer)
  uv run ocr_system.py Medical_report.jpg -q "Summarize this report" --quiet

  Run the simple version:
  uv run main.py
  # Then enter: Medical_report.jpg
  # Then enter your question

  There's already a test image included: Medical_report.jpg — use that to test immediately without needing your own image.

  ---
  Key requirement: A valid OPENAI_API_KEY — without it, the script will exit with an error on startup.


  # Run streamlit App 

  uv run streamlit run app.py