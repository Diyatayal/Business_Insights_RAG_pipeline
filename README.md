# 📊 Annual Report Intelligence Chatbot

> AI-powered financial analysis using RAG + Web Search + Groq LLM

---

## Overview

A Streamlit-based chatbot that answers questions about company financials by intelligently searching uploaded annual report PDFs or the web — depending on what the question needs. It uses OpenAI LLM with LangChain `@tool` decorators to decide which source to query automatically.

---

## Features

- RAG over company annual report PDFs using ChromaDB
- Web search fallback via SerpAPI for general or real-time questions
- Automatic tool selection — LLM decides which tool to use
- Sliding window memory — keeps last 10 messages for context


---

## Tech Stack

| Component     | Technology                      |
|---------------|---------------------------------|
| Frontend      | Streamlit                       |
| LLM           | OpenAI (gpt-4.0-mini)           |
| Embeddings    | HuggingFace `all-MiniLM-L6-v2`  |
| Vector Store  | ChromaDB                        |
| Web Search    | SerpAPI via LangChain           |
| Framework     | LangChain Core + LangChain Groq |
| PDF Loading   | LangChain PyPDFLoader           |

---

## Project Structure

```
AI_UseCase/
├── app.py                  # Main Streamlit app
├── config/
│   └── config.py           # API keys and config constants
├── models/
│   ├── llm.py              # Groq LLM setup
│   └── embeddings.py       # HuggingFace embedding model
├── utils/
│   ├── rag_utils.py        # PDF loading, ChromaDB, retrieval
│   └── web_search.py       # SerpAPI web search tool
├── data/                   # Place your PDF annual reports here
└── chroma_db/              # Auto-created vector database
```

---

## Setup & Installation

### 1. Clone and create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your API keys in `config/config.py`

```python
OPENAI_API_KEY="your_openapi_key"
SERP_API_KEY = "your_serpapi_key"
```

### 4. Add annual report PDFs

Place your company PDF annual reports inside the `data/` folder.

### 5. Run the app

```bash
streamlit run app.py
```

---

## How It Works

```
User asks a question
        ↓
Groq LLM decides which tool to use (bind_tools)
        ↓
   ┌────┴────┐
   ↓         ↓
RAG Tool   Web Tool
(ChromaDB) (SerpAPI)
   ↓         ↓
   └────┬────┘
        ↓
Groq LLM generates final cited answer
        ↓
Displayed in Streamlit chat UI
```



