# RAG Customer Support Assistant

A modular RAG project with LangChain, ChromaDB, LangGraph, and optional Human-in-the-Loop escalation.

## Project Structure

```text
rag_customer_support/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── ingestion/
│   │   ├── loader.py
│   │   └── chunking.py
│   ├── embeddings/
│   │   └── embedder.py
│   ├── vectorstore/
│   │   └── chroma_store.py
│   ├── retrieval/
│   │   └── retriever.py
│   ├── llm/
│   │   └── generator.py
│   ├── workflow/
│   │   └── langgraph_flow.py
│   ├── hitl/
│   │   └── human_loop.py
│   └── utils/
│       └── helpers.py
├── data/
├── chroma_db/
├── requirements.txt
└── README.md
```

## Setup (Windows PowerShell)

```powershell
Set-Location "C:\tech dhanu\projects\mini 4th sem\New folder\adv-gen-ai-innomatics\Rag Final Project\rag_customer_support"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set your keys.

## Run

```powershell
python smoke_test.py
streamlit run app/main.py
```

## Notes
- Add one or more PDF files into `data/` before running the app.
- The app ingests PDFs, chunks text, builds ChromaDB index, retrieves context, generates an answer, and escalates to HITL when confidence is low.
- For LangSmith tracing, set:
  - `LANGCHAIN_TRACING_V2=true`
  - `LANGSMITH_TRACING=true`
  - `LANGSMITH_API_KEY=...`
  - `LANGSMITH_PROJECT=rag-customer-support`

