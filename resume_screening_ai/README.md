# AI Resume Screening System with LangChain and LangSmith

This project implements a modular AI Resume Screening pipeline:

`Resume -> Extract -> Match -> Score -> Explain`

It includes:
- 3 resume samples (`strong`, `average`, `weak`)
- 1 Data Scientist job description
- LangChain LCEL chains with `.invoke()`
- LangSmith tracing support
- Optional debug run with intentionally wrong input

## Project Structure

- `prompts/`: Prompt templates for each stage
- `chains/`: LCEL chain builders for each stage
- `data/`: Job description and resume samples
- `main.py`: Pipeline runner

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file (for real LLM runs):

```env
OPENAI_API_KEY=your_openai_api_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
LANGSMITH_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=resume-screening-ai
```

## Run

Mock mode (works without API keys, good for local validation):

```bash
python main.py --mock --debug
```

Real OpenAI mode:

```bash
python main.py --model gpt-4o-mini --debug
```

Real Hugging Face mode:

```bash
python main.py --provider huggingface --hf-repo-id mistralai/Mistral-7B-Instruct-v0.1 --debug
```

Note: if OpenAI quota is exhausted, OpenAI mode automatically falls back to local mock mode.

## LangSmith Tracing Checklist

For assignment screenshots, ensure:
- At least 3 runs are visible (`strong`, `average`, `weak`)
- Steps are visible with tags (`extract`, `match`, `score`, `explain`)
- One debug run (`debug_incorrect_output`) is present

## Notes

- Prompts explicitly prevent hallucination.
- Outputs are structured JSON and parsed via Pydantic.
- Final ranked results are saved to `outputs/latest_report.json`.

