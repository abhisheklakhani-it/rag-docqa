# RAG Document QA

Upload documents (`.txt`, `.md`, `.pdf`), ask questions, and get answers grounded in and cited from those documents.

**Stack:** Python · FastAPI · LangChain text splitters · scikit-learn (TF-IDF) · Claude API · Docker · pytest

> 🚧 Work in progress.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env   # optional: add your ANTHROPIC_API_KEY
```
