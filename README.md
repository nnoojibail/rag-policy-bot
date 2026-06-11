# Enterprise Policy Q&A Bot

A RAG (Retrieval-Augmented Generation) pipeline that answers HR, compliance, and ethics questions from enterprise policy documents.

Built for Week 2 of the Mastering Agentic AI course — The Gen Academy / Maven.

## What it does
Ask natural language questions against a corpus of enterprise policy documents and get cited, grounded answers — with appropriate refusals when the answer isn't in the documents.

## Tech Stack
- **LangChain** — RAG pipeline orchestration
- **Nebius Token Factory** — Embeddings (Qwen3-Embedding-8B) and LLM (Llama-3.3-70B)
- **ChromaDB** — Local vector store
- **Python / uv** — Runtime and package management

## Corpus
4 publicly available enterprise policy documents (96 pages, 509 chunks):
- SUNY Research Foundation Employee Handbook
- GitLab Product Handbook
- Google Code of Conduct
- Model Employee Handbook Template

## How to run

### 1. Install dependencies
uv sync

### 2. Set up environment
Create a `.env` file with:NEBIUS_API_KEY=

### 3. Ingest documents
Add PDFs to the `/docs` folder, then run:

### 4. Embed and store
uv run ingest.py

### 5. Query
uv run embed.py

## Evaluation
15 questions tested across Easy, Medium, Edge Case, and Unanswerable categories.
- Faithfulness: 100%
- Relevance: 73%
- Appropriate refusal: 100%

See the [full evaluation report](https://docs.google.com/document/d/1YyfptY0NUilQgmeM6737h6VOQ7rR66M8ccEhLJ4_xVE/edit?usp=sharing) for failure analysis and learnings.
