---
title: Course Advisor Agent
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: AI-powered course advisor using RAG + PydanticAI + Groq
---

# Course Advisor Agent

A production-grade RAG-powered course advisor built with:

- **LlamaIndex** — Hybrid retrieval (Dense + BM25 + optional Reranker)
- **PydanticAI** — Type-safe agent framework with tool calling
- **Groq** — Fast LLM inference (llama-3.3-70b-versatile)
- **FastAPI** — Async REST API

## Architecture

```
User Query
    ↓
PydanticAI Agent (Groq LLM)
    ↓ calls tools
search_knowledge → HybridRetriever (Dense + BM25 + RRF) → Reranker
get_course_by_name → CourseRepository (exact/fuzzy)
    ↓
Synthesised Answer
```

## API

### POST /chat
```json
{ "message": "What courses do you have for beginners in data science?" }
```
Response:
```json
{ "response": "Kayfa offers several beginner-friendly data science courses..." }
```

### GET /health
```json
{ "status": "ok", "app": "Course Advisor Agent" }
```

## Local Development

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/course-advisor-agent
cd course-advisor-agent
pip install -e ".[dev]"

# 2. Set environment
cp .env.example .env
# Edit .env: set GROQ_API_KEY

# 3. Run
uvicorn app.main:app --reload --port 7860
```

## Deployment

This project auto-deploys to HuggingFace Spaces via GitHub Actions on every push to `main`.

**Required GitHub Secrets:**

| Secret | Description |
|--------|-------------|
| `GROQ_API_KEY` | Groq API key |
| `HF_TOKEN` | HuggingFace token with write access |
| `HF_SPACE_ID` | `owner/space-name` e.g. `OmarFadlallah/course-advisor` |

**Required HuggingFace Space Secrets:**

| Secret | Description |
|--------|-------------|
| `GROQ_API_KEY` | Groq API key (same as GitHub) |

## Configuration

All configuration via environment variables (see `.env.example`).
Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | required | Groq API key |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | HuggingFace embedding model |
| `VECTOR_STORE_BACKEND` | `simple` | `simple` / `chroma` / `qdrant` |
| `INDEX_STORAGE_PATH` | `/data/index` | Persisted index location |
| `RERANKER_BACKEND` | `""` | `""` / `bge` / `cohere` |
| `IS_HF_SPACES` | `true` | Auto-set in Docker; forces SimpleVectorStore |
