# FastAPI · OpenAI · Pinecone — Proposal Generator

A multi-tenant backend that writes tailored freelance proposals from a job post. It pulls in your most relevant past work, client feedback and winning proposals before it writes anything. Built with **FastAPI**, **OpenAI** (GPT-4o and `text-embedding-3-large`), **Pinecone** for vector search and **MongoDB** for storage, with a lightweight Tailwind front end.

---

## How it works

```
Job post ──► Requirement extraction (OpenAI function calling)
              │
              ▼
        Retrieval pipeline
         1. Metadata filter   – industry, skills, urgency, task type
         2. Semantic search   – Pinecone (3072-dim embeddings)
         3. Feedback analysis – patterns from top-rated past projects
              │
              ▼
        Prompt engine + hook strategy (tone-matched opening lines)
              │
              ▼
        GPT-4o generation (standard or streaming) ──► quality score + suggestions
```

## Features

- **Proposal generation**: standard and **streaming** (SSE) endpoints. Each draft is grounded in retrieved portfolio items, links and client feedback.
- **Structured requirement extraction**: turns a job post into typed requirements. A `do_not_assume` field keeps the model from inventing details.
- **Three-stage retrieval**: metadata filtering, then semantic vector search, then feedback-driven re-ranking.
- **Hook strategy engine**: varies the opening lines to match the client's tone and urgency.
- **Similar past proposal finder**: cosine-similarity lookup over a historical job→proposal corpus. Embeddings are cached and only recomputed for new or changed rows.
- **Portfolio management**: CRUD for past projects, with one Pinecone vector per item and batch embedding of pending items.
- **Outcome tracking and analytics**: records sent proposals and their outcomes (viewed / hired), with a conversion funnel and an AI-vs-manual comparison.
- **Prompt Lab**: try custom prompt instructions per section, preview the assembled prompt, and save templates.
- **Multi-tenant auth**: API-key roles (`super_admin`, `admin`, `member`), with every query scoped by organization.
- **Image feedback OCR**: pulls text from client-review screenshots via GPT-4o vision.
- **Rate limiting** (SlowAPI), configurable CORS, and a Dockerized production build.

## Tech stack

| Layer | Tools |
|---|---|
| API | Python 3.12, FastAPI, Uvicorn, Pydantic |
| AI | OpenAI GPT-4o, text-embedding-3-large, tiktoken |
| Vector DB | Pinecone (serverless) |
| Database | MongoDB / AWS DocumentDB (PyMongo) |
| Front end | HTML, Tailwind CSS (CDN), Axios |
| Infra | Docker (multi-stage, non-root), docker-compose, AWS ECS Fargate |

## Project structure

```
├── main.py                    # FastAPI app, routers, CORS, rate limit, static front end
├── app/
│   ├── config.py              # Environment-driven settings
│   ├── middleware/auth.py     # API-key auth + role/tenant context
│   ├── routes/                # proposals, jobs, portfolio, profiles, analytics, admin, lab, data entry
│   ├── services/              # business logic (proposal, embedding, corpus, analytics, ingestion)
│   ├── utils/                 # retrieval pipeline, prompt engine, hook strategy, OpenAI/Pinecone clients
│   ├── infra/mongodb/         # connection + repository layer
│   └── models/                # Pydantic schemas
├── frontend/                  # generator, history, analytics, admin, lab, training pages
├── data/proposal_corpus.json  # sample job→proposal corpus (replace with your own)
└── test/                      # retrieval, extraction and generation tests
```

## Getting started

### 1. Requirements
- Python 3.12+
- A MongoDB instance (local or Atlas)
- An OpenAI API key and a Pinecone index with **dimension 3072**

### 2. Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# fill in OPENAI_API_KEY, PINECONE_*, MONGODB_URI, ADMIN_API_KEY
```

### 4. Run

```bash
uvicorn main:app --reload --port 8000
```

- API docs: http://localhost:8000/docs
- Front end: http://localhost:8000/frontend/

For local development, point `frontend/js/config.js` at `http://localhost:8000`.

### Docker

```bash
docker build -t proposal-generator .
docker run --env-file .env -p 8000:8000 proposal-generator
# or
docker compose -f docker-compose.prod.yml up --build
```

## Key endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/auth/verify` | Validate API key, return role and tenant |
| POST | `/api/proposals/generate-stream` | Generate a proposal (streaming) |
| POST | `/api/proposals/similar-job` | Find the closest past job→proposal |
| POST | `/api/jobs/ingest` · `/ingest/bulk` | Ingest past jobs into the portfolio |
| GET/POST/PUT/DELETE | `/api/portfolio` | Manage portfolio items |
| POST | `/api/portfolio/embed-pending` | Batch-embed items into Pinecone |
| GET | `/api/analytics/...` | Conversion funnel and trends |
| — | `/api/profiles`, `/api/admin`, `/api/lab` | Profiles, key management, Prompt Lab |

The full interactive reference is at `/docs`.

## Sample data

`data/proposal_corpus.json` ships with a few **fictional sample records** that show the schema. Add your own job→proposal history in the same format. On first load the service embeds any new rows and caches the vectors in `data/proposal_corpus.embeddings.json`. If a `proposal_corpus` collection exists in MongoDB, it is used instead of the file.

## Tests

```bash
python test/test_semantic_industry_detection.py
python test/test_extraction_generalization.py
python test/test_proposal_generation.py      # requires OpenAI + Pinecone credentials
```
