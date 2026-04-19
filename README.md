# Financial Document RAG Pipeline

**Production-ready question-answering system** for complex financial documents using semantic search and LLMs, deployed on AWS.

### Project Goal
Build a multi-document RAG (Retrieval-Augmented Generation) pipeline that lets users ask natural language questions across real financial documents — SEC 10-K filings, Federal Reserve minutes, and IMF reports — and get grounded, cited answers.

### Tech Stack
- **Document Parsing** — PyMuPDF (handles messy financial PDFs)
- **Embeddings** — sentence-transformers (all-MiniLM-L6-v2)
- **Vector Database** — Pinecone Serverless (semantic search)
- **LLM (Dev)** — OpenAI GPT-4o-mini
- **LLM (Prod)** — Amazon Bedrock (Claude Haiku)
- **API** — FastAPI
- **Containerization** — Docker + Amazon ECR
- **Compute** — AWS EC2 (t2.micro)
- **Frontend** — Streamlit

### Project Structure
- `data/raw/` — Source PDF documents (SEC filings, Fed minutes, IMF reports)
- `data/processed/` — Cleaned chunks and metadata after ingestion
- `src/ingestion/` — PDF parsing and chunking scripts
- `src/embeddings/` — Embedding generation and Pinecone upload
- `src/retrieval/` — RAG pipeline and evaluation harness
- `src/api/` — FastAPI application
- `eval/` — Ground-truth Q&A pairs and scoring scripts
- `notebooks/` — Exploration and debugging
- `docker/` — Dockerfile and docker-compose

## LLM Comparison — OpenAI vs Amazon Bedrock

Both models scored 5/5 (100%) on the evaluation harness.

| Metric | OpenAI GPT-4o-mini | Amazon Bedrock Claude Haiku 4.5 |
|---|---|---|
| Eval score | 5/5 (100%) | 5/5 (100%) |
| Answer style | Concise, direct | Detailed, broken down |
| Cost model | Pay per token (OpenAI) | Pay per token (AWS) |
| Data residency | OpenAI servers | Stays within AWS |
| Best for | Development, cost efficiency | Enterprise, compliance |

The pipeline was designed with LLM abstraction in mind — swapping providers 
required changing only the model client, with zero changes to retrieval logic or prompts.

### Progress
- [x] Repository setup and structure
- [x] Document ingestion and chunking (Phase 1)
- [x] Embeddings and vector store (Phase 2)
- [x] RAG pipeline and evaluation (Phase 3)
- [x] FastAPI layer (Phase 4)
- [x] Docker and ECR (Phase 5)
- [x] AWS EC2 deployment (Phase 6)
- [x] Streamlit frontend (Phase 7)
- [ ] Amazon Bedrock swap (Phase 8)

Built by a Data Scientist with 6 years in insurance pricing pivoting to ML Engineering and MLOps.