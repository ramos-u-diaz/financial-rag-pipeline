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

### Progress
- [x] Repository setup and structure
- [ ] Document ingestion and chunking (Phase 1)
- [ ] Embeddings and vector store (Phase 2)
- [ ] RAG pipeline and evaluation (Phase 3)
- [ ] FastAPI layer (Phase 4)
- [ ] Docker and ECR (Phase 5)
- [ ] AWS EC2 deployment (Phase 6)
- [ ] Streamlit frontend (Phase 7)
- [ ] Amazon Bedrock swap (Phase 8)

Built by a Data Scientist with 6 years in insurance pricing pivoting to ML Engineering and MLOps.