from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.rag import ask

load_dotenv()

app = FastAPI(
    title="Financial RAG Pipeline",
    description="Question answering system for SEC 10-K filings",
    version="2.0.0"
)


class QueryRequest(BaseModel):
    question: str
    company: Optional[str] = None    # ← new, optional filter


class SourceModel(BaseModel):
    source: str
    company: str                     # ← new
    page_number: float
    similarity_score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    company_filter: Optional[str]    # ← new, echoes back what filter was used
    sources: list[SourceModel]


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Takes a question and optional company filter, runs it through
    the RAG pipeline, returns an answer with citations.

    company options: "Apple", "Allstate", "Progressive", or omit for all.
    """
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    try:
        result = ask(request.question, company=request.company)
        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            company_filter=result["company_filter"],
            sources=[
                SourceModel(
                    source=s["source"],
                    company=s["company"],
                    page_number=s["page_number"],
                    similarity_score=s["similarity_score"]
                )
                for s in result["sources"]
            ]
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {str(e)}"
        )