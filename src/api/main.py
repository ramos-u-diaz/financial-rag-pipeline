from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import sys
import os

# This lets us import from src/retrieval/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.rag import ask

load_dotenv()

# Initialize the FastAPI app
# This is the equivalent of creating a Flask app if you've seen that before
app = FastAPI(
    title="Financial RAG Pipeline",
    description="Question answering system for financial documents",
    version="1.0.0"
)


# Pydantic models define the shape of requests and responses
# FastAPI uses these to automatically validate incoming data
# If someone sends a request without a question field, FastAPI rejects it automatically
class QueryRequest(BaseModel):
    question: str


class SourceModel(BaseModel):
    source: str
    page_number: float
    similarity_score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceModel]


# Health check endpoint
# This is standard in every production API
# AWS, Docker, and load balancers all ping this to know if your app is alive
@app.get("/health")
def health_check():
    return {"status": "healthy"}


# Main query endpoint
# POST because we're sending data (the question) not just requesting data
@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Takes a question, runs it through the RAG pipeline,
    returns an answer with citations.
    """
    # Basic validation — don't process empty questions
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    try:
        result = ask(request.question)
        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            sources=[
                SourceModel(
                    source=s["source"],
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