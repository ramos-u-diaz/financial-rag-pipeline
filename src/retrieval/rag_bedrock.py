import os
import json
import boto3
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from pinecone import Pinecone

load_dotenv()

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')    # ← new
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-2",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)

MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def retrieve_relevant_chunks(query, top_k=10, company=None):
    """
    Retrieves top_k candidates from Pinecone.
    We retrieve 10 instead of 5 — reranker cuts it down to 5.
    """
    query_vector = embedding_model.encode(query).tolist()

    pinecone_filter = {"company": {"$eq": company}} if company else None

    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        filter=pinecone_filter
    )

    chunks = []
    for match in results.matches:
        chunks.append({
            "text": match.metadata["text"],
            "source": match.metadata["source"],
            "page_number": match.metadata["page_number"],
            "company": match.metadata["company"],
            "similarity_score": round(match.score, 3)
        })

    return chunks


def rerank_chunks(query, chunks, top_n=5):
    """
    Re-scores Pinecone candidates using a cross-encoder.
    Query and chunk are fed in together so attention runs across both.
    Returns top_n chunks sorted by rerank score.
    """
    pairs = [[query, chunk["text"]] for chunk in chunks]
    scores = reranker.predict(pairs)

    for i, chunk in enumerate(chunks):
        chunk["rerank_score"] = round(float(scores[i]), 4)

    reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_n]


def build_prompt(query, chunks):
    """
    Same domain-aware prompt as rag.py.
    """
    context_pieces = []
    for i, chunk in enumerate(chunks):
        context_pieces.append(
            f"[Source {i+1}: {chunk['source']}, Page {chunk['page_number']}]\n{chunk['text']}"
        )
    context = "\n\n".join(context_pieces)

    prompt = f"""You are a financial analyst assistant helping users understand SEC 10-K annual filings.

The documents in this system are 10-K filings from:
- Apple (technology company) — reports "net sales" split into Products and Services segments
- Allstate (insurance company) — uses insurance terminology like "combined ratio", "net premiums written", "loss ratio", and "underwriting income"
- Progressive (insurance company) — also an insurer, uses similar terminology to Allstate

Important: Insurance companies measure profitability differently than technology companies.
A "combined ratio" below 100% means the insurer is profitable on underwriting.
"Net premiums written" is the insurance equivalent of revenue.

Answer the question based ONLY on the context provided below.
If the answer is not in the context, say "I don't have enough information in the provided documents to answer that."
Always cite which document and page number your answer comes from.
Be concise and precise — this is a financial analysis tool, not a conversation.

Context:
{context}

Question: {query}

Answer:"""

    return prompt


def ask(query, company=None):
    """
    Main function — retrieve, rerank, generate with Bedrock.
    """
    print(f"\nQuestion: {query}")
    if company:
        print(f"Filtering to: {company}")

    # Step 1 — retrieve 10 candidates from Pinecone
    print("Retrieving candidates from Pinecone...")
    chunks = retrieve_relevant_chunks(query, top_k=10, company=company)
    print(f"Retrieved {len(chunks)} candidates")

    # Step 2 — rerank down to 5
    print("Reranking...")
    chunks = rerank_chunks(query, chunks, top_n=5)
    print(f"Top 5 after reranking:")
    for chunk in chunks:
        print(f"  - {chunk['company']} | {chunk['source']} page {chunk['page_number']} (rerank: {chunk['rerank_score']})")

    # Step 3 — build prompt
    prompt = build_prompt(query, chunks)

    # Step 4 — call Bedrock
    print("Generating answer with Bedrock Claude Haiku...")

    request_body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    })

    response = bedrock.invoke_model(
        modelId=MODEL_ID,
        body=request_body
    )

    response_body = json.loads(response["body"].read())
    answer = response_body["content"][0]["text"]

    return {
        "question": query,
        "answer": answer,
        "company_filter": company,
        "sources": [
            {
                "source": chunk["source"],
                "company": chunk["company"],
                "page_number": chunk["page_number"],
                "similarity_score": chunk["similarity_score"],
                "rerank_score": chunk["rerank_score"]          # ← new
            }
            for chunk in chunks
        ]
    }


if __name__ == "__main__":
    # Test 1 — no filter
    print("\n" + "="*60)
    print("TEST 1: No filter — searching all documents")
    result = ask("What were total net sales or revenue?")
    print(f"\nANSWER:\n{result['answer']}")
    print(f"\nSOURCES:")
    for source in result['sources']:
        print(f"  - {source['company']} | {source['source']}, page {source['page_number']} (rerank: {source['rerank_score']})")

    # Test 2 — Allstate total revenues (previously failing)
    print("\n" + "="*60)
    print("TEST 2: Allstate total revenues (previously failing)")
    result = ask("What were Allstate's total revenues in 2025?", company="Allstate")
    print(f"\nANSWER:\n{result['answer']}")
    print(f"\nSOURCES:")
    for source in result['sources']:
        print(f"  - {source['company']} | {source['source']}, page {source['page_number']} (rerank: {source['rerank_score']})")

    # Test 3 — Progressive combined ratio
    print("\n" + "="*60)
    print("TEST 3: Progressive combined ratio")
    result = ask("What was the combined ratio for Progressive's total underwriting operations in 2025?", company="Progressive")
    print(f"\nANSWER:\n{result['answer']}")
    print(f"\nSOURCES:")
    for source in result['sources']:
        print(f"  - {source['company']} | {source['source']}, page {source['page_number']} (rerank: {source['rerank_score']})")