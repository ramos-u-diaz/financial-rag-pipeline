import os
import json
import boto3
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

load_dotenv()

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-2",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)

MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def retrieve_relevant_chunks(query, top_k=5, company=None):
    """
    Same retrieval as rag.py with optional company filter.
    Pass company="Allstate" to restrict search to Allstate chunks only.
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
            "company": match.metadata["company"],          # ← new
            "similarity_score": round(match.score, 3)
        })

    return chunks


def build_prompt(query, chunks):
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
    Same structure as rag.py but calls Bedrock instead of OpenAI.
    company parameter optionally restricts search to one company.
    """
    print(f"\nQuestion: {query}")
    if company:
        print(f"Filtering to: {company}")
    print("Retrieving relevant chunks...")

    # Step 1 — retrieve
    chunks = retrieve_relevant_chunks(query, top_k=5, company=company)
    print(f"Found {len(chunks)} relevant chunks")
    for chunk in chunks:
        print(f"  - {chunk['company']} | {chunk['source']} page {chunk['page_number']} (similarity: {chunk['similarity_score']})")

    # Step 2 — build prompt
    prompt = build_prompt(query, chunks)

    # Step 3 — call Bedrock
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
                "similarity_score": chunk["similarity_score"]
            }
            for chunk in chunks
        ]
    }


if __name__ == "__main__":
    # Test 1 — no filter, searches all three companies
    print("\n" + "="*60)
    print("TEST 1: No filter — searching all documents")
    result = ask("What were total net sales or revenue?")
    print(f"\nANSWER:\n{result['answer']}")
    print(f"\nSOURCES:")
    for source in result['sources']:
        print(f"  - {source['company']} | {source['source']}, page {source['page_number']} (score: {source['similarity_score']})")

    # Test 2 — filtered to Allstate only
    print("\n" + "="*60)
    print("TEST 2: Filtered to Allstate only")
    result = ask("What were the main risk factors?", company="Allstate")
    print(f"\nANSWER:\n{result['answer']}")
    print(f"\nSOURCES:")
    for source in result['sources']:
        print(f"  - {source['company']} | {source['source']}, page {source['page_number']} (score: {source['similarity_score']})")

    # Test 3 — filtered to Progressive only
    print("\n" + "="*60)
    print("TEST 3: Filtered to Progressive only")
    result = ask("What was net income?", company="Progressive")
    print(f"\nANSWER:\n{result['answer']}")
    print(f"\nSOURCES:")
    for source in result['sources']:
        print(f"  - {source['company']} | {source['source']}, page {source['page_number']} (score: {source['similarity_score']})")