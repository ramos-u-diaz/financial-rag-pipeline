import os
import json
import boto3
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

load_dotenv()

# Initialize everything
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

# Initialize Bedrock client
# Notice we're using boto3 — AWS's Python SDK
# Same library you'd use for S3, EC2, anything AWS
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-2",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)

MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def retrieve_relevant_chunks(query, top_k=5):
    """
    Exact same retrieval function as rag.py
    Retrieval doesn't change — only the LLM changes
    """
    query_vector = embedding_model.encode(query).tolist()

    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    chunks = []
    for match in results.matches:
        chunks.append({
            "text": match.metadata["text"],
            "source": match.metadata["source"],
            "page_number": match.metadata["page_number"],
            "similarity_score": round(match.score, 3)
        })

    return chunks


def build_prompt(query, chunks):
    """
    Exact same prompt as rag.py
    """
    context_pieces = []
    for i, chunk in enumerate(chunks):
        context_pieces.append(
            f"[Source {i+1}: {chunk['source']}, Page {chunk['page_number']}]\n{chunk['text']}"
        )
    context = "\n\n".join(context_pieces)

    prompt = f"""You are a financial analyst assistant helping users understand complex financial documents.

Answer the question based ONLY on the context provided below.
If the answer is not in the context, say "I don't have enough information in the provided documents to answer that."
Always cite which document and page number your answer comes from.

Context:
{context}

Question: {query}

Answer:"""

    return prompt


def ask(query):
    """
    Same structure as rag.py but calls Bedrock instead of OpenAI.
    This is the only part that changes — everything else is identical.
    """
    print(f"\nQuestion: {query}")
    print("Retrieving relevant chunks...")

    # Step 1 — retrieve (identical to rag.py)
    chunks = retrieve_relevant_chunks(query, top_k=5)
    print(f"Found {len(chunks)} relevant chunks")

    # Step 2 — build prompt (identical to rag.py)
    prompt = build_prompt(query, chunks)

    # Step 3 — call Bedrock instead of OpenAI
    # This is the key difference — the API format is different
    # but the concept is identical
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
        "sources": [
            {
                "source": chunk["source"],
                "page_number": chunk["page_number"],
                "similarity_score": chunk["similarity_score"]
            }
            for chunk in chunks
        ]
    }


if __name__ == "__main__":
    # Quick test
    result = ask("What were Apple's total net sales in 2025?")
    print("\n" + "="*60)
    print(f"ANSWER:\n{result['answer']}")
    print(f"\nSOURCES:")
    for source in result['sources']:
        print(f"  - {source['source']}, page {source['page_number']}")
    print("="*60)