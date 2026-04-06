import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from openai import OpenAI

load_dotenv()

# Initialize everything once when the module loads
# This is important — loading the embedding model takes a few seconds
# so we do it once, not every time a question is asked
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
openai_client = OpenAI()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))


def retrieve_relevant_chunks(query, top_k=5):
    """
    Takes a user question and finds the most relevant chunks in Pinecone.
    
    top_k=5 means we retrieve the 5 most similar chunks.
    This is standard in RAG pipelines — retrieve a few, use the best ones.
    """
    # Convert the question to a vector using the same model we used for documents
    query_vector = embedding_model.encode(query).tolist()

    # Ask Pinecone for the most similar chunks
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True  # this gives us back the text and source info
    )

    # Extract the useful parts from Pinecone's response
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
    Builds the prompt we send to GPT.
    
    This is the core of RAG — we're giving the LLM the retrieved context
    and telling it to answer ONLY from that context.
    This is what prevents hallucinations.
    """
    # Format the retrieved chunks into readable context
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
    The main function — takes a question, returns an answer with citations.
    This is what the FastAPI layer will call in Phase 4.
    """
    print(f"\nQuestion: {query}")
    print("Retrieving relevant chunks...")

    # Step 1 — retrieve
    chunks = retrieve_relevant_chunks(query, top_k=5)
    print(f"Found {len(chunks)} relevant chunks")
    for chunk in chunks:
        print(f"  - {chunk['source']} page {chunk['page_number']} (similarity: {chunk['similarity_score']})")

    # Step 2 — build prompt
    prompt = build_prompt(query, chunks)

    # Step 3 — generate answer
    print("Generating answer...")
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0      # 0 = deterministic, no creativity
                           # we want factual answers, not creative ones
    )

    answer = response.choices[0].message.content

    # Step 4 — package everything up
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
    # Test questions against our Apple 10-K
    test_questions = [
        "What were Apple's total net sales in 2025?",
        "What are the main risk factors Apple identified?",
        "How much did Apple spend on research and development?"
    ]

    for question in test_questions:
        result = ask(question)
        print("\n" + "="*60)
        print(f"ANSWER:\n{result['answer']}")
        print(f"\nSOURCES:")
        for source in result['sources']:
            print(f"  - {source['source']}, page {source['page_number']} (score: {source['similarity_score']})")
        print("="*60)