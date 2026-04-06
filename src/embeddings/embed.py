import json
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

def load_chunks(chunks_path):
    """
    Loads the chunks we saved in Phase 1.
    """
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks from {chunks_path}")
    return chunks


def setup_pinecone_index(pc, index_name):
    """
    Creates a Pinecone index if it doesn't exist yet.
    Think of an index like a table in a database — it's where 
    all your vectors get stored and searched.
    """
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if index_name not in existing_indexes:
        print(f"Creating new Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=384,      # must match our embedding model output size
            metric="cosine",    # how we measure similarity between vectors
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print("Index created successfully")
    else:
        print(f"Index '{index_name}' already exists, skipping creation")

    return pc.Index(index_name)


def generate_and_upload_embeddings(chunks, index):
    """
    Converts each chunk to a vector and uploads to Pinecone.
    We do this in batches so we don't overwhelm the API.
    """
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded")

    # Process in batches of 32
    batch_size = 32
    total_uploaded = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        # Extract just the text from each chunk for embedding
        texts = [chunk['text'] for chunk in batch]

        # Convert texts to vectors
        embeddings = model.encode(texts)

        # Package everything up for Pinecone
        # Each record needs: a unique ID, the vector, and metadata
        vectors = []
        for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
            vectors.append({
                "id": f"chunk_{i + j}",
                "values": embedding.tolist(),
                "metadata": {
                    "text": chunk['text'],
                    "page_number": chunk['page_number'],
                    "source": chunk['source']
                }
            })

        # Upload this batch to Pinecone
        index.upsert(vectors=vectors)
        total_uploaded += len(vectors)
        print(f"Uploaded {total_uploaded}/{len(chunks)} chunks...")

    print(f"\nDone! {total_uploaded} chunks uploaded to Pinecone")


if __name__ == "__main__":
    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME")

    print("Step 1: Loading chunks...")
    chunks = load_chunks("data/processed/Apple_10K_2025_chunks.json")

    print("\nStep 2: Setting up Pinecone index...")
    index = setup_pinecone_index(pc, index_name)

    print("\nStep 3: Generating embeddings and uploading to Pinecone...")
    generate_and_upload_embeddings(chunks, index)