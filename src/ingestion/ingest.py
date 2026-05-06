import fitz  # PyMuPDF
import os
import json
from dotenv import load_dotenv

load_dotenv()

def parse_metadata_from_filename(filename):
    """
    Extracts company, doc_type, and year from a filename like 'Allstate_10K_2025.pdf'
    This metadata gets attached to every chunk and stored in Pinecone for filtering.
    """
    name = os.path.splitext(filename)[0]  # strip .pdf
    parts = name.split("_")              # ['Allstate', '10K', '2025']

    return {
        "company": parts[0],             # 'Allstate'
        "doc_type": parts[1],            # '10K'
        "year": int(parts[2])            # 2025
    }


def extract_text_from_pdf(pdf_path):
    """
    Opens a PDF and extracts text page by page.
    Returns a list of dictionaries, one per page.
    """
    doc = fitz.open(pdf_path)
    filename = os.path.basename(pdf_path)
    metadata = parse_metadata_from_filename(filename)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()

        if len(text.strip()) < 50:
            continue

        pages.append({
            "text": text,
            "page_number": page_num + 1,
            "source": filename,
            "company": metadata["company"],
            "doc_type": metadata["doc_type"],
            "year": metadata["year"]
        })

    print(f"Extracted {len(pages)} pages from {filename}")
    return pages


def chunk_text(pages, chunk_size=800, overlap=80):
    """
    Takes pages and splits them into smaller chunks.
    Each chunk keeps track of where it came from.
    """
    chunks = []

    for page in pages:
        text = page["text"]
        words = text.split()

        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunk_text_str = " ".join(chunk_words)

            chunks.append({
                "text": chunk_text_str,
                "page_number": page["page_number"],
                "source": page["source"],
                "company": page["company"],
                "doc_type": page["doc_type"],
                "year": page["year"]
            })

            start += chunk_size - overlap

    print(f"Created {len(chunks)} chunks")
    return chunks


def save_chunks(chunks, output_path):
    """
    Saves chunks to a JSON file so other scripts can pick them up.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(chunks)} chunks to {output_path}")


if __name__ == "__main__":
    pdf_files = [
        "data/raw/Apple_10K_2025.pdf",
        "data/raw/Allstate_10K_2025.pdf",
        "data/raw/Progressive_10K_2025.pdf"
    ]

    all_chunks = []

    for pdf_path in pdf_files:
        print(f"\nProcessing {os.path.basename(pdf_path)}...")
        pages = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(pages)
        all_chunks.extend(chunks)

    print(f"\nTotal chunks across all documents: {len(all_chunks)}")

    save_chunks(all_chunks, "data/processed/all_10K_chunks.json")

    # Preview
    print("\n--- Preview of first chunk ---")
    print(f"Company: {all_chunks[0]['company']}")
    print(f"Source: {all_chunks[0]['source']}")
    print(f"Page: {all_chunks[0]['page_number']}")
    print(f"Text preview: {all_chunks[0]['text'][:300]}")