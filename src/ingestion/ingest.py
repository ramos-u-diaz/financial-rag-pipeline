import fitz  # PyMuPDF
import os
from dotenv import load_dotenv

load_dotenv()

def extract_text_from_pdf(pdf_path):
    """
    Opens a PDF and extracts text page by page.
    Returns a list of dictionaries, one per page.
    """
    doc = fitz.open(pdf_path)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()

        # Skip pages with barely any text (usually cover pages or images)
        if len(text.strip()) < 50:
            continue

        pages.append({
            "text": text,
            "page_number": page_num + 1,  # humans count from 1, not 0
            "source": os.path.basename(pdf_path)  # just the filename, not full path
        })

    print(f"Extracted {len(pages)} pages from {os.path.basename(pdf_path)}")
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

        # Slide a window of chunk_size words across the page
        # overlap means we repeat some words at boundaries so context isn't lost
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            chunks.append({
                "text": chunk_text,
                "page_number": page["page_number"],
                "source": page["source"]
            })

            # Move forward by chunk_size minus overlap
            start += chunk_size - overlap

    print(f"Created {len(chunks)} chunks total")
    return chunks


if __name__ == "__main__":
    # Run this script directly to test it
    pdf_path = "data/raw/Apple_10K_2025.pdf"

    print("Step 1: Extracting text from PDF...")
    pages = extract_text_from_pdf(pdf_path)

    print("\nStep 2: Chunking text...")
    chunks = chunk_text(pages)

    # Preview the first chunk so we can see what we're working with
    print("\n--- Preview of first chunk ---")
    print(f"Source: {chunks[0]['source']}")
    print(f"Page: {chunks[0]['page_number']}")
    print(f"Text preview: {chunks[0]['text'][:300]}")
    print(f"\nTotal chunks ready for embedding: {len(chunks)}")