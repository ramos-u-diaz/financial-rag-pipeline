import fitz  # PyMuPDF
import os
import json
import boto3
import tempfile
import nltk
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize
from dotenv import load_dotenv

load_dotenv()

# S3 configuration
S3_BUCKET = "financial-rag-documents-rd"
S3_PREFIX = "10k/"


def download_pdfs_from_s3(local_dir="data/raw"):
    """
    Downloads all PDFs from S3 bucket to local directory.
    This replaces manually placing PDFs in data/raw/.
    In production, S3 is the source of truth for documents.
    """
    s3 = boto3.client(
        "s3",
        region_name="us-east-2",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )

    os.makedirs(local_dir, exist_ok=True)

    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_PREFIX)

    if "Contents" not in response:
        print(f"No files found in s3://{S3_BUCKET}/{S3_PREFIX}")
        return []

    downloaded = []
    for obj in response["Contents"]:
        key = obj["Key"]
        if not key.endswith(".pdf"):
            continue

        filename = os.path.basename(key)
        local_path = os.path.join(local_dir, filename)

        print(f"Downloading s3://{S3_BUCKET}/{key} → {local_path}")
        s3.download_file(S3_BUCKET, key, local_path)
        downloaded.append(local_path)

    print(f"Downloaded {len(downloaded)} PDFs from S3")
    return downloaded


def parse_metadata_from_filename(filename):
    """
    Extracts company, doc_type, and year from filename like 'Allstate_10K_2025.pdf'
    """
    name = os.path.splitext(filename)[0]
    parts = name.split("_")

    return {
        "company": parts[0],
        "doc_type": parts[1],
        "year": int(parts[2])
    }


def extract_text_from_pdf(pdf_path):
    """
    Opens a PDF and extracts text page by page.
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


def chunk_text(pages, max_words=800, overlap_sentences=3):
    """
    Sentence-aware chunking — chunks never break mid-sentence.
    Overlap is measured in sentences not words.
    """
    chunks = []

    for page in pages:
        sentences = sent_tokenize(page["text"])

        current_sentences = []
        current_word_count = 0

        for sentence in sentences:
            sentence_words = len(sentence.split())

            if current_word_count + sentence_words > max_words and current_sentences:
                chunks.append({
                    "text": " ".join(current_sentences),
                    "page_number": page["page_number"],
                    "source": page["source"],
                    "company": page["company"],
                    "doc_type": page["doc_type"],
                    "year": page["year"]
                })

                current_sentences = current_sentences[-overlap_sentences:]
                current_word_count = sum(len(s.split()) for s in current_sentences)

            current_sentences.append(sentence)
            current_word_count += sentence_words

        if current_sentences:
            chunks.append({
                "text": " ".join(current_sentences),
                "page_number": page["page_number"],
                "source": page["source"],
                "company": page["company"],
                "doc_type": page["doc_type"],
                "year": page["year"]
            })

    print(f"Created {len(chunks)} chunks")
    return chunks


def save_chunks(chunks, output_path):
    """
    Saves chunks to a JSON file.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(chunks)} chunks to {output_path}")


if __name__ == "__main__":
    print("Step 0: Downloading PDFs from S3...")
    pdf_files = download_pdfs_from_s3("data/raw")

    if not pdf_files:
        print("No PDFs downloaded, exiting.")
        exit(1)

    all_chunks = []

    for pdf_path in pdf_files:
        print(f"\nProcessing {os.path.basename(pdf_path)}...")
        pages = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(pages)
        all_chunks.extend(chunks)

    print(f"\nTotal chunks across all documents: {len(all_chunks)}")

    save_chunks(all_chunks, "data/processed/all_10K_chunks.json")

    print("\n--- Preview of first chunk ---")
    print(f"Company: {all_chunks[0]['company']}")
    print(f"Source: {all_chunks[0]['source']}")
    print(f"Page: {all_chunks[0]['page_number']}")
    print(f"Text preview: {all_chunks[0]['text'][:300]}")