from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from pypdf import PdfReader


load_dotenv()


PDF_PATH = Path(__file__).resolve().parent / "pythonBOOK.pdf"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "pythonBOOK"


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def extract_chunks(pdf_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    reader = PdfReader(str(pdf_path))
    texts: list[str] = []
    metadatas: list[dict[str, str]] = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        for index, chunk in enumerate(chunk_text(page_text), start=1):
            texts.append(chunk)
            metadatas.append(
                {
                    "page_label": str(page_number),
                    "source": pdf_path.name,
                    "chunk": str(index),
                }
            )

    return texts, metadatas


def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
    texts, metadatas = extract_chunks(PDF_PATH)

    if not texts:
        raise ValueError(f"No text could be extracted from {PDF_PATH.name}")

    QdrantVectorStore.from_texts(
        texts=texts,
        metadatas=metadatas,
        embedding=embedding_model,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
    )

    print(
        f"Loaded {len(texts)} chunks from {PDF_PATH.name} into Qdrant collection '{COLLECTION_NAME}'."
    )


if __name__ == "__main__":
    main()