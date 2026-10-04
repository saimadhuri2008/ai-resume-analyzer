"""Convert text chunks into embedding vectors using Google's Gemini embedding model."""
import os

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

embedder = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed multiple chunks at once (e.g. all chunks of a resume)."""
    return embedder.embed_documents(texts)


def embed_query(text: str) -> list[float]:
    """Embed a single piece of text (e.g. a job description used as a search query)."""
    return embedder.embed_query(text)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from chunking import chunk_text
    from parser import load_document

    default_path = str(Path(__file__).resolve().parent.parent / "data" / "sample_resumes" / "sample_resume.txt")
    test_path = sys.argv[1] if len(sys.argv) > 1 else default_path

    text = load_document(test_path)
    chunks = chunk_text(text)
    vectors = embed_texts(chunks)

    print(f"Embedded {len(chunks)} chunks")
    print(f"Each vector has {len(vectors[0])} dimensions")
    print(f"\nFirst 8 numbers of chunk 1's vector:\n{vectors[0][:8]}")
