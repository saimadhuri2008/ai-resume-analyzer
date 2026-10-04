"""Store resume chunk embeddings in FAISS and search them by similarity to a query."""
from langchain_community.vectorstores import FAISS

from embeddings import embedder


def build_index(chunks: list[str]) -> FAISS:
    """Embed every chunk and store the vectors in a searchable FAISS index."""
    return FAISS.from_texts(chunks, embedding=embedder)


def search(index: FAISS, query: str, k: int = 3):
    """Return the k chunks most similar to the query, each paired with a distance score."""
    return index.similarity_search_with_score(query, k=k)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from chunking import chunk_text
    from parser import load_document

    default_path = str(Path(__file__).resolve().parent.parent / "data" / "sample_resumes" / "sample_resume.txt")
    test_path = sys.argv[1] if len(sys.argv) > 1 else default_path

    text = load_document(test_path)
    chunks = chunk_text(text)
    index = build_index(chunks)

    query = "Looking for a backend engineer with strong Python and REST API experience."
    results = search(index, query, k=2)

    print(f"Query: {query}\n")
    for chunk, score in results:
        print(f"--- Match (distance score: {score:.4f}) ---")
        print(chunk.page_content)
        print()
