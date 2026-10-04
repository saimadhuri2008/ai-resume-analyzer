"""Split resume/job description text into overlapping chunks for embedding."""
from langchain_text_splitters import RecursiveCharacterTextSplitter

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=75,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def chunk_text(text: str) -> list[str]:
    return [c for c in _splitter.split_text(text) if c.strip()]


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from parser import load_document

    default_path = str(Path(__file__).resolve().parent.parent / "data" / "sample_resumes" / "sample_resume.txt")
    test_path = sys.argv[1] if len(sys.argv) > 1 else default_path

    text = load_document(test_path)
    chunks = chunk_text(text)

    print(f"Split {len(text)} characters into {len(chunks)} chunks\n")
    for i, chunk in enumerate(chunks, start=1):
        print(f"--- Chunk {i} ({len(chunk)} chars) ---")
        print(chunk)
        print()
