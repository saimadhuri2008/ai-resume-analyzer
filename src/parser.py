"""Load resume and job description text from plain text or PDF files."""
from pathlib import Path

from pypdf import PdfReader


def load_document(path: str) -> str:
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        reader = PdfReader(str(p))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return p.read_text(encoding="utf-8")


if __name__ == "__main__":
    import sys

    # Defaults to the generic sample resume that ships with this repo.
    # Pass any resume file (PDF or .txt) as an argument to parse a different one:
    #   python parser.py "C:\path\to\any_resume.pdf"
    default_path = str(Path(__file__).resolve().parent.parent / "data" / "sample_resumes" / "sample_resume.txt")
    test_path = sys.argv[1] if len(sys.argv) > 1 else default_path

    text = load_document(test_path)
    print(f"Loaded {len(text)} characters from {test_path}\n")
    print("--- First 500 characters ---")
    print(text[:500])
