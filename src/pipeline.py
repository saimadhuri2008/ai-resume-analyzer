"""End-to-end pipeline: resume + job description -> match score.

This is the single entry point everything else (a future API, the eval
harness, etc.) should call, instead of each re-implementing the same
load -> chunk -> index -> retrieve -> score sequence.
"""
from pathlib import Path

from chunking import chunk_text
from parser import load_document
from scoring import score_match
from vectorstore import build_index, search


def analyze_resume(resume_path: str, job_description: str, k: int = 3) -> dict:
    text = load_document(resume_path)
    chunks = chunk_text(text)
    index = build_index(chunks)

    top_matches = search(index, job_description, k=k)
    context_chunks = [doc.page_content for doc, _score in top_matches]

    result = score_match(job_description, context_chunks)
    result["retrieved_chunks"] = context_chunks
    return result


if __name__ == "__main__":
    import sys

    default_resume = str(Path(__file__).resolve().parent.parent / "data" / "sample_resumes" / "sample_resume.txt")
    resume_path = sys.argv[1] if len(sys.argv) > 1 else default_resume

    job_description = (
        "Looking for a backend engineer with strong Python and REST API experience, "
        "ideally with FastAPI and some database performance tuning background."
    )

    result = analyze_resume(resume_path, job_description)

    print(f"Job description: {job_description}\n")
    print(f"Score: {result['score']}/100")
    print(f"Reason: {result['reason']}")
