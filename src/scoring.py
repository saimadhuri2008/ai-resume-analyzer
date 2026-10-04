"""Score how well retrieved resume context matches a job description, using an LLM."""
import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",  # Free tier: 1000 requests/day (gemini-flash-latest currently
                                     # resolves to a newer model capped at only 20/day)
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

PROMPT_TEMPLATE = """You are evaluating how well a candidate matches a job, based only on the resume excerpts provided below.

JOB DESCRIPTION:
{job_description}

MOST RELEVANT RESUME EXCERPTS (retrieved by semantic search):
{context}

Respond in exactly this format:
SCORE: <a number from 0 to 100>
REASON: <one or two sentences explaining the score, referencing specific evidence from the excerpts>
"""


def score_match(job_description: str, context_chunks: list[str]) -> dict:
    context = "\n\n".join(context_chunks)
    prompt = PROMPT_TEMPLATE.format(job_description=job_description, context=context)

    response = llm.invoke(prompt)
    return _parse_response(_extract_text(response.content))


def _extract_text(content) -> str:
    """response.content can be a plain string OR a list of content blocks, depending
    on the model. Normalize either shape into a single string."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and "text" in block:
            parts.append(block["text"])
    return "".join(parts)


def _parse_response(text: str) -> dict:
    score = None
    reason = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("SCORE:"):
            digits = "".join(c for c in stripped if c.isdigit())
            score = int(digits) if digits else None
        elif stripped.upper().startswith("REASON:"):
            reason = stripped.split(":", 1)[1].strip()
    return {"score": score, "reason": reason, "raw": text}


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from chunking import chunk_text
    from parser import load_document
    from vectorstore import build_index, search

    default_path = str(Path(__file__).resolve().parent.parent / "data" / "sample_resumes" / "sample_resume.txt")
    resume_path = sys.argv[1] if len(sys.argv) > 1 else default_path

    job_description = (
        "Looking for a backend engineer with strong Python and REST API experience, "
        "ideally with FastAPI and some database performance tuning background."
    )

    text = load_document(resume_path)
    chunks = chunk_text(text)
    index = build_index(chunks)

    top_matches = search(index, job_description, k=2)
    context_chunks = [doc.page_content for doc, _score in top_matches]

    result = score_match(job_description, context_chunks)

    print(f"Job description: {job_description}\n")
    print(f"Score: {result['score']}/100")
    print(f"Reason: {result['reason']}")
