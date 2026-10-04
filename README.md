# AI Resume Analyzer

A Retrieval-Augmented Generation (RAG) pipeline that scores how well a resume matches a job description, using semantic search (not keyword matching) plus an LLM-generated 0–100 score with reasoning.

## How it works

1. **Parse** — extract raw text from a resume (PDF or plain text).
2. **Chunk** — split the resume into overlapping text chunks, so retrieval can find the *specific* relevant section instead of matching the whole document at once.
3. **Embed** — convert each chunk into a vector using Google's Gemini embedding model, so text can be compared by meaning, not just keywords.
4. **Index** — store the chunk vectors in a FAISS index for fast similarity search.
5. **Retrieve** — given a job description, find the resume chunks most semantically relevant to it.
6. **Score** — feed the retrieved chunks + job description to Gemini, which returns a 0–100 match score with a grounded explanation (the prompt explicitly constrains it to only use the retrieved excerpts, not general knowledge).

## Status

Core pipeline (steps 1–6 above) is implemented and working end to end, consolidated into a single reusable `pipeline.py` entry point (`analyze_resume()`), which every module below builds on.

- [x] Consolidate the pipeline into a single reusable module
- [x] RAGAS evaluation harness (`eval/ragas_eval.py`) — measures `faithfulness` (does the LLM's reasoning stick to the retrieved resume context, or hallucinate) across a 3-case test set spanning distinct roles (backend, data science, frontend). Runs one case at a time with an explicit delay between calls to respect the Gemini free-tier quota. Current result: **0.83 / 0.60 / 0.75 faithfulness** (avg. ~0.73) across the three cases — the data-scientist case scoring lower is a real signal worth investigating further, not an error.
  - `answer_relevancy` was evaluated and dropped: it requires the LLM to return multiple candidate generations per call, which isn't supported by the Gemini model used here.
- [ ] LangSmith tracing for observability into the retrieval/generation chain

## Project structure

```
src/
  parser.py       # Load resume text from PDF or .txt
  chunking.py     # Split text into overlapping chunks
  embeddings.py   # Gemini embedding model wrapper
  vectorstore.py  # FAISS index build + similarity search
  scoring.py      # RAG: retrieved context -> LLM -> 0-100 score + reasoning
  pipeline.py     # Single entry point: analyze_resume() ties every step above together
eval/
  ragas_eval.py   # RAGAS evaluation harness (faithfulness, answer_relevancy)
data/
  sample_resumes/ # Generic synthetic sample resumes for testing (3 different profiles)
list_models.py    # Utility: list which Gemini models your API key can access
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your own [Google AI Studio API key](https://aistudio.google.com/app/apikey):

```
GOOGLE_API_KEY=your-key-here
```

## Try it

Each module can be run directly to see that step in isolation, against the included generic sample resume:

```bash
python src\parser.py
python src\chunking.py
python src\embeddings.py
python src\vectorstore.py
python src\scoring.py      # runs the full pipeline end to end
```

Pass any resume file path as an argument to test against a different resume:

```bash
python src\scoring.py "C:\path\to\any_resume.pdf"
```
