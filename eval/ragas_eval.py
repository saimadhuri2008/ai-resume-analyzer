"""Evaluate the RAG pipeline's answer quality using RAGAS metrics.

Runs the full pipeline (parser -> chunking -> embeddings -> vectorstore -> scoring)
against a small set of resume/job-description test cases, then scores each one
individually with RAGAS's faithfulness metric.

Note: answer_relevancy was dropped -- it requires the LLM to return multiple
candidate generations per call, which Gemini's API doesn't support here
("Multiple candidates is not enabled for this model"). faithfulness alone
(does the reasoning stick to the retrieved context) is the more important
metric for a RAG system anyway.
"""
import sys
import time
from pathlib import Path

# eval/ is a sibling of src/, not inside it -- add src/ to the import path manually
# so this script can import our own pipeline modules.
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.append(str(SRC_DIR))

from datasets import Dataset
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import faithfulness
from ragas.run_config import RunConfig

from embeddings import embedder
from pipeline import analyze_resume
from scoring import llm

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_resumes"

TEST_CASES = [
    {
        "resume": str(DATA_DIR / "sample_resume.txt"),
        "job_description": (
            "Looking for a backend engineer with strong Python and REST API experience, "
            "ideally with FastAPI and some database performance tuning background."
        ),
    },
    {
        "resume": str(DATA_DIR / "sample_resume_2.txt"),
        "job_description": (
            "Seeking a data scientist experienced in building predictive models with "
            "Python and Scikit-learn, and communicating results via dashboards."
        ),
    },
    {
        "resume": str(DATA_DIR / "sample_resume_3.txt"),
        "job_description": (
            "Hiring a frontend developer skilled in React and TypeScript, with a focus "
            "on performance and accessibility."
        ),
    },
]

# Seconds to wait between test cases. Each case involves several LLM calls
# (embedding, scoring, then RAGAS's own faithfulness decomposition + verification
# calls) -- running cases back-to-back with no gap was still hitting the Gemini
# free-tier rate limit even with max_workers=1, so we space them out explicitly.
DELAY_BETWEEN_CASES = 20


def evaluate_one(case: dict):
    """Run the pipeline on one test case, then score just that one result with RAGAS."""
    result = analyze_resume(case["resume"], case["job_description"])

    dataset = Dataset.from_dict({
        "question": [case["job_description"]],
        "contexts": [result["retrieved_chunks"]],
        "answer": [result["reason"]],
    })

    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embedder)
    run_config = RunConfig(max_workers=1, timeout=300)

    return evaluate(
        dataset,
        metrics=[faithfulness],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=run_config,
    )


if __name__ == "__main__":
    all_scores = []

    for i, case in enumerate(TEST_CASES, start=1):
        print(f"\n--- Test case {i}/{len(TEST_CASES)}: {case['job_description'][:60]}... ---")
        scores = evaluate_one(case)
        print(scores)
        all_scores.append(scores)

        if i < len(TEST_CASES):
            print(f"Waiting {DELAY_BETWEEN_CASES}s before next case (rate limit)...")
            time.sleep(DELAY_BETWEEN_CASES)

    print("\n--- Summary ---")
    for i, scores in enumerate(all_scores, start=1):
        print(f"Case {i}: {scores}")
