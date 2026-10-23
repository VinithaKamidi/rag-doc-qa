"""Measure retrieval quality on a small question set.

Hit rate = % of questions where the top-k retrieved chunks contain the expected answer text.
Add your own questions to QUESTIONS (and your own documents to ./data) to get real numbers for your README/resume.
"""
from pathlib import Path

from app.rag import Retriever, load_text

QUESTIONS = [
    ("What torque should be used to tighten the process connection?", "25 Newton"),
    ("How often should the sensor be recalibrated?", "12 months"),
    ("What is the maximum pressure the sensor can measure?", "400 bar"),
    ("What does an output below 3.6 mA mean?", "detected a fault"),
    ("How long is the warranty?", "24 months"),
    ("What is the operating temperature range?", "minus 40"),
    ("What IP rating does the housing have?", "IP67"),
    ("What supply voltage is allowed?", "10 and 30"),
]


def main(top_k: int = 3):
    data = Path(__file__).parent / "data"
    docs = {p.name: load_text(p) for p in data.glob("*") if p.suffix.lower() in {".txt", ".md", ".pdf"}}
    r = Retriever()
    r.build(docs, chunk_size=60, overlap=15)
    hits = 0
    for question, expected in QUESTIONS:
        found = any(expected.lower() in h.text.lower() for h in r.search(question, top_k))
        hits += found
        print(f"{'PASS' if found else 'FAIL'}  {question}")
    print(f"\nBackend: {r.backend} | top_k={top_k} | hit rate: {hits}/{len(QUESTIONS)} = {100 * hits / len(QUESTIONS):.0f}%")


if __name__ == "__main__":
    main()
