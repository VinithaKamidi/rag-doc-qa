from pathlib import Path

import pytest

from app.rag import Retriever, chunk_text, load_text

DATA = Path(__file__).resolve().parent.parent / "data" / "sensor_manual.txt"


def test_chunk_text_overlap_and_size():
    text = " ".join(str(i) for i in range(100))
    chunks = chunk_text(text, chunk_size=20, overlap=5)
    assert all(len(c.split()) <= 20 for c in chunks)
    # consecutive chunks share 5 words
    assert chunks[0].split()[-5:] == chunks[1].split()[:5]


def test_chunk_text_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_text("a b c", chunk_size=5, overlap=5)


@pytest.fixture(scope="module")
def retriever():
    r = Retriever(backend="tfidf")  # fast, no model download needed for tests
    r.build({"sensor_manual.txt": load_text(DATA)}, chunk_size=60, overlap=15)
    return r


@pytest.mark.parametrize(
    "question, expected_word",
    [
        ("What torque should I use when installing the sensor?", "25"),
        ("How often is recalibration recommended?", "12 months"),
        ("How long is the warranty?", "24 months"),
        ("What is the supply voltage range?", "10 and 30"),
    ],
)
def test_retrieval_finds_right_passage(retriever, question, expected_word):
    hits = retriever.search(question, top_k=3)
    assert any(expected_word in h.text for h in hits)


def test_unknown_topic_scores_low(retriever):
    hits = retriever.search("quantum chromodynamics", top_k=1)
    assert hits[0].score == 0
