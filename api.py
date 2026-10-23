"""FastAPI service: loads documents from ./data at startup and answers questions."""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.rag import Retriever, answer, load_text

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

app = FastAPI(title="Document Q&A Assistant")
retriever = Retriever()


class Question(BaseModel):
    question: str = Field(min_length=3)
    top_k: int = Field(default=3, ge=1, le=10)


@app.on_event("startup")
def load_documents():
    docs = {}
    for path in sorted(DATA_DIR.glob("*")):
        if path.suffix.lower() in {".pdf", ".txt", ".md"}:
            docs[path.name] = load_text(path)
    if docs:
        retriever.build(docs)


@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexed": len(retriever.chunks), "backend": retriever.backend}


@app.post("/ask")
def ask(q: Question):
    if not retriever.chunks:
        raise HTTPException(status_code=503, detail="No documents indexed. Add files to ./data and restart.")
    hits = retriever.search(q.question, q.top_k)
    return {
        "answer": answer(q.question, hits),
        "sources": [{"file": h.source, "score": round(h.score, 3), "passage": h.text} for h in hits],
    }
