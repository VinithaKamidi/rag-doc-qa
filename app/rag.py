"""Core RAG logic: load documents -> chunk -> embed -> retrieve -> answer."""
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ---------- 1. Loading ----------
def load_text(path: Path) -> str:
    """Read a .pdf, .txt or .md file and return its text."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    raise ValueError(f"Unsupported file type: {path.name}")


# ---------- 2. Chunking ----------
def chunk_text(text: str, chunk_size: int = 120, overlap: int = 30) -> list[str]:
    """Split text into overlapping chunks of `chunk_size` words."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    words = text.split()
    chunks, step = [], chunk_size - overlap
    for start in range(0, len(words), step):
        piece = words[start : start + chunk_size]
        if piece:
            chunks.append(" ".join(piece))
        if start + chunk_size >= len(words):
            break
    return chunks


@dataclass
class Hit:
    text: str
    source: str
    score: float


# ---------- 3. Retrieval ----------
class Retriever:
    """Semantic search (sentence-transformers + FAISS) with a TF-IDF fallback."""

    def __init__(self, backend: str = "auto"):
        self.backend = backend
        self.chunks: list[str] = []
        self.sources: list[str] = []

    def _pick_backend(self) -> str:
        if self.backend != "auto":
            return self.backend
        try:
            import faiss  # noqa: F401
            import sentence_transformers  # noqa: F401

            return "semantic"
        except ImportError:
            return "tfidf"

    def build(self, docs: dict[str, str], chunk_size: int = 120, overlap: int = 30):
        """docs maps filename -> full text."""
        self.chunks, self.sources = [], []
        for name, text in docs.items():
            for chunk in chunk_text(text, chunk_size, overlap):
                self.chunks.append(chunk)
                self.sources.append(name)
        if not self.chunks:
            raise ValueError("No text found in the documents.")

        self.backend = self._pick_backend()
        if self.backend == "semantic":
            import faiss
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            emb = self.model.encode(self.chunks, normalize_embeddings=True)
            emb = np.asarray(emb, dtype="float32")
            self.index = faiss.IndexFlatIP(emb.shape[1])  # cosine via inner product
            self.index.add(emb)
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self.vectorizer = TfidfVectorizer(stop_words="english")
            self.matrix = self.vectorizer.fit_transform(self.chunks)

    def search(self, query: str, top_k: int = 3) -> list[Hit]:
        top_k = min(top_k, len(self.chunks))
        if self.backend == "semantic":
            q = self.model.encode([query], normalize_embeddings=True)
            scores, ids = self.index.search(np.asarray(q, dtype="float32"), top_k)
            pairs = list(zip(ids[0], scores[0]))
        else:
            q = self.vectorizer.transform([query])
            scores = (self.matrix @ q.T).toarray().ravel()
            ids = np.argsort(scores)[::-1][:top_k]
            pairs = [(i, scores[i]) for i in ids]
        return [Hit(self.chunks[i], self.sources[i], float(s)) for i, s in pairs]


# ---------- 4. Answering ----------
def build_prompt(question: str, hits: list[Hit]) -> str:
    context = "\n\n".join(f"[{i + 1}] ({h.source}) {h.text}" for i, h in enumerate(hits))
    return (
        "Answer the question using ONLY the context below. "
        "Cite sources like [1]. If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )


def answer(question: str, hits: list[Hit]) -> str:
    """Use an LLM if ANTHROPIC_API_KEY is set; otherwise return the best passage."""
    if not hits or hits[0].score <= 0:
        return "I couldn't find anything relevant in the documents."
    if os.getenv("ANTHROPIC_API_KEY"):
        import anthropic

        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=os.getenv("LLM_MODEL", "claude-sonnet-5-5"),
            max_tokens=500,
            messages=[{"role": "user", "content": build_prompt(question, hits)}],
        )
        return msg.content[0].text
    return f"(No LLM key set; showing best matching passage) {hits[0].text}"
