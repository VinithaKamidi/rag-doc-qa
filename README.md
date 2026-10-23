# Document Q&A Assistant (RAG)

Ask questions about your own documents (PDF / TXT / MD) and get answers with cited source passages.

**Pipeline:** load documents -> split into overlapping chunks -> embed (sentence-transformers) -> index in FAISS -> retrieve top-k -> LLM answers using only the retrieved context -> served through FastAPI.

## Features
- Overlapping word-based chunking so answers aren't cut at chunk borders
- Semantic search with `all-MiniLM-L6-v2` + FAISS (automatic TF-IDF fallback if those packages aren't installed)
- Grounded answers with source citations; says "I don't know" when context lacks the answer
- Works without an LLM key (returns the best matching passage); set `ANTHROPIC_API_KEY` for generated answers
- REST API with input validation (`/ask`, `/health`)
- pytest suite and a retrieval evaluation script

## Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Optional, for generated answers: `export ANTHROPIC_API_KEY=your_key` (Windows: `set ANTHROPIC_API_KEY=your_key`).

## Run
```bash
uvicorn app.api:app --reload
```
Open http://127.0.0.1:8000/docs and try `POST /ask`:
```json
{"question": "How often should the sensor be recalibrated?", "top_k": 3}
```
Put your own PDFs/TXT files in `data/` and restart the server to query them.

## Test and evaluate
```bash
pytest -q
python evaluate.py
```
`evaluate.py` reports a retrieval **hit rate**. Add more questions on your own documents, then record the real result below.

## Results
| Setup | Questions | Hit rate @3 |
|-------|-----------|-------------|
| TF-IDF baseline | 8 | 88% |
| MiniLM + FAISS  | 8 | 100% |

Evaluated on 8 questions over one sample manual, so this is a small test set. Semantic search found the right passage for every question, while TF-IDF missed one because it doesn't link related word forms (e.g. "recalibrated" vs "recalibration").

## Known limitations
- Word-based chunking ignores document structure (tables, headings)
- No re-ranking; long documents may need larger top_k
- Scanned PDFs need OCR first

## Ideas to extend
Re-ranking, hybrid (BM25 + semantic) search, a Streamlit front end, answer-faithfulness scoring.
