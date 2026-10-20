"""rag_core.py - "Chat with your PDF": extract -> chunk -> embed -> retrieve -> grounded answer.

Design goals (from the project checklist):
  * No black-box wrappers: similarity is explicit (cosine) and the threshold is ours.
  * Edge cases: API rate limits (retry + backoff), empty input, no relevant context.
"""
from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

import chromadb
import groq
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import APIConnectionError, APITimeoutError, Groq, RateLimitError
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

load_dotenv()
log = logging.getLogger("rag")

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHAT_MODEL = os.getenv("CHAT_MODEL", "openai/gpt-oss-120b")
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.25"))
MAX_QUESTION_CHARS = 2000
NO_ANSWER = "I couldn't find that in the document."

# Rate limits and transient network errors are retried with exponential backoff.
api_retry = retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError)),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)


def get_client() -> Groq:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")
    # max_retries=0: tenacity (above) owns the retry policy so behaviour is explicit.
    return groq.Groq(api_key=key, timeout=30, max_retries=0)


@dataclass
class Source:
    page: int
    text: str
    similarity: float


@dataclass
class Answer:
    text: str
    sources: List[Source] = field(default_factory=list)
    grounded: bool = True  # False when we refused because nothing relevant was found


# ----------------------------------------------------------------- ingestion
def load_pdf(path: str) -> List[tuple]:
    """Return [(page_number, text), ...] for pages that contain text."""
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = " ".join((page.extract_text() or "").split())  # normalise whitespace
        if text:
            pages.append((i, text))
    if not pages:
        raise ValueError("No extractable text found (is this a scanned PDF? Try OCR first).")
    return pages


def chunk_pages(pages, chunk_size: int = 800, overlap: int = 150):
    """Split page text into overlapping chunks, remembering the source page."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    texts, metas = [], []
    for page_no, text in pages:
        for piece in splitter.split_text(text):
            if piece.strip():
                texts.append(piece)
                metas.append({"page": page_no})
    return texts, metas


# ----------------------------------------------------------------- pipeline
class PDFChatbot:
    def __init__(self, client: Optional[Groq] = None):
        self.client = client or get_client()
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self._db = chromadb.EphemeralClient()
        self._col = None

    def _embed(self, texts: List[str]) -> List[List[float]]:
        vectors = self.embedder.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    def _embed_all(self, texts: List[str], batch: int = 64) -> List[List[float]]:
        out: List[List[float]] = []
        for i in range(0, len(texts), batch):
            out.extend(self._embed(texts[i : i + batch]))
        return out

    def ingest(self, pdf_path: str) -> int:
        """Index a PDF. Returns number of chunks stored."""
        texts, metas = chunk_pages(load_pdf(pdf_path))
        embeddings = self._embed_all(texts)
        # hnsw:space=cosine -> Chroma returns distance = 1 - cosine_similarity
        self._col = self._db.create_collection(
            f"pdf_{uuid.uuid4().hex[:12]}", metadata={"hnsw:space": "cosine"}
        )
        self._col.add(
            ids=[f"c{i}" for i in range(len(texts))],
            documents=texts,
            embeddings=embeddings,
            metadatas=metas,
        )
        return len(texts)

    def retrieve(self, question: str, k: int = 4, min_similarity: Optional[float] = None) -> List[Source]:
        threshold = MIN_SIMILARITY if min_similarity is None else min_similarity
        q_emb = self._embed([question])[0]
        res = self._col.query(query_embeddings=[q_emb], n_results=min(k, self._col.count()))
        sources = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            sim = 1.0 - dist
            if sim >= threshold:
                sources.append(Source(page=meta["page"], text=doc, similarity=round(sim, 3)))
        return sources

    @api_retry
    def _complete(self, messages) -> str:
        r = self.client.chat.completions.create(model=CHAT_MODEL, messages=messages, temperature=0)
        return (r.choices[0].message.content or "").strip()

    def ask(self, question: str, k: int = 4, min_similarity: Optional[float] = None) -> Answer:
        # Edge case 1: empty / whitespace / oversized input
        question = (question or "").strip()
        if not question:
            return Answer("Please type a question.", grounded=False)
        if len(question) > MAX_QUESTION_CHARS:
            return Answer(f"Question too long (max {MAX_QUESTION_CHARS} characters).", grounded=False)
        if self._col is None:
            return Answer("Please upload and index a PDF first.", grounded=False)

        # Edge case 2: nothing relevant -> refuse WITHOUT calling the LLM (no hallucination possible)
        sources = self.retrieve(question, k, min_similarity)
        if not sources:
            return Answer(NO_ANSWER, grounded=False)

        context = "\n\n".join(f"[Page {s.page}] {s.text}" for s in sources)
        messages = [
            {"role": "system", "content": (
                "You answer questions strictly using the provided context. "
                "Never use outside knowledge. Cite pages like (p. 3). "
                f"If the context does not contain the answer, reply exactly: {NO_ANSWER}")},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]
        text = self._complete(messages)
        return Answer(text, sources, grounded=(text != NO_ANSWER))
