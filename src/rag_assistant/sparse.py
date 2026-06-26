"""Sparse lexical retrieval via BM25 (Okapi).

Dense vectors capture meaning but miss exact terms — names, codes, rare keywords. BM25 is the
classic lexical complement: it rewards documents that contain the query's terms, weighted by how
rare each term is (IDF) and how often it appears (saturating TF). Fusing it with dense retrieval
(see `retrieval.py`) is why "hybrid" beats either alone.

Pure-Python and in-memory — no extra services. For a production Postgres deployment you'd use
`tsvector`/`ts_rank` instead; the fusion logic downstream is identical.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from rag_assistant.models import Chunk, RetrievedChunk

_TOKEN = re.compile(r"[a-z0-9]+")
_K1 = 1.5  # term-frequency saturation
_B = 0.75  # length normalization


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    """An in-memory BM25 index over chunks."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._tokens: list[list[str]] = []
        self._doc_freq: Counter[str] = Counter()
        self._avgdl: float = 0.0

    def add(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            toks = _tokenize(chunk.text)
            self._chunks.append(chunk)
            self._tokens.append(toks)
            for term in set(toks):
                self._doc_freq[term] += 1
        lengths = [len(t) for t in self._tokens]
        self._avgdl = sum(lengths) / len(lengths) if lengths else 0.0

    def _idf(self, term: str) -> float:
        n = len(self._chunks)
        df = self._doc_freq.get(term, 0)
        # BM25 idf with +1 smoothing to keep it non-negative.
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        if not self._chunks:
            return []
        q_terms = _tokenize(query)
        scores: list[float] = []
        for toks in self._tokens:
            tf = Counter(toks)
            dl = len(toks)
            score = 0.0
            for term in q_terms:
                if term not in tf:
                    continue
                freq = tf[term]
                denom = freq + _K1 * (1 - _B + _B * dl / (self._avgdl or 1))
                score += self._idf(term) * (freq * (_K1 + 1)) / denom
            scores.append(score)
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [
            RetrievedChunk(chunk=self._chunks[i], score=scores[i], source="sparse")
            for i in ranked
            if scores[i] > 0
        ]

    def __len__(self) -> int:
        return len(self._chunks)
