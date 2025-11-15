from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Iterable

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


@dataclass
class ChunkMeta:
    chunk_id: str
    web_id: str
    url: str
    kind: str
    title: str
    offset_start: int
    offset_end: int


class FAISSRetriever:
    def __init__(self, index_path: str, meta_path: str, model_name: str):
        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        self.chunk_texts: List[str] = meta["chunk_texts"]
        self.metas: List[ChunkMeta] = [ChunkMeta(**m) for m in meta["metas"]]
        self.embedder = SentenceTransformer(model_name)
        self.normalize = True

    @staticmethod
    def load(index_dir: str) -> "FAISSRetriever":
        with open(os.path.join(index_dir, "config.json"), "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return FAISSRetriever(
            index_path=os.path.join(index_dir, "faiss.index"),
            meta_path=os.path.join(index_dir, "faiss_meta.pkl"),
            model_name=cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
        )

    def search(self, query: str, top_k: int = 100) -> List[Tuple[int, float]]:
        q = self.embedder.encode([query], normalize_embeddings=self.normalize)
        D, I = self.index.search(np.asarray(q, dtype=np.float32), top_k)
        return [(int(i), float(-d)) for i, d in zip(I[0], D[0]) if i != -1]


class BM25Retriever:
    def __init__(self, bm25_path: str, tokens_path: str, metas_path: str):
        with open(bm25_path, "rb") as f:
            self.bm25: BM25Okapi = pickle.load(f)
        with open(tokens_path, "rb") as f:
            self.corpus_tokens: List[List[str]] = pickle.load(f)
        with open(metas_path, "rb") as f:
            meta = pickle.load(f)
        self.chunk_texts: List[str] = meta["chunk_texts"]
        self.metas: List[ChunkMeta] = [ChunkMeta(**m) for m in meta["metas"]]

    @staticmethod
    def load(index_dir: str) -> "BM25Retriever":
        return BM25Retriever(
            bm25_path=os.path.join(index_dir, "bm25.pkl"),
            tokens_path=os.path.join(index_dir, "bm25_tokens.pkl"),
            metas_path=os.path.join(index_dir, "faiss_meta.pkl"),
        )

    def search(self, query_tokens: Iterable[str], top_k: int = 100) -> List[Tuple[int, float]]:
        scores = self.bm25.get_scores(list(query_tokens))
        idxs = np.argpartition(scores, -top_k)[-top_k:]
        idxs = idxs[np.argsort(scores[idxs])[::-1]]
        return [(int(i), float(scores[i])) for i in idxs]


def rrf_merge(
    dense: List[Tuple[int, float]],
    sparse: List[Tuple[int, float]],
    c: int = 60,
) -> Dict[int, float]:
    ranks: Dict[int, float] = {}
    for rank, (i, _) in enumerate(dense, start=1):
        ranks[i] = ranks.get(i, 0.0) + 1.0 / (c + rank)
    for rank, (i, _) in enumerate(sparse, start=1):
        ranks[i] = ranks.get(i, 0.0) + 1.0 / (c + rank)
    return ranks


def aggregate_by_web_id(
    indices_scores: Dict[int, float], metas: List[ChunkMeta], top_pages: int = 5
) -> List[Tuple[str, float]]:
    page_scores: Dict[str, float] = {}
    for idx, score in indices_scores.items():
        web = metas[idx].web_id
        if web not in page_scores or score > page_scores[web]:
            page_scores[web] = score
    ranked_pages = sorted(page_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_pages[:top_pages]


class HybridRetriever:
    def __init__(self, faiss_ret: FAISSRetriever, bm25_ret: BM25Retriever, tokenizer):
        self.faiss_ret = faiss_ret
        self.bm25_ret = bm25_ret
        self.tokenizer = tokenizer

    @staticmethod
    def load(index_dir: str, tokenizer) -> "HybridRetriever":
        return HybridRetriever(
            FAISSRetriever.load(index_dir),
            BM25Retriever.load(index_dir),
            tokenizer,
        )

    def search(self, query: str, top_k_dense: int = 200, top_k_bm25: int = 200, pages: int = 5) -> Dict[str, Any]:
        dense = self.faiss_ret.search(query, top_k_dense)
        sparse = self.bm25_ret.search(self.tokenizer(query), top_k_bm25)
        merged = rrf_merge(dense, sparse)
        top_pages = aggregate_by_web_id(merged, self.faiss_ret.metas, top_pages=pages)
        return {
            "pages": top_pages,
            "merged_scores": merged,
        }
