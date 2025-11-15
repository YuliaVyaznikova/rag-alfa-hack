import argparse
import json
import os
import pickle
from typing import Dict, List

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from utils_text import clean_text, chunk_text, simple_tokenize


def build_indices(
    websites_csv: str,
    out_dir: str,
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    chunk_size: int = 1200,
    overlap: int = 300,
    min_chunk_len: int = 100,
    hnsw_m: int = 32,
    ef_search: int = 64,
):
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(websites_csv)
    required_cols = {"web_id", "url", "kind", "title", "text"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in Websites.csv: {missing}")

    embedder = SentenceTransformer(embedding_model)

    chunk_texts: List[str] = []
    metas: List[Dict] = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        web_id = str(row["web_id"]) if not pd.isna(row["web_id"]) else ""
        url = str(row["url"]) if not pd.isna(row["url"]) else ""
        kind = str(row["kind"]) if not pd.isna(row["kind"]) else ""
        title = clean_text(str(row["title"]) if not pd.isna(row["title"]) else "")
        text = clean_text(str(row["text"]) if not pd.isna(row["text"]) else "")
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for j, (s, e, ch) in enumerate(chunks):
            if len(ch) < min_chunk_len:
                continue
            cid = f"{web_id}:{j}"
            chunk_texts.append(ch)
            metas.append(
                {
                    "chunk_id": cid,
                    "web_id": web_id,
                    "url": url,
                    "kind": kind,
                    "title": title,
                    "offset_start": int(s),
                    "offset_end": int(e),
                }
            )

    emb = embedder.encode(chunk_texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    emb = np.asarray(emb, dtype=np.float32)

    dim = emb.shape[1]
    index = faiss.IndexHNSWFlat(dim, hnsw_m)
    index.hnsw.efSearch = ef_search
    index.add(emb)

    faiss.write_index(index, os.path.join(out_dir, "faiss.index"))

    with open(os.path.join(out_dir, "faiss_meta.pkl"), "wb") as f:
        pickle.dump({"chunk_texts": chunk_texts, "metas": metas}, f)

    tokenized_corpus = [simple_tokenize(t) for t in chunk_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(os.path.join(out_dir, "bm25.pkl"), "wb") as f:
        pickle.dump(bm25, f)
    with open(os.path.join(out_dir, "bm25_tokens.pkl"), "wb") as f:
        pickle.dump(tokenized_corpus, f)

    cfg = {
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "min_chunk_len": min_chunk_len,
        "hnsw_m": hnsw_m,
        "ef_search": ef_search,
        "counts": {"chunks": len(chunk_texts), "docs": int(len(df))},
    }
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print(f"Saved FAISS, BM25 and metadata to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--websites_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--embedding_model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--chunk_size", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=300)
    parser.add_argument("--min_chunk_len", type=int, default=100)
    parser.add_argument("--hnsw_m", type=int, default=32)
    parser.add_argument("--ef_search", type=int, default=64)
    args = parser.parse_args()

    build_indices(
        websites_csv=args.websites_csv,
        out_dir=args.out_dir,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        min_chunk_len=args.min_chunk_len,
        hnsw_m=args.hnsw_m,
        ef_search=args.ef_search,
    )
