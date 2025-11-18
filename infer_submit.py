import argparse
import os
import pandas as pd
from typing import List, Tuple

from sentence_transformers import CrossEncoder
from tqdm import tqdm

from retrievers import HybridRetriever
from utils_text import simple_tokenize


def rerank_and_select(
    retriever: HybridRetriever,
    query: str,
    top_k_dense: int = 200,
    top_k_bm25: int = 200,
    top_k_chunks_for_rerank: int = 200,
    pages_out: int = 5,
    cross_encoder: CrossEncoder | None = None,
) -> List[str]:
    result = retriever.search(query, top_k_dense=top_k_dense, top_k_bm25=top_k_bm25, pages=pages_out)
    merged = result["merged_scores"]
    top_chunks = sorted(merged.items(), key=lambda x: x[1], reverse=True)[:top_k_chunks_for_rerank]
    if not top_chunks:
        return []
    if cross_encoder is None:
        cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs: List[Tuple[str, str]] = []
    chunk_indices: List[int] = []
    for idx, _ in top_chunks:
        pairs.append((query, retriever.faiss_ret.chunk_texts[idx]))
        chunk_indices.append(idx)
    scores = cross_encoder.predict(pairs, batch_size=64, show_progress_bar=False)
    web_best: dict = {}
    for idx, s in zip(chunk_indices, scores):
        meta = retriever.faiss_ret.metas[idx]
        title = (meta.title or "").lower()
        url = (meta.url or "").lower()
        kind = (meta.kind or "").lower()

        if (".pdf" in title or ".pdf" in url
            or "dogovor" in title or "договор" in title
            or "политик" in title
            or kind in {"pdf", "doc", "docx"}):
            continue

        web_id = int(meta.web_id)
        if web_id not in web_best or s > web_best[web_id]:
            web_best[web_id] = float(s)
    top_pages = sorted(web_best.items(), key=lambda x: x[1], reverse=True)[:pages_out]
    return [str(int(w)) for w, _ in top_pages]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions_csv", required=True)
    parser.add_argument("--index_dir", required=True)
    parser.add_argument("--out_csv", default="submit.csv")
    parser.add_argument("--top_k_dense", type=int, default=200)
    parser.add_argument("--top_k_bm25", type=int, default=200)
    parser.add_argument("--top_k_chunks_for_rerank", type=int, default=200)
    parser.add_argument("--pages_out", type=int, default=5)
    parser.add_argument("--cross_encoder_model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--format", choices=["wide", "list"], default="wide", help="wide: q_id,web_id_1..5; list: q_id,web_ids")
    args = parser.parse_args()

    print("[INFO] Loading hybrid retriever...")
    retriever = HybridRetriever.load(args.index_dir, tokenizer=simple_tokenize)
    print("[INFO] Retriever loaded.")

    print(f"[INFO] Loading questions from {args.questions_csv}...")
    qdf = pd.read_csv(args.questions_csv)
    required_cols = {"q_id", "query"}
    missing = required_cols - set(qdf.columns)
    if missing:
        raise ValueError(f"Missing columns in Questions.csv: {missing}")

    print("[INFO] Loading cross-encoder model (this may take some time on first run)...")
    cross_encoder = CrossEncoder(args.cross_encoder_model)
    print("[INFO] Cross-encoder loaded. Starting inference over questions...")

    rows = []
    for _, row in tqdm(qdf.iterrows(), total=len(qdf), desc="Questions"):
        q_id = row["q_id"]
        query = str(row["query"]) if pd.notna(row["query"]) else ""
        web_ids = rerank_and_select(
            retriever,
            query=query,
            top_k_dense=args.top_k_dense,
            top_k_bm25=args.top_k_bm25,
            top_k_chunks_for_rerank=args.top_k_chunks_for_rerank,
            pages_out=args.pages_out,
            cross_encoder=cross_encoder,
        )
        if args.format == "wide":
            row_out = {"q_id": q_id}
            for i in range(args.pages_out):
                web_id = str(int(web_ids[i])) if i < len(web_ids) and web_ids[i] else ""
                row_out[f"web_id_{i+1}"] = web_id
            rows.append(row_out)
        else:
            web_ids_clean = [str(int(w)) for w in web_ids if w]
            rows.append({"q_id": q_id, "web_ids": " ".join(web_ids_clean)})

    odf = pd.DataFrame(rows)
    odf.to_csv(args.out_csv, index=False)
    print(f"Saved {len(odf)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
