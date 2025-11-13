from sentence_transformers import SentenceTransformer
# from langchain. import RecursiveCharacterTextSplitter
from transformers import pipeline
import faiss
import numpy as np
import ollama


class Rag():
    def __init__(self, texts, chunk_size=200):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.chunks = []
        for text in texts:
            self.chunks += self.split_text_to_chunks(text, chunk_size=chunk_size)
        
        embeddings = self.embedder.encode(self.chunks, normalize_embeddings=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexHNSWFlat(dim, 32)
        self.index.hnsw.efSearch = 64
        self.index.add(embeddings)

    def split_text_to_chunks(self, text, chunk_size=200, overlap=50):
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += chunk_size - overlap
        return chunks
    
    def retrieve(self, query, top_k=10):
        query_emb = self.embedder.encode([query])
        D, I = self.index.search(np.array(query_emb, dtype=np.float32), top_k)
        print("retrieve ", D)
        print()
        return [self.chunks[i] for i in I[0]]



