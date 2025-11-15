from typing import List, Dict, Any

class MinimalRAG:
    def __init__(self):
        self.docs = []

    def ingest(self, texts: List[str], metas: List[Dict[str, Any]]):
        for t, m in zip(texts, metas):
            self.docs.append({"text": t, "meta": m})

    def retrieve(self, query: str, k: int = 3):
        # Simple keyword overlap similarity
        q_words = set(query.lower().split())
        scored = []

        for d in self.docs:
            score = len(q_words.intersection(set(d["text"].lower().split())))
            scored.append((score, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:k]]
