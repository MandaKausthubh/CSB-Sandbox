from langGraph.Agent import LegalRAGEngine
from langchain_text_splitter import RecursiveCharacterTextSplitter
import glob

def load_legal_documents(path="./legal_docs/*.txt"):
    files = glob.glob(path)
    docs = []
    metadata = []

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    for f in files:
        with open(f, "r") as fp:
            raw = fp.read()
        chunks = splitter.split_text(raw)
        for c in chunks:
            docs.append(c)
            metadata.append({
                "source": f,
            })

    return docs, metadata

def build_rag():
    engine = LegalRAGEngine()
    docs, meta = load_legal_documents()
    engine.ingest(docs, meta)
    return engine

if __name__ == "__main__":
    rag = build_rag()
    print("Legal RAG built with documents:", len(rag.documents))
