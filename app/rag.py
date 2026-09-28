import glob
import chromadb
from chromadb import EmbeddingFunction
from sklearn.feature_extraction.text import HashingVectorizer

class HashEmbed(EmbeddingFunction):
    """Offline, deterministic embeddings so no model download is needed."""
    def __init__(self):
        self.v = HashingVectorizer(n_features=512, alternate_sign=False, norm="l2", stop_words="english")
    @staticmethod
    def name():
        return "hash-embed"
    def __call__(self, input):
        return self.v.transform(input).toarray().tolist()

_col = None
def _collection():
    global _col
    if _col is None:
        _col = chromadb.Client().get_or_create_collection("knowledge_base", embedding_function=HashEmbed())
        docs = [p.strip() for f in sorted(glob.glob("data/kb/*.md")) for p in open(f).read().split("\n\n") if p.strip()]
        _col.add(documents=docs, ids=[f"d{i}" for i in range(len(docs))])
    return _col

def search(query, k=2):
    return _collection().query(query_texts=[query], n_results=k)["documents"][0]
