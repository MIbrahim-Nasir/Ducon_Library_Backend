import chromadb

CLIENT = chromadb.PersistentClient(path="./chroma_db")

COLLECTION_NAME = "fused_store_legacy"


def get_db_collection():
    """Returns the single shared Ducon vector store collection."""
    return CLIENT.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def retrieve(collection: chromadb.Collection, query_embedding: list[float], n_results: int = 5):
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )
    return results
