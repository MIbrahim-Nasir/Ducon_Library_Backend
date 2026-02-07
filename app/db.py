import chromadb

CLIENT = chromadb.PersistentClient(path="./chroma_db")

def get_db_collection(name: str):
    """
    Gets a specific collection by name.
    """
    # chroma_client = chromadb.PersistentClient(path="chroma_db")

    collection = CLIENT.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}
    )

    return collection

def retrieve(collection: chromadb.Collection,   query_embedding: list[float], n_results: int = 5):
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    return results
