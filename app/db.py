import chromadb

def get_db_collection():
    chroma_client = chromadb.PersistentClient(path="chroma_db")

    collection = chroma_client.get_or_create_collection("ducon_library")

    return collection

def retrieve(collection: chromadb.Collection, query_embedding: list[float], n_results: int = 5):
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    return results
