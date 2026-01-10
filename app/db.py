import chromadb

def get_db_collection():
    chroma_client = chromadb.PersistentClient(path="chroma_db")

    collection = chroma_client.get_or_create_collection("ducon_library")

    return collection
