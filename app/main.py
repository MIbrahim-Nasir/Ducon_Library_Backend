from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from . import db
from app.ml import EmbeddingModel



@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.collection = db.get_db_collection()
    app.state.model = EmbeddingModel()
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root(request: Request):
    col = request.app.state.collection

    return {"item_count": request.app.state.collection.count() }

@app.get("/searchtext")
def search_text(request: Request, query: str):
    embedding = app.state.model.get_embedding(query)
    results = db.retrieve(app.state.collection, embedding)

    return {"results": results}


@app.post("/model")
def test_model(request: str):
    embedding = app.state.model.get_embedding(request)

    return {"embedding": embedding}

def main():
    print("Hello from ducon-library-backend!")


if __name__ == "__main__":
    main()
