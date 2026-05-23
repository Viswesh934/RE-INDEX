from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from db import init_db, get_db, User, Document, Embedding, SessionLocal
from schemas import UserCreate, UserResponse, DocumentCreate, DocumentResponse, QueryRequest, QueryResponse, QueryResult
from sqlalchemy.orm import Session
import uuid
import json
import numpy as np
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="RAG Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    init_db()
    print("✓ Database initialized")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/users", response_model=UserResponse, status_code=201)
async def create_user(user: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.username == user.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    new_user = User(
        id=str(uuid.uuid4()),
        username=user.username,
        email=user.email,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user


@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@app.post("/documents", response_model=DocumentResponse, status_code=201)
async def create_document(doc: DocumentCreate, owner_id: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == owner_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    new_doc = Document(
        id=str(uuid.uuid4()),
        owner_id=owner_id,
        title=doc.title,
        content=doc.content,
        is_public=int(doc.is_public),
    )
    db.add(new_doc)
    db.commit()
    db.refresh(new_doc)
    return new_doc


@app.get("/documents/{doc_id}", response_model=DocumentResponse)
async def get_document(doc_id: str, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@app.get("/users/{owner_id}/documents", response_model=list[DocumentResponse])
async def list_user_documents(owner_id: str, db: Session = Depends(get_db)):
    docs = db.query(Document).filter(Document.owner_id == owner_id).all()
    return docs


def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = end - overlap
    return [c for c in chunks if c.strip()]


def get_mock_embedding(text: str) -> list[float]:
    import hashlib
    hash_obj = hashlib.md5(text.encode())
    hash_int = int(hash_obj.hexdigest(), 16)
    np.random.seed(hash_int % (2**32))
    return np.random.randn(384).tolist()


@app.post("/documents/{doc_id}/embed", status_code=202)
async def embed_document(doc_id: str, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    db.query(Embedding).filter(Embedding.document_id == doc_id).delete()
    
    chunks = chunk_text(doc.content)
    for idx, chunk in enumerate(chunks):
        embedding_vector = get_mock_embedding(chunk)
        embedding = Embedding(
            id=str(uuid.uuid4()),
            document_id=doc_id,
            chunk_text=chunk,
            chunk_index=idx,
            embedding=json.dumps(embedding_vector),
        )
        db.add(embedding)
    
    db.commit()
    return {"status": "embedding started", "chunks": len(chunks)}


@app.post("/query", response_model=QueryResponse)
async def query_documents(query_req: QueryRequest, owner_id: str, db: Session = Depends(get_db)):
    query_embedding = get_mock_embedding(query_req.query)
    
    docs = db.query(Document).filter(
        (Document.owner_id == owner_id) | (Document.is_public == 1)
    ).all()
    
    if not docs:
        return QueryResponse(query=query_req.query, results=[])
    
    doc_ids = [d.id for d in docs]
    embeddings = db.query(Embedding).filter(Embedding.document_id.in_(doc_ids)).all()
    
    results = []
    for emb in embeddings:
        emb_vector = json.loads(emb.embedding)
        score = cosine_similarity(query_embedding, emb_vector)
        results.append({"embedding": emb, "score": score})
    
    results.sort(key=lambda x: x["score"], reverse=True)
    top_results = results[:query_req.top_k]
    
    query_results = [
        QueryResult(
            document_id=r["embedding"].document_id,
            content=r["embedding"].chunk_text,
            score=r["score"]
        )
        for r in top_results
    ]
    
    return QueryResponse(query=query_req.query, results=query_results)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
