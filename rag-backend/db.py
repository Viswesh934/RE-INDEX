from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/rag_db")

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    
    id = Column(String(36), primary_key=True)
    username = Column(String(255), unique=True)
    email = Column(String(255), unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Document(Base):
    __tablename__ = "documents"
    
    id = Column(String(36), primary_key=True)
    owner_id = Column(String(36), ForeignKey("users.id"))
    title = Column(String(255))
    content = Column(Text)
    is_public = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


class Embedding(Base):
    __tablename__ = "embeddings"
    
    id = Column(String(36), primary_key=True)
    document_id = Column(String(36), ForeignKey("documents.id"))
    chunk_text = Column(Text)
    chunk_index = Column(Integer)
    embedding = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
