from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List


class UserBase(BaseModel):
    username: str
    email: str


class UserCreate(UserBase):
    pass


class UserResponse(UserBase):
    id: str
    created_at: datetime
    
    class Config:
        from_attributes = True


class DocumentBase(BaseModel):
    title: str
    content: str
    is_public: bool = False


class DocumentCreate(DocumentBase):
    pass


class DocumentResponse(DocumentBase):
    id: str
    owner_id: str
    created_at: datetime
    
    class Config:
        from_attributes = True


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class QueryResult(BaseModel):
    document_id: str
    content: str
    score: float


class QueryResponse(BaseModel):
    query: str
    results: List[QueryResult]
