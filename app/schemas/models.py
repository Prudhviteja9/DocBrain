# Data models for DocBrain API

from pydantic import BaseModel
from typing import Optional


class QuestionRequest(BaseModel):
    question: str
    collection_name: str = "documents"
    n_chunks: int = 5


class AnswerResponse(BaseModel):
    answer: str
    sources: list[dict]
    chunks_used: int


class DocumentInfo(BaseModel):
    filename: str
    pages: int
    chunks: int
    collection: str
