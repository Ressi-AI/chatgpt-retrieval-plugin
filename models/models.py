from pydantic import BaseModel
from typing import List, Optional, Dict, Union
from enum import Enum


class Source(str, Enum):
    email = "email"
    file = "file"
    chat = "chat"
    # ressi-updated
    file_header = "file_header"
    youtube = "youtube"
    article = "article"
    notes = "notes"


class DocumentMetadata(BaseModel):
    source: Optional[Source] = None
    source_id: Optional[str] = None
    url: Optional[str] = None
    created_at: Optional[str] = None
    author: Optional[str] = None
    # ressi-updated
    title: Optional[str] = None
    total_tokens: Optional[int] = None
    total_chunks: Optional[int] = None
    user_id: Optional[Union[str, int]] = None
    company_id: Optional[Union[str, int]] = None
    collection_id: Optional[Union[str, int]] = None
    public: Optional[bool] = None


class DocumentChunkMetadata(DocumentMetadata):
    document_id: Optional[str] = None
    # ressi-updated
    chunk_tokens: Optional[int] = None


class DocumentChunk(BaseModel):
    id: Optional[str] = None
    text: str
    metadata: DocumentChunkMetadata
    embedding: Optional[List[float]] = None


class DocumentChunkWithScore(DocumentChunk):
    score: float


class Document(BaseModel):
    id: Optional[str] = None
    text: str
    metadata: Optional[DocumentMetadata] = None


class DocumentWithChunks(Document):
    chunks: List[DocumentChunk]


class DocumentMetadataFilter(BaseModel):
    document_id: Optional[str] = None
    source: Optional[Source] = None
    source_id: Optional[str] = None
    author: Optional[str] = None
    start_date: Optional[str] = None  # any date string format
    end_date: Optional[str] = None  # any date string format
    # ressi-updated
    title: Optional[str] = None
    user_id: Optional[str] = None
    public: Optional[bool] = None
    extra_filters: Optional[Dict] = None  # other filters specific to the datastore (e.g. pinecone can receive $or, $and etc.) # noqa: E501


class Query(BaseModel):
    query: str
    filter: Optional[DocumentMetadataFilter] = None
    top_k: Optional[int] = 3


class QueryWithEmbedding(Query):
    embedding: List[float]


class QueryResult(BaseModel):
    query: str
    results: List[DocumentChunkWithScore]
