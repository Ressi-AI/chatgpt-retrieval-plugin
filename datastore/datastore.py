import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import asyncio
from loguru import logger
from time import sleep

from models.models import (
    Document,
    DocumentChunk,
    DocumentMetadataFilter,
    Query,
    QueryResult,
    QueryWithEmbedding,
)
if True:
    from services.langchain_chunks import get_document_chunks
else:
    from services.chunks import get_document_chunks
from services.openai_calls import get_embeddings

EMBEDDING_DIMENSION = int(os.environ.get("EMBEDDING_DIMENSION", 256))


class DataStore(ABC):
    async def upsert(
        self, documents: List[Document], chunk_token_size: Optional[int] = None
    ) -> List[str]:
        """
        Takes in a list of documents and inserts them into the database.
        First deletes all the existing vectors with the document id (if necessary, depends on the vector db), then inserts the new ones.
        Return a list of document ids.
        """
        # Delete any existing vectors for documents with the input document ids
        await asyncio.gather(
            *[
                self.delete(
                    filter=DocumentMetadataFilter(
                        document_id=document.id,
                    ),
                    delete_all=False,
                )
                for document in documents
                if document.id
            ]
        )

        chunks = get_document_chunks(documents, chunk_token_size)

        return await self._upsert(chunks)

    @abstractmethod
    async def _upsert(self, chunks: Dict[str, List[DocumentChunk]]) -> List[str]:
        """
        Takes in a list of document chunks and inserts them into the database.
        Return a list of document ids.
        """

        raise NotImplementedError

    async def query(self, queries: List[Query]) -> List[QueryResult]:
        """
        Takes in a list of queries and filters and returns a list of query results with matching document chunks and scores.
        """
        # get a list of just the queries from the Query list
        query_texts = [query.query for query in queries]
        query_embeddings = get_embeddings(query_texts)
        # hydrate the queries with embeddings
        queries_with_embeddings = [
            QueryWithEmbedding(**query.dict(), embedding=embedding)
            for query, embedding in zip(queries, query_embeddings)
        ]
        return await self._query(queries_with_embeddings)

    @abstractmethod
    async def _query(self, queries: List[QueryWithEmbedding]) -> List[QueryResult]:
        """
        Takes in a list of queries with embeddings and filters and returns a list of query results with matching document chunks and scores.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[DocumentMetadataFilter] = None,
        delete_all: Optional[bool] = None,
    ) -> bool:
        """
        Removes vectors by ids, filter, or everything in the datastore.
        Multiple parameters can be used at once.
        Returns whether the operation was successful.
        """
        raise NotImplementedError

    async def _batch_update_metadata(self, documents: List[Document], batch_size=100):
        nr_documents = len(documents)
        if nr_documents % batch_size == 0:
            nr_batches = nr_documents // batch_size
        else:
            nr_batches = 1 + nr_documents // batch_size

        logger.info(f"Updating metadata for {nr_batches} batches (batch_size={batch_size}) ...")
        for batch_idx in range(nr_batches):
            batch_start = batch_idx * batch_size
            batch_end = (batch_idx + 1) * batch_size
            batch_documents = documents[batch_start:batch_end]
            sleep(1.0)
            await asyncio.gather(
                *[
                    self._update_metadata(document=document)
                    for document in batch_documents
                    if document.id and document.metadata
                ]
            )
            logger.info(f"Batch {batch_idx+1}/{nr_batches} completed.")

        return

    async def update_metadata(self, documents: List[Document], direct=True, update_batch_size=100):
        if direct:
            await self._batch_update_metadata(documents=documents, batch_size=update_batch_size)
            return

        queries_with_embeddings = [
            QueryWithEmbedding(
                query='', filter=DocumentMetadataFilter(document_id=d.id),
                top_k=1, embedding=[0.0 for _ in range(EMBEDDING_DIMENSION)]
            )
            for d in documents
        ]
        queries_results = await self._query(queries_with_embeddings)
        chunk_documents = []
        for i, r in enumerate(queries_results):
            if len(r.results) == 0:
                logger.error(f"No query result for document {documents[i].id}")
                continue

            top_result = r.results[0]
            if top_result.metadata.total_chunks is None:
                logger.error(f"No total chunks for document {documents[i].id}")
                continue

            total_chunks = int(top_result.metadata.total_chunks)
            for j in range(total_chunks):
                chunk_documents.append(Document(
                    id=f"{documents[i].id}_{j}",
                    text=documents[i].text,
                    metadata=documents[i].metadata,
                ))

        await self._batch_update_metadata(documents=chunk_documents, batch_size=update_batch_size)
        return

    @abstractmethod
    async def _update_metadata(self, document: Document):
        raise NotImplementedError
