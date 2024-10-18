import uuid
import tiktoken
import pytz
from datetime import datetime
from typing import Optional, List, Tuple, Dict
from models.models import Document, DocumentChunk, DocumentChunkMetadata
from services.openai_calls import get_embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

tokenizer = tiktoken.get_encoding(
    "cl100k_base"
)  # The encoding scheme to use for tokenization

# Constants
CHUNK_SIZE = 2000  # The target size of each text chunk in tokens
CHUNK_OVERLAP = 100
EMBEDDINGS_BATCH_SIZE = 128  # The number of embeddings to request at a time


def get_text_chunks(text: str, chunk_token_size: Optional[int] = None) -> Tuple[List[str], int]:
    # Return an empty list if the text is empty or whitespace
    if not text or text.isspace():
        return [], 0

    chunk_size = chunk_token_size or CHUNK_SIZE
    text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=chunk_size,
      chunk_overlap=CHUNK_OVERLAP,
      separators=[
        "\nARTÍCULO", "\n ARTÍCULO", "\nArtículo", "\n Artículo",
        "\nARTICULO", "\n ARTICULO", "\nArticulo", "\n Articulo",
        "\n\n", "\n", " ", ""
      ]
    )

    chunks = text_splitter.split_text(text)
    n_tokens = len(tokenizer.encode(text, disallowed_special=()))
    return chunks, n_tokens


def create_document_chunks(
    doc: Document,
    chunk_token_size: Optional[int],
) -> Tuple[List[DocumentChunk], str]:
  """
  Create a list of document chunks from a document object and return the document id.

  Args:
      doc: The document object to create chunks from. It should have a text attribute and optionally an id and a metadata attribute.
      chunk_token_size: The target size of each chunk in tokens, or None to use the default CHUNK_SIZE.
  Returns:
      A tuple of (doc_chunks, doc_id), where doc_chunks is a list of document chunks, each of which is a DocumentChunk object with an id, a document_id, a text, and a metadata attribute,
      and doc_id is the id of the document object, generated if not provided. The id of each chunk is generated from the document id and a sequential number, and the metadata is copied from the document object.
  """
  # Check if the document text is empty or whitespace
  if not doc.text or doc.text.isspace():
    return [], doc.id or str(uuid.uuid4())

  # Generate a document id if not provided
  doc_id = doc.id or str(uuid.uuid4())

  # Split the document text into chunks
  text_chunks, total_tokens = get_text_chunks(doc.text, chunk_token_size=chunk_token_size)

  # Initialize an empty list of chunks for this document
  doc_chunks = []

  created_at = datetime.now(tz=pytz.timezone("UTC")).strftime("%Y-%m-%d %H:%M:%S")

  # Assign each chunk a sequential number and create a DocumentChunk object
  for i, text_chunk in enumerate(text_chunks):
    chunk_id = f"{doc_id}_{i}"

    chunk_tokens = len(tokenizer.encode(text_chunk, disallowed_special=()))

    dct_metadata = {
      'document_id': doc_id,
      'chunk_tokens': chunk_tokens, 'total_tokens': total_tokens, 'total_chunks': len(text_chunks),
      'created_at': created_at,
    }
    if doc.metadata is not None:
      for _k, _v in doc.metadata.__dict__.items():
        if _v is not None and _k not in dct_metadata:
          dct_metadata[_k] = _v

    doc_chunk = DocumentChunk(
      id=chunk_id,
      text=text_chunk,
      metadata=DocumentChunkMetadata(**dct_metadata),
    )
    # Append the chunk object to the list of chunks for this document
    doc_chunks.append(doc_chunk)

  # Return the list of chunks and the document id
  return doc_chunks, doc_id


def get_document_chunks(
    documents: List[Document],
    chunk_token_size: Optional[int],
) -> Dict[str, List[DocumentChunk]]:
  """
  Convert a list of documents into a dictionary from document id to list of document chunks.

  Args:
      documents: The list of documents to convert.
      chunk_token_size: The target size of each chunk in tokens, or None to use the default CHUNK_SIZE.
  Returns:
      A dictionary mapping each document id to a list of document chunks, each of which is a DocumentChunk object
      with text, metadata, and embedding attributes.
  """
  # Initialize an empty dictionary of lists of chunks
  chunks: Dict[str, List[DocumentChunk]] = {}

  # Initialize an empty list of all chunks
  all_chunks: List[DocumentChunk] = []

  # Loop over each document and create chunks
  for doc in documents:
    doc_chunks, doc_id = create_document_chunks(doc, chunk_token_size)

    # Append the chunks for this document to the list of all chunks
    all_chunks.extend(doc_chunks)

    # Add the list of chunks for this document to the dictionary with the document id as the key
    chunks[doc_id] = doc_chunks

  # Check if there are no chunks
  if not all_chunks:
    return {}

  # Get all the embeddings for the document chunks in batches, using get_embeddings
  embeddings: List[List[float]] = []
  for i in range(0, len(all_chunks), EMBEDDINGS_BATCH_SIZE):
    # Get the text of the chunks in the current batch
    batch_texts = [
      chunk.text for chunk in all_chunks[i: i + EMBEDDINGS_BATCH_SIZE]
    ]

    # Get the embeddings for the batch texts
    batch_embeddings = get_embeddings(batch_texts)

    # Append the batch embeddings to the embeddings list
    embeddings.extend(batch_embeddings)

  # Update the document chunk objects with the embeddings
  for i, chunk in enumerate(all_chunks):
    # Assign the embedding from the embeddings list to the chunk object
    chunk.embedding = embeddings[i]

  return chunks

