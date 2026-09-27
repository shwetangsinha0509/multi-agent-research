import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

# NOTE: RAG functionality (sentence-transformers, chromadb) is disabled in cloud deployment
# due to free-tier memory constraints (512MB limit; torch+sentence-transformers requires ~1.5GB).
# To enable full RAG, run locally or on an instance with at least 1GB RAM and uncomment below:
#
# from rag.ingestor import ingest_pdf, get_vectorstore
# from rag.retriever import retrieve_chunks

load_dotenv()

mcp = FastMCP("RAG Document Server", host="0.0.0.0", port=1000)


@mcp.tool()
def search_document(query: str, collection_name: str) -> str:
    """
    Search the uploaded PDF document for information relevant to the query.
    Use this when the user has uploaded a document and asks questions about it.
    Returns the most relevant chunks with page numbers.

    Args:
        query: The search query or question to find relevant content for
        collection_name: The ChromaDB collection name for the uploaded document
    """
    # CLOUD STUB: document search disabled in cloud deployment (memory constraints)
    # Uncomment below for local/self-hosted deployment with sufficient RAM:
    #
    # if not collection_name:
    #     return "No document collection specified."
    # try:
    #     return retrieve_chunks(query, collection_name)
    # except Exception as e:
    #     return f"Document search failed: {str(e)}"

    return "No document uploaded."


@mcp.tool()
def ingest_document(file_path: str, collection_name: str) -> str:
    """
    Ingest a PDF document into the vector store for later retrieval.
    Call this before search_document to make a PDF searchable.

    Args:
        file_path: Absolute path to the PDF file on disk
        collection_name: Name to give this document's ChromaDB collection
    """
    # CLOUD STUB: document ingestion disabled in cloud deployment (memory constraints)
    # Uncomment below for local/self-hosted deployment with sufficient RAM:
    #
    # if not os.path.exists(file_path):
    #     return f"File not found: {file_path}"
    # try:
    #     chunk_count = ingest_pdf(file_path, collection_name)
    #     return f"Successfully ingested {chunk_count} chunks from {file_path}"
    # except Exception as e:
    #     return f"Ingestion failed: {str(e)}"

    return "Document ingestion is available in self-hosted deployment."


@mcp.tool()
def list_collections() -> str:
    """
    List all available document collections in the vector store.
    Use this to check which documents have been ingested.
    """
    # CLOUD STUB: ChromaDB disabled in cloud deployment (memory constraints)
    # Uncomment below for local/self-hosted deployment with sufficient RAM:
    #
    # try:
    #     import chromadb
    #     client = chromadb.PersistentClient(path="chroma_store")
    #     collections = client.list_collections()
    #     if not collections:
    #         return "No documents have been ingested yet."
    #     names = [c.name for c in collections]
    #     return f"Available collections: {', '.join(names)}"
    # except Exception as e:
    #     return f"Failed to list collections: {str(e)}"

    return "No documents have been ingested yet."


if __name__ == "__main__":
    print("Starting RAG MCP Server on port 8004...")
    mcp.run(transport="streamable-http")