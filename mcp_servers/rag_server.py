import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from mcp.server.fastmcp import FastMCP
from rag.ingestor import ingest_pdf, get_vectorstore
from rag.retriever import retrieve_chunks
from dotenv import load_dotenv

load_dotenv()

# Create the MCP server
# This is YOUR custom MCP server — it exposes your RAG pipeline
# as a standard MCP interface that any MCP client can connect to:
# your Document Agent, Claude Desktop, Cursor, ChatGPT, etc.
mcp = FastMCP("RAG Document Server", host="0.0.0.0", port=8004)


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
    if not collection_name:
        return "No document collection specified."

    try:
        return retrieve_chunks(query, collection_name)
    except Exception as e:
        return f"Document search failed: {str(e)}"


@mcp.tool()
def ingest_document(file_path: str, collection_name: str) -> str:
    """
    Ingest a PDF document into the vector store for later retrieval.
    Call this before search_document to make a PDF searchable.

    Args:
        file_path: Absolute path to the PDF file on disk
        collection_name: Name to give this document's ChromaDB collection
    """
    if not os.path.exists(file_path):
        return f"File not found: {file_path}"

    try:
        chunk_count = ingest_pdf(file_path, collection_name)
        return f"Successfully ingested {chunk_count} chunks from {file_path}"
    except Exception as e:
        return f"Ingestion failed: {str(e)}"


@mcp.tool()
def list_collections() -> str:
    """
    List all available document collections in the vector store.
    Use this to check which documents have been ingested.
    """
    try:
        import chromadb
        client = chromadb.PersistentClient(path="chroma_store")
        collections = client.list_collections()
        if not collections:
            return "No documents have been ingested yet."
        names = [c.name for c in collections]
        return f"Available collections: {', '.join(names)}"
    except Exception as e:
        return f"Failed to list collections: {str(e)}"


if __name__ == "__main__":
    # Run as a networked HTTP server on port 8004
    # Transport is "http" (Streamable HTTP) — NOT "sse" which is deprecated
    # The server is reachable at http://localhost:8004/mcp
    print("Starting RAG MCP Server on port 8004...")
    mcp.run(transport="streamable-http")