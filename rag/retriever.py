import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from rag.ingestor import get_vectorstore


def retrieve_chunks(question: str, collection_name: str, top_k: int = 5) -> str:
    """
    Fetch top_k relevant chunks from ChromaDB for a given question.
    Returns a single formatted string — the agent reads this as tool output.
    """
    vectorstore = get_vectorstore(collection_name)

    results = vectorstore.similarity_search(question, k=top_k)

    if not results:
        return "No relevant information found in the uploaded document."

    chunks_text = "\n\n---\n\n".join([
        f"[Page {doc.metadata.get('page', '?')}]: {doc.page_content}"
        for doc in results
    ])

    return chunks_text