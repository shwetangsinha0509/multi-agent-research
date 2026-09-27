import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv

# CLOUD STUB: heavy RAG dependencies disabled for free-tier cloud deployment (~1.5GB RAM required)
# To enable locally: pip install langchain-huggingface langchain-chroma chromadb sentence-transformers
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

load_dotenv()

CHROMA_DIR = "chroma_store"

_embeddings = None


def get_embeddings():
    # CLOUD STUB — returns None when RAG dependencies are not installed
    # Original: initializes HuggingFaceEmbeddings with "all-MiniLM-L6-v2"
    if not RAG_AVAILABLE:
        return None
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embeddings


def get_vectorstore(collection_name: str):
    """Load an existing ChromaDB collection as a LangChain vectorstore."""
    # CLOUD STUB — returns None when RAG dependencies are not installed
    # Original: Chroma(collection_name=collection_name, embedding_function=get_embeddings(), persist_directory=CHROMA_DIR)
    if not RAG_AVAILABLE:
        return None
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR
    )


def ingest_pdf(file_path: str, collection_name: str) -> int:
    """Load PDF → split → embed → store in ChromaDB. Returns chunk count."""
    # CLOUD STUB — returns 0 when RAG dependencies are not installed
    # Original: loads PDF, splits into chunks, embeds and stores in ChromaDB
    if not RAG_AVAILABLE:
        return 0
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=collection_name,
        persist_directory=CHROMA_DIR
    )

    return len(chunks)