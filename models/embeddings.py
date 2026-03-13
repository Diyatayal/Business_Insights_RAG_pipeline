from langchain_community.embeddings import HuggingFaceEmbeddings
from config.config import EMBEDDING_MODEL

def get_embedding_model():
    """Load HuggingFace embedding model (cached after first load)."""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        return embeddings
    except Exception as e:
        raise RuntimeError(f"Failed to load embedding model: {e}")