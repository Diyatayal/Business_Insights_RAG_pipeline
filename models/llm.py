from langchain_groq import ChatGroq
from config.config import GROQ_API_KEY, GROQ_MODEL

def get_llm(temperature: float = 0.3):
    """Return a ChatGroq LLM instance."""
    try:
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model_name=GROQ_MODEL,
            temperature=temperature
        )
        return llm
    except Exception as e:
        raise RuntimeError(f"Failed to initialize LLM: {e}")