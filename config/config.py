import os
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "your_api_key")
OPENAI_MODEL = "gpt-4o-mini"

SERP_API_KEY = os.environ.get("SERP_API_KEY", "your_api_key")

OPENAI_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_PERSIST_DIR = "./chroma_db"
DATA_DIR = "./data"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K_RESULTS = 4