import os

# -----------------------------
# Environment Variables
# -----------------------------

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "ENTER")
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY", "ENTER")
os.environ["NEO4J_URI"] = os.getenv("NEO4J_URI", "ENTER")
os.environ["NEO4J_USER"] = os.getenv("NEO4J_USER", "neo4j")
os.environ["NEO4J_PASSWORD"] = os.getenv("NEO4J_PASSWORD", "ENTER")
os.environ["PINECONE_INDEX_NAME"] = os.getenv("PINECONE_INDEX_NAME", "vietnam-travel-index")

# -----------------------------
# Configuration Constants
# -----------------------------
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]
PINECONE_VECTOR_DIM = 384  # all-MiniLM-L6-v2 has 384 dimensions

NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USER = os.environ["NEO4J_USER"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

GEMINI_API_KEY = os.environ["GOOGLE_API_KEY"]
GEMINI_MODEL = "models/gemini-2.0-flash"

# Retrieval & cache configuration
TOP_K = 5
EMBED_CACHE_PATH = "emb_cache.json"

# Regions and providers
PINECONE_REGION = "us-east1"
PINECONE_CLOUD = "aws"

DATA_FILE = "vietnam_travel_dataset.json"
HF_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32