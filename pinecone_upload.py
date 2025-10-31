import os
import json
import time
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

from config import (
    PINECONE_API_KEY,
    DATA_FILE,
    BATCH_SIZE,
    PINECONE_INDEX_NAME,    
    HF_MODEL_NAME,
    PINECONE_VECTOR_DIM
)

# ======================
# Initialize Pinecone client
# ======================
print("Initializing Pinecone client...")
pc = Pinecone(api_key=PINECONE_API_KEY)

# ======================
# Create or connect to index
# ======================
existing_indexes = pc.list_indexes().names()
if PINECONE_INDEX_NAME not in existing_indexes:
    print(f"🆕 Creating Pinecone index: {PINECONE_INDEX_NAME}")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=PINECONE_VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
else:
    print(f"✅ Index {PINECONE_INDEX_NAME} already exists.")

index = pc.Index(PINECONE_INDEX_NAME)

# ======================
# Load embedding model
# ======================
print(f"📥 Loading Hugging Face model: {HF_MODEL_NAME}")
model = SentenceTransformer(HF_MODEL_NAME)

# ======================
# Helper functions
# ======================
def get_embeddings(texts):
    """Generate embeddings for one or multiple texts."""
    if isinstance(texts, str):
        texts = [texts]
    return model.encode(texts, show_progress_bar=False).tolist()

def chunked(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]

# ======================
# Main function
# ======================
def main():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"❌ Dataset file not found: {DATA_FILE}")

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    items = []
    for node in nodes:
        semantic_text = node.get("semantic_text") or (node.get("description") or "")[:1000]
        if not semantic_text.strip():
            continue
        meta = {
            "id": node.get("id"),
            "type": node.get("type"),
            "name": node.get("name"),
            "city": node.get("city", node.get("region", "")),
            "tags": node.get("tags", []),
            "description": node.get("description", "")[:500]
        }
        items.append((node["id"], semantic_text, meta))

    print(f"Preparing to upsert {len(items)} items to Pinecone...")

    for batch in tqdm(list(chunked(items, BATCH_SIZE)), desc="Uploading batches"):
        ids = [item[0] for item in batch]
        texts = [item[1] for item in batch]
        metas = [item[2] for item in batch]

        embeddings = get_embeddings(texts)
        vectors = [
            {"id": _id, "values": emb, "metadata": meta}
            for _id, emb, meta in zip(ids, embeddings, metas)
        ]

        index.upsert(vectors)
        time.sleep(0.2)

    print("✅ All items uploaded successfully to Pinecone.")

# ======================
# Run script
# ======================
if __name__ == "__main__":
    main()

