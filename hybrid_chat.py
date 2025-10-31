import os
import json
from hashlib import sha256
from typing import List
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from neo4j import GraphDatabase
import google.generativeai as genai

# Import Configuration
from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_VECTOR_DIM,
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    TOP_K,
    EMBED_CACHE_PATH,
    PINECONE_REGION,
    PINECONE_CLOUD,
)
# -----------------------------
# Initialize Clients
# -----------------------------
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

pc = Pinecone(api_key=PINECONE_API_KEY)
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating Pinecone index {PINECONE_INDEX_NAME}...")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=PINECONE_VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
    )
index = pc.Index(PINECONE_INDEX_NAME)

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(GEMINI_MODEL)


# -----------------------------
# Embedding Cache
# -----------------------------
if os.path.exists(EMBED_CACHE_PATH):
    with open(EMBED_CACHE_PATH, "r") as f:
        emb_cache = json.load(f)
else:
    emb_cache = {}


def get_cached_embedding(text: str) -> List[float]:
    """Return embedding from cache or compute and store."""
    key = sha256(text.encode()).hexdigest()
    if key in emb_cache:
        return emb_cache[key]
    emb = embed_model.encode(text).tolist()
    emb_cache[key] = emb
    with open(EMBED_CACHE_PATH, "w") as f:
        json.dump(emb_cache, f)
    return emb


# -----------------------------
# Pinecone Query
# -----------------------------
def pinecone_query(text: str, top_k=TOP_K):
    vec = get_cached_embedding(text)
    res = index.query(vector=vec, top_k=top_k, include_metadata=True)
    return res["matches"]


# -----------------------------
# Neo4j Graph Fetch
# -----------------------------
def fetch_graph_context(node_ids: List[str]):
    facts = []
    with driver.session() as session:
        for nid in node_ids:
            q = (
                "MATCH (n:Entity {id:$nid})-[r]-(m:Entity) "
                "RETURN type(r) AS rel, m.id AS id, m.name AS name, "
                "m.description AS description LIMIT 10"
            )
            recs = session.run(q, nid=nid)
            for r in recs:
                facts.append({
                    "source": nid,
                    "rel": r["rel"],
                    "target_id": r["id"],
                    "target_name": r["name"],
                    "target_desc": (r["description"] or "")[:400]
                })
    return facts


# -----------------------------
# Search Summary
# -----------------------------
def search_summary(pinecone_matches, graph_facts, max_nodes=5):
    summary = "### Top Locations & Activities\n"
    for m in pinecone_matches[:max_nodes]:
        meta = m["metadata"]
        summary += f"- {meta.get('name','Unknown')} ({meta.get('type','')}) — {meta.get('city','')} : {meta.get('description','')[:200]}...\n"
    summary += "\n### Related Graph Facts\n"
    for rel in graph_facts[:max_nodes]:
        summary += f"- {rel['source']} -[{rel['rel']}]-> {rel['target_name']}: {rel['target_desc']}\n"
    return summary


# -----------------------------
# Build Prompt
# -----------------------------
def build_prompt(user_query, pinecone_matches, graph_facts):
    summary = search_summary(pinecone_matches, graph_facts)

    # --- Infer mood/theme from the query ---
    query_lower = user_query.lower()

    if any(word in query_lower for word in ["romantic", "couple", "honeymoon", "love"]):
        mood = "romantic and intimate"
        tone = "gentle, poetic, and emotionally appealing"
    elif any(word in query_lower for word in ["adventure", "thrill", "hike", "explore"]):
        mood = "adventurous and energetic"
        tone = "exciting, bold, and full of action"
    elif any(word in query_lower for word in ["family", "kids", "children"]):
        mood = "family-friendly and joyful"
        tone = "warm, inclusive, and comforting"
    elif any(word in query_lower for word in ["relax", "peaceful", "calm", "spa"]):
        mood = "relaxing and rejuvenating"
        tone = "soothing, tranquil, and wellness-focused"
    elif any(word in query_lower for word in ["cultural", "heritage", "history"]):
        mood = "cultural and educational"
        tone = "insightful, respectful, and enriching"
    else:
        mood = "balanced and general-purpose"
        tone = "clear, helpful, and informative"

    # --- Build final dynamic prompt ---
    prompt = f"""
You are a world-class travel planner and itinerary designer.

Your goal is to create a {mood} travel experience for the user.
Use a {tone} tone throughout your answer.

Follow these steps carefully:
1. Understand the user's travel request.
2. Review the available context (locations, activities, attractions) below.
3. Design a complete, personalized, and well-structured itinerary that fits the mood.
4. Include day-by-day plans, recommendations, and reasoning for your choices.
5. Use clear markdown formatting.

### User Query
{user_query}

### Context Summary
{summary}

### Output Instructions
- Create a 4-day itinerary (or adapt if duration is mentioned).
- Include location names, brief descriptions, and why they fit the theme.
- Use bullet points, day headers, and emojis (optional) for style.
- Make sure your tone matches the mood: *{tone}*.

Now provide the final detailed itinerary below:
"""
    return prompt


# -----------------------------
# Gemini Chat
# -----------------------------
def call_gemini_chat(prompt: str):
    response = gemini_model.generate_content(prompt)
    return response.text.strip()


# -----------------------------
# Answer Query
# -----------------------------
def answer_query(user_query):
    matches = pinecone_query(user_query)
    if not matches:
        return "No related destinations found. Try rephrasing your query."

    match_ids = [m["id"] for m in matches]
    graph_facts = fetch_graph_context(match_ids)

    prompt = build_prompt(user_query, matches, graph_facts)
    answer = call_gemini_chat(prompt)
    return answer


# -----------------------------
# Interactive CLI
# -----------------------------
def interactive_chat():
    print("Hybrid Travel Assistant (Gemini + HF + Neo4j + Pinecone)")
    print("Type 'exit' to quit.\n")
    while True:
        query = input("Enter your travel question: ").strip()
        if not query or query.lower() in ("exit", "quit"):
            break
        print("\nThinking...\n")
        answer = answer_query(query)
        print("\n=== Assistant Answer ===\n")
        print(answer)
        print("\n========================\n")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    interactive_chat()
