# Hybrid RAG Travel Assistant (Gemini + Neo4j + Pinecone)

This project is a sophisticated travel itinerary planner that demonstrates a **Hybrid Retrieval-Augmented Generation (RAG)** architecture. It combines the power of semantic search (Pinecone) and graph-based context (Neo4j) to provide rich, factual context to a generative AI (Google Gemini) for creating personalized and detailed travel plans.

The assistant anwers user queries (e.g., "I want a 4-day romantic trip in Vietnam") by first finding relevant locations and then exploring their connections to build a complete, grounded answer.

## 🚀 Features

  * **Hybrid Retrieval:** Uses **Pinecone** for fast, semantic search to find relevant locations/activities and **Neo4j** to retrieve specific, connected facts about those results (e.g., "what's near this place?", "what activities are in this city?").
  * **Dynamic Prompting:** The system analyzes the user's query to infer a "mood" (e.g., romantic, adventure, family-friendly) and instructs the LLM to tailor its tone and recommendations accordingly.
  * **RAG Architecture:** Provides grounded, factual answers based on a custom knowledge base, significantly reducing LLM "hallucinations."
  * **Knowledge Graph:** All data is structured in a Neo4j graph, allowing for complex queries about relationships between places, activities, and regions.
  * **Interactive CLI:** Comes with a simple-to-use command-line interface to chat with the assistant.

## 🏛️ Architecture / How it Works

The application follows a multi-step hybrid retrieval process before generation:

1.  **User Query:** The user asks a question (e.g., "I want an adventurous trip to Vietnam").
2.  **Embedding:** The query is converted into a vector embedding using `all-MiniLM-L6-v2`.
3.  **Vector Search (Pinecone):** The embedding is used to search Pinecone for the `TOP_K` most semantically similar locations or activities from the knowledge base.
4.  **Graph Fetch (Neo4j):** The unique IDs from the Pinecone results are used to query the Neo4j graph. This fetches 1st-degree connections, finding related entities and facts (e.g., "Sapa Trekking" `IS_LOCATED_IN` "Lao Cai Province" and is `RELATED_TO` "Local Village Homestay").
5.  **Context Building:** The results from both Pinecone (the "what") and Neo4j (the "connections") are compiled into a "Context Summary."
6.  **Dynamic Prompting:** The script detects the "adventure" theme and builds a system prompt instructing Gemini to use an "energetic and exciting" tone.
7.  **LLM Generation (Gemini):** The user query, the rich context summary, and the dynamic prompt are all sent to the Gemini API, which generates a detailed, day-by-day itinerary that is grounded in the retrieved data.

## 🛠️ Technology Stack

  * **Generative AI:** Google Gemini
  * **Vector Database:** Pinecone (Serverless)
  * **Graph Database:** Neo4j
  * **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
  * **Backend:** Python
  * **Core Libraries:** `google-generativeai`, `pinecone-client`, `neo4j`, `sentence-transformers`, `tqdm`

## ⚙️ Setup and Installation

### 1\. Prerequisites

You must have active accounts and API keys for:

  * Google AI Studio (for a Gemini API Key)
  * Pinecone
  * A Neo4j instance (e.g., a free AuraDB instance)

### 2\. Clone the Repository

```bash
git clone https://github.com/chitranshu-codes/hybrid-rag-travel-asssistant.git
cd hybrid-rag-travel-asssistant
```

### 3\. Set Up Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 4\. Install Dependencies

install requirements.txt:

```bash
pip install -r requirements.txt
```

### 5\. Configure Environment Variables

The `config.py` script loads configuration from your environment variables. You must set these in your terminal or a `.env` file.

**Important:** Do NOT hardcode your keys in `config.py`.

```bash
export GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
export PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
export NEO4J_URI="YOUR_NEO4J_BOLT_URI"        # e.g., "neo4j+s://xxxx.databases.neo4j.io"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="YOUR_NEO4J_PASSWORD"
export PINECONE_INDEX_NAME="vietnam-travel-index"
```

### 6\. Prepare Your Data

You need a `vietnam_travel_dataset.json` file in the root directory.

## 🚀 Usage

### Step 1: Load Data into Databases

These are one-time scripts to populate your databases. You must run them in this order.

**1. Upload to Neo4j (Graph Database):**
This script creates all the nodes and their relationships.

```bash
python upload_neo4j.py
```

**2. Upload to Pinecone (Vector Database):**
This script embeds the `semantic_text` for each item and uploads it to Pinecone.

```bash
python pinecone_upload.py
```

### Step 2: Run the Chat Assistant

Once your data is loaded, you can start the interactive assistant:

```bash
python hybrid_chat.py
```

You will see a prompt. Ask your travel question and the assistant will retrieve context and generate a detailed plan.

```
Hybrid Travel Assistant (Gemini + HF + Neo4j + Pinecone)
Type 'exit' to quit.

Enter your travel question: I'm looking for a 4-day relaxed family trip

Thinking...

=== Assistant Answer ===

(Gemini's detailed, 4-day itinerary will appear here)

========================

Enter your travel question:
```

## 📂 File Structure

```
.
├── config.py                 # Manages all API keys and configuration
├── upload_neo4j.py           # One-time script to load data into Neo4j
├── pinecone_upload.py        # One-time script to embed and load data into Pinecone
├── hybrid_chat.py            # The main interactive chat application
├── vietnam_travel_dataset.json # Your source data file (NOT included)
├── emb_cache.json            # Caches embeddings locally (auto-generated)
└── requirements.txt          # Python dependencies
```
