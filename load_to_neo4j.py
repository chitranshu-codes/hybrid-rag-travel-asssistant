import json
import logging
from tqdm import tqdm
from neo4j import GraphDatabase
import os
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, DATA_FILE

# -----------------------------
# LOGGING SETUP
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# -----------------------------
# CONNECT TO DATABASE
# -----------------------------
try:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    logger.info("✅ Connected to Neo4j successfully.")
except Exception as e:
    logger.error(f"❌ Failed to connect to Neo4j: {e}")
    raise SystemExit(e)


# -----------------------------
# NEO4J OPERATIONS
# -----------------------------
def create_constraints(tx):
    """Ensure Entity node IDs are unique."""
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE")
    logger.debug("Created constraint for Entity nodes.")


def upsert_node(tx, node):
    """Create or update a node with given labels and properties."""
    labels = [node.get("type", "Unknown"), "Entity"]
    label_cypher = ":" + ":".join(labels)
    props = {
        k: v
        for k, v in node.items()
        if k not in ("connections",) and isinstance(v, (str, int, float, list))
    }

    tx.run(
        f"MERGE (n{label_cypher} {{id: $id}}) "
        "SET n += $props",
        id=node["id"],
        props=props,
    )


def create_relationship(tx, source_id, rel):
    """Create relationships between nodes."""
    rel_type = rel.get("relation", "RELATED_TO").upper()
    target_id = rel.get("target")

    if not target_id:
        return

    cypher = (
        "MATCH (a:Entity {id: $source_id}), (b:Entity {id: $target_id}) "
        f"MERGE (a)-[r:{rel_type}]->(b) "
        "RETURN r"
    )
    tx.run(cypher, source_id=source_id, target_id=target_id)


# -----------------------------
# MAIN FUNCTION
# -----------------------------
def main():
    if not os.path.exists(DATA_FILE):
        logger.error(f"❌ Data file not found: {DATA_FILE}")
        return

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    logger.info(f"📦 Loaded {len(nodes)} nodes from {DATA_FILE}")

    with driver.session() as session:
        session.execute_write(create_constraints)

        # Create nodes
        for node in tqdm(nodes, desc="🧱 Creating nodes"):
            try:
                session.execute_write(upsert_node, node)
            except Exception as e:
                logger.warning(f"⚠️ Skipped node {node.get('id')} - {e}")

        # Create relationships
        for node in tqdm(nodes, desc="🔗 Creating relationships"):
            conns = node.get("connections", [])
            for rel in conns:
                try:
                    session.execute_write(create_relationship, node["id"], rel)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to link {node['id']} -> {rel}: {e}")

    logger.info("✅ Successfully loaded all data into Neo4j.")


if __name__ == "__main__":
    try:
        main()
    finally:
        driver.close()
        logger.info("🔒 Neo4j connection closed.")
