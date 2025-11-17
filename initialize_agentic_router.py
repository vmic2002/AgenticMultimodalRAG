"""
Initialize Agentic RAG Router

This module initializes and exports the agentic_rag_router for use in other modules.
It sets up connections to Neo4j (Knowledge Graph) and Weaviate (Vector Database),
and creates the AgenticRAGRouter instance.

Assumes the Vector DB and KG are already constructed.
To construct them, run the construct_KG_and_VectorDB.ipynb notebook.
"""

import warnings
warnings.filterwarnings("ignore")

from langchain_community.graphs import Neo4jGraph
from langchain_community.llms import Ollama

from KG_RAG import KG_RAG_Agent
import weaviate
from VectorDB_RAG import VectorDB_RAG
from AgenticRAGRouter import AgenticRAGRouter
import json

# ASSUMES THE VECTOR DB AND KG ARE ALREADY CONSTRUCTED
# TO CONSTRUCT THE VECTOR DB AND KG, RUN THE construct_KG_and_VectorDB.ipynb notebook

cypher_llm = Ollama(
    model="llama3.1:8b",  # or try: "mistral", "llama3", "phi3", "gemma2"
    temperature=0
)

# Neo4j connection details
with open("neo4J_settings.json", "r") as f:
    neo4j_settings = json.load(f)
NEO4J_URI = neo4j_settings["NEO4J_URI"]
NEO4J_USERNAME = neo4j_settings["NEO4J_USERNAME"]
NEO4J_PASSWORD = neo4j_settings["NEO4J_PASSWORD"]

# Initialize Neo4j connection using your existing setup
# refresh_schema=False avoids needing the APOC plugin
graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    refresh_schema=True  # Disable APOC-dependent schema introspection
)

print("✓ Neo4j connection established!")

kg_rag_agent = KG_RAG_Agent(cypher_llm, graph)
print("✓ KG_RAG_Agent initialized!")

print("\n🔧 Connecting to Weaviate...")

try:
    # Connect to local Weaviate instance
    client = weaviate.connect_to_local(
        host="localhost",
        port=8080,
        grpc_port=50051
    )
    print("✅ Connected to Weaviate!")
    print(f"   Weaviate is ready: {client.is_ready()}")
except Exception as e:
    print(f"❌ Error connecting to Weaviate: {e}")
    print("\n💡 Remember to start Weaviate locally:")
    raise

COLLECTION_NAME = "DocumentQA"
vectordb_rag = VectorDB_RAG(client, COLLECTION_NAME)
print("✓ VectorDB_RAG initialized!")

# Initialize the router LLM
router_llm = Ollama(
    model="llama3.1:8b",  # or try: "mistral", "llama3", "phi3", "gemma2"
    temperature=0
)

# Initialize the Agentic RAG Router
# use_hybrid_search=True enables keyword matching + vector similarity (recommended!)
# hybrid_alpha=0.7 means 70% vector, 30% keyword (good default)
agentic_rag_router = AgenticRAGRouter(
    router_llm, 
    kg_rag_agent, 
    vectordb_rag, 
    verbose=True,
    use_hybrid_search=True,  # Enable hybrid search (vector + keyword)
    hybrid_alpha=0.7  # 70% vector, 30% keyword
)
print("✓ AgenticRAGRouter initialized with Hybrid Search!")

