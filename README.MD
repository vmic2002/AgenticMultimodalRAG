# Enterprise RAG System - Multimodal Hybrid Search

A modular prototype of an **Enterprise Retrieval-Augmented Generation (RAG) system** that supports text and image inputs, builds a searchable knowledge graph, and enables hybrid search using keyword and vector retrieval (Graph RAG System).

Everything runs locally! The vector DB, the Neo4J KG, the LLM, and the LVLM all run locally !

## 🚀 Quick Start

### Step 1: Setup & Database Construction

**Start here:** Follow the instructions in [`README_construct_KG_and_VectorDB.md`](README_construct_KG_and_VectorDB.md) to:
- Set up all dependencies
- Create the Knowledge Graph (Neo4j)
- Create the Vector Database (Weaviate)

### Step 2: Test the Router

Once the databases are created, test the agentic router:
- **`routing_accuracy_test.py`** - Test router accuracy
- **`interactive_test_router.py`** - Try your own queries with the router

### Step 3: Run the Complete System

Run the full multimodal RAG pipeline:
- **`LVLM_Generation_with_Multimodal_RAG.py`** - Start with **demo mode** to understand the system

### Step 4: Evaluate System Performance (still in testing, not yet finished)

Evaluate your RAG system using DeepEval with local Ollama (no API costs):
```bash
python evaluation_pipeline.py --quick-test
```

> ⚠️ **Note**: DeepEval integration is currently in testing. See evaluation report for status.

See **`README_Evaluation.md`** for details.

## 📚 Additional Documentation

- **`README_LVLM.md`** - Detailed information about each file and the LVLM integration
- **`README_construct_KG_and_VectorDB.md`** - Complete setup and database construction guide
- **`README_Evaluation.md`** - Evaluation pipeline guide