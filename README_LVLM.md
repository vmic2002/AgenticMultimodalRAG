# Multimodal RAG with LLaVA - Complete Guide

## 🎯 What This Does

The complete end-to-end system:
1. **User asks a question** (e.g., "What is the subject of document 280?")
2. **AgenticRAGRouter** intelligently retrieves from KG or Vector DB
3. **Document extraction** finds relevant document IDs and images
4. **LLaVA (LVLM)** generates a natural language answer using:
   - Original question
   - Retrieved RAG context
   - Document image (when available)

## 🚀 Quick Start

### Step 1: Install Ollama Python Client

```bash
pip install ollama
```

### Step 2: Pull LLaVA Model

```bash
ollama pull llava-phi3
```

This system uses `llava-phi3` which is a lightweight model (~2.9GB, ~3.8B parameters) that works well on most systems without crashing.

### Step 3: Run the System

```bash
python LVLM_Generation_with_Multimodal_RAG.py
```

## 📊 Workflow Visualization

```
┌─────────────────┐
│  User Question  │
└────────┬────────┘
         │
         ▼
┌──────────────────────────────┐
│  AgenticRAGRouter            │
│  ├─ Router LLM classifies    │
│  └─ Routes to KG or Vector   │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│  RAG Retrieval               │
│  ├─ KG: Cypher query         │
│  └─ Vector: Hybrid search    │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│  Document Extraction         │
│  └─ Extract Doc IDs          │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│  Image Loading               │
│  └─ Load doc images (if any) │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│  LLaVA Answer Generation     │
│  ├─ Multimodal: text + image │
│  └─ Text-only: just context  │
└────────┬─────────────────────┘
         │
         ▼
┌─────────────────┐
│  Final Answer   │
└─────────────────┘

--

## 📚 Files Involved

- `LVLM_Generation_with_Multimodal_RAG.py` - Main system
- `initialize_agentic_router.py` - AgenticRAGRouter initialization
- `AgenticRAGRouter.py` - Query routing
- `KG_RAG.py` - Knowledge graph queries
- `VectorDB_RAG.py` - Vector database + hybrid search
- `doc_id_to_image_path.json` - Document image mapping
- `construct_KG_and_VectorDB.ipynb` - Constructs the DBs
- `interactive_test_router.py` - To manually test the router LLM with user inputs
- `routing_accuracy_test.py` - Test accuracy of router LLM
- `visualize_documents.py` - To see documents given their doc ID

## 🎉 Summary

**complete multimodal RAG system** with:
- ✅ Intelligent query routing (KG vs Vector)
- ✅ Hybrid search (semantic + keyword)
- ✅ Multimodal answer generation (text + images)
- ✅ Graceful fallback to text-only
- ✅ Comprehensive latency tracking
- ✅ User-friendly interface