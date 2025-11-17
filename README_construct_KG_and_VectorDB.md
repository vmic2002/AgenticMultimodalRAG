# Knowledge Graph & Vector Database Construction

## 📋 Overview

This notebook (`construct_KG_and_VectorDB.ipynb`) constructs two complementary data structures for a **Multimodal Hybrid Search RAG System**:

1. **Knowledge Graph (Neo4j)** - Captures structured relationships between entities in Q&A pairs
2. **Vector Database (Weaviate)** - Enables semantic search across text and images using CLIP embeddings

Together, these form the foundation for a hybrid search system that combines:
- **Structured retrieval** (KG: entity relationships, graph traversal)
- **Semantic retrieval** (Vector DB: multimodal similarity search)

## 🎯 Purpose

The system is designed for **document Q&A tasks** using the SP-DocVQA dataset, which contains:
- Historical document images (invoices, forms, letters)
- Questions about the documents
- Ground truth answers

The goal is to build a retrieval system that can find relevant documents using both:
- **Semantic similarity** (text and visual features)
- **Structured relationships** (entities and their connections)

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA SOURCES                             │
│  • train_v1.0_withQT.json (Q&A pairs)                       │
│  • Document images (PNG files)                               │
└──────────────────┬──────────────────────────────────────────┘
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
┌────────────────┐  ┌────────────────┐
│  Knowledge     │  │  Vector        │
│  Graph         │  │  Database      │
│  (Neo4j)       │  │  (Weaviate)    │
└────────────────┘  └────────────────┘
         │                   │
         └─────────┬─────────┘
                   ▼
         ┌─────────────────┐
         │  Hybrid Search   │
         │  RAG System      │
         └─────────────────┘
```

### Component 1: Knowledge Graph (Neo4j)

**Purpose:** Extract and store structured information from Q&A pairs as entity-relationship triples.

**Process:**
1. Use **Ollama (local LLM)** with few-shot prompting to extract triples from each Q&A pair
2. Parse triples in format: `(subject, relationship, object)`
3. Store in Neo4j as: `(Subject)-[RELATIONSHIP]->(Object)`
4. Link to document ID for cross-referencing with Vector DB

**Example:**
```
Q: "Which corporation's letterhead is this?"
A: "Brown & Williamson Tobacco Corporation"

Triple extracted:
(letterhead, BELONGS_TO, Brown & Williamson Tobacco Corporation)

Neo4j representation:
(letterhead)-[BELONGS_TO]->(Brown & Williamson Tobacco Corporation)
```


### Component 2: Vector Database (Weaviate)

**Purpose:** Enable multimodal semantic search across document text and images.

**Process:**
1. **OCR extraction:** Extract text from document images using EasyOCR
2. **Confidence filtering:** Filter low-quality OCR (< 0.5 confidence)
3. **Image encoding:** Convert images to base64
4. **Multimodal embedding:** Use CLIP to embed text AND images in shared 512-dim space
5. **Store in Weaviate:** All embeddings combined via weighted average

**Vectorization Architecture:**

```
Document Image ──┐
                 ├──► EasyOCR ──► OCR Text (filtered) ──┐
                 │                                      │
                 │                                      ├──► CLIP Text Encoder ──┐
                 │                                      │                        │
Question + Answer ──────────────────────────────────────┘                        │
                                                                                 │
                                                                                 ├──► Weighted Average
                                                                                 │    (512-dim unified vector)
Document Image ──────────────────────► CLIP Image Encoder ────────────────────┘

Unified Vector = (questionAnswer + ocrText + image) / 3  (if OCR available)
              OR (questionAnswer + image) / 2            (if OCR filtered out)
```

---

## 🔧 Setup Requirements

### 1. Python Environment

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Mac/Linux

# Install dependencies
pip install neo4j langchain langchain-community pydantic
pip install weaviate-client easyocr pillow torch torchvision
```

### 2. Neo4j Database

**Option A: Neo4j Desktop** (Recommended)
1. Download from [neo4j.com/download](https://neo4j.com/download/)
2. Create a new project and database
3. Set password (e.g., `victormicha`)
4. Start the database
5. Connection string: `bolt://localhost:7687`
6. Fill the fields in a file called `neo4J_settings.json`. Also include `MAX_ITEMS`,
    which determines how many of the 39463 data points will be used to build the databases. For testing and our purposes, I recommend a small number such as 10 to be able to visualize each example more easily.
    {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "?",
        "MAX_ITEMS": 8
    }
7. Make sure to install the APOC plugin in Neo4j!
8. In the neo4j.conf file, make sure to include the following 2 lines:
# Apoc is needed to query the schema, which the LLM_CYPHER needs to be able to generate valid cypher queries for GraphRAG
dbms.security.procedures.unrestricted=apoc.*
dbms.security.procedures.allowlist=apoc.*,gds.*

**Option B: Neo4j Cloud (AuraDB)**
- Free tier available at [neo4j.com/cloud/aura](https://neo4j.com/cloud/aura)

### 3. Ollama (Local LLM)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Download llama3.2:1b model
ollama pull llama3.2:1b

# Start Ollama server (keep this running in a separate terminal)
ollama serve

# Verify installation (in another terminal)
ollama list
```

**Important:** Keep `ollama serve` running in a separate terminal while using the notebook.

### 4. Weaviate + CLIP (Docker)

**Start CLIP inference service:**
```bash
docker run -d \
  --name clip-inference \
  -p 8001:8080 \
  -e ENABLE_CUDA=0 \
  semitechnologies/multi2vec-clip:sentence-transformers-clip-ViT-B-32-multilingual-v1
```

**Start Weaviate:**
```bash
docker run -d \
  --name weaviate \
  -p 8080:8080 \
  -p 50051:50051 \
  -e QUERY_DEFAULTS_LIMIT=25 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e PERSISTENCE_DATA_PATH='/var/lib/weaviate' \
  -e DEFAULT_VECTORIZER_MODULE='multi2vec-clip' \
  -e ENABLE_MODULES='multi2vec-clip' \
  -e CLIP_INFERENCE_API='http://host.docker.internal:8001' \
  -e CLUSTER_HOSTNAME='node1' \
  semitechnologies/weaviate:latest
```

**Verify containers are running:**
```bash
docker ps
# Should show both 'weaviate' and 'clip-inference' containers
```

---

## 📁 Data Structure

### Input Data

Please download the data at:
https://rrc.cvc.uab.es/?ch=17&com=downloads
Specifically, the SP-DocVQA

The notebook expects:

```
AgenticMultimodalRAGHybridSearch/
├── data/
│   ├── spdocvqa_qas/
│   │   └── train_v1.0_withQT.json    # Q&A pairs with metadata
│   └── spdocvqa_images/
│       └── *.png                      # Document images
└── construct_KG_and_VectorDB.ipynb
```

### Dataset Format

**train_v1.0_withQT.json** structure:
```json
{
  "data": [
    {
      "questionId": 337,
      "question": "what is the date mentioned in this letter?",
      "question_types": ["handwritten", "form"],
      "image": "documents/xnbl0037_1.png",
      "docId": 279,
      "ucsf_document_id": "xnbl0037",
      "ucsf_document_page_no": "1",
      "answers": ["1/8/93"],
      "data_split": "train"
    }
  ]
}
```

---

## 🚀 Usage

# Simply Run all the cells and the VectorDB and the Knowledge Graph will be constructed!

---

## 📊 Schema Details

### Weaviate Vector Database

**Collection: `DocumentQA`**

**Properties:**

| Property | Type | Vectorized? | Description |
|----------|------|-------------|-------------|
| `docId` | INT | ❌ | Document ID (links to KG) |
| `question` | TEXT | ❌ | Question text (stored for display) |
| `answer` | TEXT | ❌ | Answer text (stored for display) |
| `questionAnswer` | TEXT | ✅ | Combined Q&A (33.3% weight) |
| `ocrText` | TEXT | ✅ | Filtered OCR text (33.3% weight) |
| `ocrConfidence` | NUMBER | ❌ | OCR quality metric (0.0-1.0) |
| `image` | BLOB | ✅ | Base64 image (33.3% weight) |

**Vectorizer:** `multi2vec-clip` (CLIP ViT-B-32)

**Vector Index:** HNSW with cosine distance

---

## 📚 References

### Datasets
- **SP-DocVQA:** https://rrc.cvc.uab.es/?ch=17&com=downloads

### Technologies
- **Neo4j:** [https://neo4j.com/docs/](https://neo4j.com/docs/)
- **Weaviate:** [https://weaviate.io/developers/weaviate](https://weaviate.io/developers/weaviate)
- **CLIP:** [https://github.com/openai/CLIP](https://github.com/openai/CLIP)
- **LangChain:** [https://python.langchain.com/docs/get_started/introduction](https://python.langchain.com/docs/get_started/introduction)
- **Ollama:** [https://ollama.com/](https://ollama.com/)
- **EasyOCR:** [https://github.com/JaidedAI/EasyOCR](https://github.com/JaidedAI/EasyOCR)