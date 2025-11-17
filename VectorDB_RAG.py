import weaviate


class VectorDB_RAG:
    def __init__(self, client, COLLECTION_NAME):
        self.client = client
        self.COLLECTION_NAME = COLLECTION_NAME

    def search_vector_db(self, query_text, limit=2, verbose=False):
        """
        Multimodal semantic search using CLIP embeddings
        
        🔍 How it works with multi2vec-clip:
        1. Query text → CLIP text encoder → 512-dim query vector
        
        2. **IMPORTANT: Unified Vector per Object (Weighted Average)**
        When an object is stored, Weaviate COMBINES vectorized CLIP embeddings via 
        WEIGHTED AVERAGE (mean pooling) into a SINGLE 512-dim unified vector:
        - questionAnswer (512-dim) + ocrText (512-dim) + image (512-dim)
        - Formula: v = (vec₁ + vec₂ + vec₃) / 3
        - ⚠️ question & answer stored but NOT vectorized (avoid redundancy)
        → Result: Balanced 512-dim unified vector (33.3% Q&A, 33.3% OCR, 33.3% image)
        
        3. Search compares query vector to this unified vector:
        - similarity(query_vector, unified_object_vector)
        - The unified vector contains information from BOTH text AND image!
        - ✅ Text query IS compared against image embeddings!
        
        4. Weaviate returns top-k results by cosine similarity
        
        ✅ Why .near_text() works for multimodal search:
        - .near_text() searches against the UNIFIED vector (not just text!)
        - The unified vector includes image information via CLIP
        - Text query "medical form" matches:
            * Text content (OCR, Q&A)
            * Visual appearance (layout, structure)
        - True cross-modal retrieval across ALL modalities!
        
        Args:
            query_text: Text query to search for
            limit: Number of results to return
            
        Returns:
            list: Search results ranked by similarity (considering ALL modalities)
        """
        try:
            collection = self.client.collections.get(self.COLLECTION_NAME)
            
            # Perform semantic search
            # Include return_metadata to get distance/similarity scores
            response = collection.query.near_text(
                query=query_text,
                limit=limit,
                return_properties=["docId", "question", "answer", "ocrText"],
                return_metadata=["distance"]  # Request distance metadata
            )
            
            if verbose:
                print(f"\n🔍 Search results for: '{query_text}'")
                print("="*80)
            
            # Enumerate results starting from 1
            # enumerate(iterable, start=1) returns (index, item) pairs
            # where index starts at 1 instead of 0
            #for i, obj in enumerate[GroupByObject[WeaviateProperties, CrossReferences] | GroupByObject[WeaviateProperties, None] | Object[WeaviateProperties, CrossReferences] | Object[WeaviateProperties, None]](response.objects, 1):

            str_response = ""
            for i, obj in enumerate(response.objects, 1):
                str_response += f"\n{i}. Doc {obj.properties['docId']}"
                str_response += f"   Question: {obj.properties['question']}"
                str_response += f"   Answer: {obj.properties['answer']}"
                str_response += f"   OCR Preview: {obj.properties['ocrText'][:100]}..."
                #response += f"   Distance: {obj.metadata.distance:.4f}" if obj.metadata.distance is not None else "   Distance: N/A"
                #print(f"   Distance: {obj.metadata.distance:.4f}" if obj.metadata.distance is not None else "   Distance: N/A")
                #print(f"   OCR Preview: {obj.properties['ocrText'][:100]}...")
                # Distance: Cosine distance (NOT cosine similarity!)
                # cosine_distance = 1 - cosine_similarity
                # Lower is better: 0.0 = identical, 1.0 = orthogonal, 2.0 = opposite
                str_response += "\n"
            return str_response
            
        except Exception as e:
            print(f"❌ Error searching: {e}")
            return []
    
    def search_hybrid(self, query_text, limit=2, alpha=0.5, verbose=False):
        """
        Hybrid search combining vector similarity + BM25 keyword matching
        
        🔍 How hybrid search works:
        1. Vector Search (semantic): Uses CLIP embeddings for semantic similarity
        2. BM25 Search (keyword): Traditional keyword matching on text fields
        3. Fusion: Combines both scores using the alpha parameter
        
        Alpha parameter controls the balance:
        - alpha=0.0  → Pure BM25 keyword search (exact matches only)
        - alpha=0.5  → Balanced hybrid (50% vector, 50% keyword)
        - alpha=0.75 → Mostly vector, some keyword boost
        - alpha=1.0  → Pure vector search (same as search_vector_db)
        
        📈 Why hybrid search is better:
        - Catches exact term matches (e.g., "document 280", "June 11, 1990")
        - Still understands semantic meaning (e.g., "tobacco" matches "cigarette")
        - Best of both worlds for RAG systems!
        
        Performance: ~10-20% slower than pure vector search, but often better results
        
        Args:
            query_text: Text query to search for
            limit: Number of results to return
            alpha: Balance between keyword (0) and vector (1) search
            verbose: Print detailed search information
            
        Returns:
            str: Formatted search results ranked by hybrid score
        """
        try:
            collection = self.client.collections.get(self.COLLECTION_NAME)
            
            # Perform hybrid search (vector + keyword)
            response = collection.query.hybrid(
                query=query_text,
                limit=limit,
                alpha=alpha,  # Balance between keyword (0) and vector (1)
                return_properties=["docId", "question", "answer", "ocrText"],
                return_metadata=["score"]  # Hybrid score (combination of both)
            )
            
            if verbose:
                print(f"\n🔍 Hybrid search results for: '{query_text}' (alpha={alpha})")
                print(f"   📊 Alpha: {alpha} ({int((1-alpha)*100)}% keyword, {int(alpha*100)}% vector)")
                print("="*80)
            
            str_response = ""
            for i, obj in enumerate(response.objects, 1):
                str_response += f"\n{i}. Doc {obj.properties['docId']}"
                str_response += f"   Question: {obj.properties['question']}"
                str_response += f"   Answer: {obj.properties['answer']}"
                str_response += f"   OCR Preview: {obj.properties['ocrText'][:100]}..."
                
                # Show hybrid score if verbose
                if verbose and obj.metadata.score is not None:
                    print(f"{i}. Doc {obj.properties['docId']} - Score: {obj.metadata.score:.4f}")
                
                str_response += "\n"
            
            return str_response
            
        except Exception as e:
            print(f"❌ Error in hybrid search: {e}")
            return []