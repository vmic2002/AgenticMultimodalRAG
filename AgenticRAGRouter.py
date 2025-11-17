# this is the class that will route the query to the appropriate RAG
# either the KG or the vector db

class AgenticRAGRouter:
    def __init__(self, router_llm, kg_rag_agent, vector_db_rag, verbose=True, use_hybrid_search=True, hybrid_alpha=0.7):
        self.router_llm = router_llm
        self.kg_rag_agent = kg_rag_agent
        self.vector_db_rag = vector_db_rag
        self.verbose = verbose
        self.use_hybrid_search = use_hybrid_search
        self.hybrid_alpha = hybrid_alpha  # 0=keyword, 1=vector, 0.7=mostly vector with keyword boost
        
        # Router template with few-shot prompting
        self.router_llm_template = """You are a query routing assistant. Your job is to classify user queries into one of two categories:

**KG** (Knowledge Graph): For queries about structured relationships, metadata, entities, and specific factual lookups like:
- Who sent/received a document
- What is the date/subject/number of a document
- Who was CCd on a document
- What letterhead/heading does a document have
- Specific entity relationships and document metadata

**VECTOR** (Vector Database): For queries about semantic content, detailed text, summaries, or visual features like:
- General content or topic of a document
- Summarization requests
- Questions about document appearance or visual features
- Queries requiring semantic understanding of unstructured text
- Questions about handwritten content or OCR text

**Examples:**

Query: "Who sent the document dated June 11, 1990?"
Classification: KG
Reason: Asks about a specific relationship (Sent_By) and date metadata.

Query: "What is the subject of document 280?"
Classification: KG
Reason: Asks about metadata property (Has_Subject).

Query: "Find all documents from 1993"
Classification: KG
Reason: Asks about date metadata (Has_Date).

Query: "Who was CCd on document 280?"
Classification: KG
Reason: Asks about a specific relationship (CCd_On).

Query: "What does the handwritten note in the document say?"
Classification: VECTOR
Reason: Requires OCR text and semantic understanding of unstructured content.

Query: "Find documents about flavor development"
Classification: VECTOR
Reason: Requires semantic search across document content.

Query: "Summarize the main points of document 297"
Classification: VECTOR
Reason: Requires understanding of detailed content and summarization.

Query: "What is the letterhead of document 280?"
Classification: KG
Reason: Asks about specific metadata property (Has_Letterhead).

Query: "Find forms from the 1990s"
Classification: VECTOR
Reason: Requires semantic understanding and visual recognition of document types.

Query: "What is the date in document 279?"
Classification: KG
Reason: Asks about specific metadata property (Has_Date).

**Now classify this query:**

Query: "{query}"
Classification: """

    def route_query(self, query):
        """
        Routes the query to the appropriate RAG system (KG or Vector DB).
        
        Args:
            query: User's natural language query
            
        Returns:
            dict: Contains 'route' (KG or VECTOR), 'response', and routing info
        """
        try:
            # Step 1: Use LLM to classify the query
            prompt = self.router_llm_template.format(query=query)
            llm_response = self.router_llm.invoke(prompt)
            
            # Extract classification from LLM response
            # Look for "KG" or "VECTOR" in the response
            classification = self._extract_classification(llm_response)
            
            if self.verbose:
                print(f"\n{'='*80}")
                print(f"🔀 Query Routing Decision")
                print(f"{'='*80}")
                print(f"Query: {query}")
                print(f"Route: {classification}")
                print(f"{'='*80}\n")
            
            # Step 2: Route to appropriate system
            if classification == "KG":
                if self.verbose:
                    print("📊 Routing to Knowledge Graph RAG...")
                    
                # Use the my_cypher_chain method which is more reliable
                cypher_query, kg_response = self.kg_rag_agent.my_cypher_chain(query)
                
                if self.verbose:
                    print(f"\n🔍 Generated Cypher Query:\n{cypher_query}")
                    print(f"\n📋 KG Response:\n{kg_response}\n")
                
                return {
                    "route": "KG",
                    "query": query,
                    "cypher_query": cypher_query,
                    "response": kg_response,
                    "raw_response": kg_response
                }
                
            elif classification == "VECTOR":
                if self.verbose:
                    search_type = "Hybrid" if self.use_hybrid_search else "Vector"
                    print(f"🔍 Routing to Vector Database RAG ({search_type} Search)...")
                    
                # Choose between hybrid or pure vector search
                if self.use_hybrid_search:
                    vector_response = self.vector_db_rag.search_hybrid(
                        query, 
                        alpha=self.hybrid_alpha,
                        verbose=self.verbose
                    )
                    search_method = f"Hybrid (α={self.hybrid_alpha})"
                else:
                    vector_response = self.vector_db_rag.search_vector_db(
                        query,
                        verbose=self.verbose
                    )
                    search_method = "Vector Only"
                
                if self.verbose:
                    print(f"\n📋 Vector DB Response:\n{vector_response}\n")
                
                return {
                    "route": "VECTOR",
                    "query": query,
                    "search_method": search_method,
                    "response": vector_response,
                    "raw_response": vector_response
                }
                
            else:
                # Default to Vector DB if classification is unclear
                if self.verbose:
                    search_type = "Hybrid" if self.use_hybrid_search else "Vector"
                    print(f"⚠️ Unclear classification, defaulting to Vector Database ({search_type})...")
                    
                # Use same search method as VECTOR route
                if self.use_hybrid_search:
                    vector_response = self.vector_db_rag.search_hybrid(
                        query, 
                        alpha=self.hybrid_alpha,
                        verbose=self.verbose
                    )
                else:
                    vector_response = self.vector_db_rag.search_vector_db(
                        query,
                        verbose=self.verbose
                    )
                
                return {
                    "route": "VECTOR (default)",
                    "query": query,
                    "response": vector_response,
                    "raw_response": vector_response
                }
                
        except Exception as e:
            print(f"❌ Error routing query: {e}")
            return {
                "route": "ERROR",
                "query": query,
                "response": None,
                "error": str(e)
            }
    
    def _extract_classification(self, llm_response):
        """
        Extracts KG or VECTOR classification from LLM response.
        
        Args:
            llm_response: Raw response from the LLM
            
        Returns:
            str: "KG" or "VECTOR" or "UNKNOWN"
        """
        response_upper = str(llm_response).upper()
        
        # Look for explicit classification
        if "CLASSIFICATION: KG" in response_upper or response_upper.strip().startswith("KG"):
            return "KG"
        elif "CLASSIFICATION: VECTOR" in response_upper or response_upper.strip().startswith("VECTOR"):
            return "VECTOR"
        
        # Fallback: check which keyword appears first
        kg_pos = response_upper.find("KG")
        vector_pos = response_upper.find("VECTOR")
        
        if kg_pos != -1 and (vector_pos == -1 or kg_pos < vector_pos):
            return "KG"
        elif vector_pos != -1:
            return "VECTOR"
        
        return "UNKNOWN"