"""
LVLM Answer Generation with Multimodal RAG

Complete workflow:
1. User asks a question
2. AgenticRAGRouter retrieves from KG or Vector DB
3. Extract document ID from results (if available)
4. Load document image using doc_id_to_image_path.json
5. LLaVA generates final answer using RAG context + image

Handles both:
- Multimodal: Question + RAG context + Document image
- Text-only: Question + RAG context (when no document available)
"""

import warnings
warnings.filterwarnings("ignore")

import json
import time
import re
from pathlib import Path

from initialize_agentic_router import agentic_rag_router

try:
    import ollama
except ImportError:
    print("❌ Ollama library not found. Install with: pip install ollama")
    print("   Also ensure Ollama is installed: https://ollama.ai/")
    exit(1)


class LVLMAnswerGenerator:
    def __init__(self, doc_id_map_path="doc_id_to_image_path.json"):
        """
        Initialize LVLM Answer Generator
        
        Uses llava-phi3 model (lightweight, ~3.8B parameters)
        
        Args:
            doc_id_map_path: Path to JSON mapping doc IDs to image paths
        """
        self.llava_model = "llava-phi3"
        self.doc_id_map_path = doc_id_map_path
        
        # Load document ID to image path mapping
        self.doc_id_to_image = {}
        if Path(doc_id_map_path).exists():
            with open(doc_id_map_path, "r") as f:
                self.doc_id_to_image = json.load(f)
            print(f"✅ Loaded {len(self.doc_id_to_image)} document image mappings")
        else:
            print(f"⚠️  Warning: {doc_id_map_path} not found. Image-based answers will be unavailable.")
        
        # Check if LLaVA model is available
        self._check_llava_availability()
    
    def _check_llava_availability(self):
        """Check if llava-phi3 model is available in Ollama"""
        try:
            models = ollama.list()
            available_models = [model['name'] for model in models.get('models', [])]
            
            if not any('llava-phi3' in model for model in available_models):
                print(f"\n⚠️  llava-phi3 model not found in Ollama!")
                print(f"\n📥 To install it, run:")
                print(f"   ollama pull llava-phi3")
                print(f"\n💡 llava-phi3 is a lightweight model (~2.9GB) that works well on most systems")
                
                response = input(f"\n❓ Continue anyway? (y/n): ").strip().lower()
                if response != 'y':
                    exit(1)
            else:
                print(f"✅ llava-phi3 model is ready!")
        except Exception as e:
            print(f"⚠️  Could not verify Ollama models: {e}")
    
    def extract_doc_ids(self, rag_result):
        """
        Extract document IDs from RAG results
        
        Args:
            rag_result: Result dictionary from AgenticRAGRouter
            
        Returns:
            list: List of document IDs found
        """
        doc_ids = []
        
        # Extract from response text
        response_text = str(rag_result.get('response', ''))
        
        # Pattern 1: "Doc 280" or "Doc 314"
        pattern1 = re.findall(r'Doc\s+(\d+)', response_text)
        doc_ids.extend(pattern1)
        
        # Pattern 2: "document 280" or "document: 280"
        pattern2 = re.findall(r'document[:\s]+(\d+)', response_text, re.IGNORECASE)
        doc_ids.extend(pattern2)
        
        # Pattern 3: standalone numbers that look like doc IDs (279, 280, etc.)
        pattern3 = re.findall(r'\b(2\d{2}|3\d{2})\b', response_text)
        doc_ids.extend(pattern3)
        
        # Remove duplicates and return
        return list(set(doc_ids))
    
    def get_document_images(self, doc_ids):
        """
        Get image paths for document IDs
        
        Args:
            doc_ids: List of document IDs
            
        Returns:
            list: List of (doc_id, image_path) tuples for valid documents
        """
        valid_images = []
        
        for doc_id in doc_ids:
            if doc_id in self.doc_id_to_image:
                image_path = self.doc_id_to_image[doc_id]
                if Path(image_path).exists():
                    valid_images.append((doc_id, image_path))
                else:
                    print(f"   ⚠️  Image not found for Doc {doc_id}: {image_path}")
            else:
                print(f"   ⚠️  No image mapping for Doc {doc_id}")
        
        return valid_images
    
    def generate_answer_with_llava(self, query, rag_result, image_path=None, doc_id=None):
        """
        Generate answer using LLaVA with RAG context and optional image
        
        Args:
            query: Original user question
            rag_result: Result from AgenticRAGRouter
            image_path: Optional path to document image
            doc_id: Optional document ID for context
            
        Returns:
            str: Generated answer
        """
        # Prepare context from RAG results
        route = rag_result.get('route', 'UNKNOWN')
        response = rag_result.get('response', '')
        
        # Build prompt based on whether we have an image
        if image_path:
            prompt = f"""You are an expert document analyst. You have been given a question about a document and some retrieved context from a database.

Question: {query}

Retrieved Context from {route}:
{response}

The document image is provided. Please analyze both the retrieved context and the visual information in the document image to provide a comprehensive, accurate answer.

Your answer should:
1. Directly answer the question
2. Reference specific information from both the context and image
3. Be concise but complete
4. Cite specific details from the document when relevant

Answer:"""
        else:
            prompt = f"""You are an expert document analyst. You have been given a question and some retrieved context from a database.

Question: {query}

Retrieved Context from {route}:
{response}

Note: No document image is available for this query. Please provide an answer based solely on the retrieved context.

Your answer should:
1. Directly answer the question
2. Reference specific information from the context
3. Be concise but complete
4. Indicate if more information would be needed

Answer:"""
        
        # Call LLaVA
        try:
            if image_path:
                # Multimodal: text + image
                response = ollama.chat(
                    model=self.llava_model,
                    messages=[{
                        'role': 'user',
                        'content': prompt,
                        'images': [image_path]
                    }]
                )
            else:
                # Text-only mode
                response = ollama.chat(
                    model=self.llava_model,
                    messages=[{
                        'role': 'user',
                        'content': prompt
                    }]
                )
            
            return response['message']['content']
        
        except Exception as e:
            print(f"\n❌ Error calling LLaVA: {e}")
            return f"Error generating answer: {e}"
    
    def answer_question(self, query, verbose=True):
        """
        Complete pipeline: RAG retrieval → Image loading → LLaVA answer generation
        
        Args:
            query: User's question
            verbose: Print progress information
            
        Returns:
            dict: Complete result with answer and timing information
        """
        total_start = time.time()
        timing = {}
        
        if verbose:
            print("\n" + "="*80)
            print("🚀 Starting Multimodal RAG Answer Generation Pipeline")
            print("="*80)
            print(f"\n📝 Question: {query}")
        
        # Step 1: RAG Retrieval
        if verbose:
            print(f"\n{'─'*80}")
            print("📊 Step 1: Retrieving relevant information from databases...")
        
        rag_start = time.time()
        rag_result = agentic_rag_router.route_query(query)
        rag_time = (time.time() - rag_start) * 1000
        timing['rag_retrieval'] = rag_time
        
        if verbose:
            print(f"✅ Retrieved from: {rag_result['route']}")
            print(f"⏱️  RAG Latency: {rag_time:.2f}ms")
        
        # Step 2: Extract Document IDs
        if verbose:
            print(f"\n{'─'*80}")
            print("🔍 Step 2: Extracting document IDs from results...")
        
        extract_start = time.time()
        doc_ids = self.extract_doc_ids(rag_result)
        extract_time = (time.time() - extract_start) * 1000
        timing['doc_extraction'] = extract_time
        
        if verbose:
            if doc_ids:
                print(f"✅ Found document IDs: {', '.join(doc_ids)}")
            else:
                print("⚠️  No document IDs found in results")
            print(f"⏱️  Extraction Latency: {extract_time:.2f}ms")
        
        # Step 3: Load Document Images
        image_path = None
        doc_id = None
        
        if doc_ids:
            if verbose:
                print(f"\n{'─'*80}")
                print("📸 Step 3: Loading document images...")
            
            image_start = time.time()
            valid_images = self.get_document_images(doc_ids)
            image_time = (time.time() - image_start) * 1000
            timing['image_loading'] = image_time
            
            if valid_images:
                # Use the first valid image
                doc_id, image_path = valid_images[0]
                if verbose:
                    print(f"✅ Using document: Doc {doc_id}")
                    print(f"📁 Image path: {image_path}")
                    if len(valid_images) > 1:
                        print(f"   (Found {len(valid_images)} images, using first)")
            else:
                if verbose:
                    print("⚠️  No valid images found, proceeding with text-only mode")
            
            if verbose:
                print(f"⏱️  Image Loading Latency: {image_time:.2f}ms")
        else:
            if verbose:
                print(f"\n{'─'*80}")
                print("📸 Step 3: No document IDs found, proceeding with text-only mode")
        
        # Step 4: Generate Answer with LLaVA
        if verbose:
            print(f"\n{'─'*80}")
            if image_path:
                print(f"🤖 Step 4: Generating answer with LLaVA (multimodal: text + image)...")
            else:
                print(f"🤖 Step 4: Generating answer with LLaVA (text-only mode)...")
        
        llava_start = time.time()
        answer = self.generate_answer_with_llava(query, rag_result, image_path, doc_id)
        llava_time = (time.time() - llava_start) * 1000
        timing['llava_generation'] = llava_time
        
        if verbose:
            print(f"✅ Answer generated!")
            print(f"⏱️  LLaVA Latency: {llava_time:.2f}ms")
        
        # Calculate total time
        total_time = (time.time() - total_start) * 1000
        timing['total'] = total_time
        
        # Display final answer
        if verbose:
            print(f"\n{'='*80}")
            print("📋 FINAL ANSWER:")
            print("="*80)
            print(answer)
            print("="*80)
        
        # Display timing summary
        if verbose:
            print(f"\n{'='*80}")
            print("⏱️  TIMING SUMMARY:")
            print("="*80)
            print(f"  RAG Retrieval:       {timing['rag_retrieval']:.2f}ms")
            print(f"  Doc ID Extraction:   {timing['doc_extraction']:.2f}ms")
            if 'image_loading' in timing:
                print(f"  Image Loading:       {timing['image_loading']:.2f}ms")
            print(f"  LLaVA Generation:    {timing['llava_generation']:.2f}ms")
            print(f"  {'─'*40}")
            print(f"  TOTAL:               {timing['total']:.2f}ms ({timing['total']/1000:.2f}s)")
            print("="*80)
        
        return {
            'query': query,
            'answer': answer,
            'rag_result': rag_result,
            'doc_id': doc_id,
            'image_path': image_path,
            'used_image': image_path is not None,
            'timing': timing
        }


def interactive_mode():
    """
    Interactive mode with example questions and user input
    """
    print("\n" + "="*80)
    print("🎉 Welcome to Multimodal RAG with LLaVA!")
    print("="*80)
    
    # Initialize answer generator
    print("\n🔧 Initializing system...")
    print("📦 Using llava-phi3 model")
    
    try:
        generator = LVLMAnswerGenerator()
    except Exception as e:
        print(f"❌ Error initializing generator: {e}")
        return
    
    # Example questions
    print("\n" + "="*80)
    print("💡 Example Questions You Can Ask:")
    print("="*80)
    print("\n📊 Knowledge Graph Queries (structured metadata):")
    print("  • What is the subject of document 280?")
    print("  • Who sent the document dated June 11, 1990?")
    print("  • What is the date in document 279?")
    print("  • Who was CCd on document 280?")
    
    print("\n🔍 Vector Database Queries (semantic content):")
    print("  • Find documents about flavor development")
    print("  • What documents discuss tobacco products?")
    print("  • Find letters from 1990")
    print("  • Show me handwritten forms from 1993")
    
    print("\n🎯 You can also ask follow-up or complex questions!")
    print("="*80)
    
    # Interactive loop
    while True:
        print("\n" + "─"*80)
        query = input("\n❓ Your Question (or 'quit' to exit): ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Thank you for using Multimodal RAG! Goodbye!")
            break
        
        if not query:
            print("⚠️  Please enter a question.")
            continue
        
        try:
            # Generate answer
            result = generator.answer_question(query, verbose=True)
            
            # Offer to continue
            print("\n" + "─"*80)
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user.")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


def demo_mode():
    """
    Demo mode with predefined questions
    """
    print("\n" + "="*80)
    print("🎬 Demo Mode: Multimodal RAG with LLaVA")
    print("="*80)
    print("📦 Using llava-phi3 model")
    
    # Initialize answer generator
    generator = LVLMAnswerGenerator()
    
    # Demo questions
    demo_questions = [
        "What is the subject of document 280?",
        "Find documents about flavor development",
        "Who sent the document dated June 11, 1990?",
    ]
    
    for i, question in enumerate(demo_questions, 1):
        print(f"\n{'='*80}")
        print(f"Demo Question {i}/{len(demo_questions)}")
        print(f"{'='*80}")
        
        try:
            result = generator.answer_question(question, verbose=True)
            
            if i < len(demo_questions):
                input("\n⏸️  Press Enter to continue to next question...")
        
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    print("\n🎉 Demo complete!")


if __name__ == "__main__":
    import sys
    
    print("\n" + "="*80)
    print("🚀 Multimodal RAG Answer Generation System")
    print("="*80)
    print("\nMode Selection:")
    print("  1. Interactive Mode (ask your own questions)")
    print("  2. Demo Mode (predefined questions)")
    
    mode = input("\nSelect mode (1-2, or press Enter for interactive): ").strip()
    
    if mode == "2":
        demo_mode()
    else:
        interactive_mode()

