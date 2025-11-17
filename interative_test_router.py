"""
This script is used to test different queries to the AgenticRAGRouter
and see if the router is able to route the query to the appropriate RAG
and return the appropriate response.
"""

import warnings
warnings.filterwarnings("ignore")

from initialize_agentic_router import agentic_rag_router

def interactive_mode():
    """Interactive mode for testing custom queries"""
    
    print("\n" + "="*80)
    print("🎮 Interactive Mode")
    print("="*80)
    print("\nEnter your queries below (type 'quit' or 'exit' to stop):\n")
    
    while True:
        try:
            query = input("Query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
                
            if not query:
                continue
                
            print(f"\n{'─'*80}")
            result = agentic_rag_router.route_query(query)
            
            print(f"\n✅ Routing Result:")
            print(f"   Route: {result['route']}")
            
            if result['route'] == 'KG':
                print(f"\n   Generated Cypher Query:")
                print(f"   {result.get('cypher_query', 'N/A')}")
                print(f"\n   KG Response:")
                print(f"   {result['response']}")
            elif 'VECTOR' in result['route']:
                print(f"\n   Vector DB Response:")
                response_str = str(result['response'])
                if len(response_str) > 500:
                    print(f"   {response_str[:500]}...")
                else:
                    print(f"   {response_str}")
            else:
                print(f"   Error: {result.get('error', 'Unknown error')}")
                
            print(f"{'─'*80}\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")

if __name__ == "__main__":
    interactive_mode()