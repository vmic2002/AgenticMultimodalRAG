"""
This script is used to test the routing accuracy of the AgenticRAGRouter
by comparing the expected route with the actual route returned by the router.
Also measures latency for each query type.
"""

import warnings
warnings.filterwarnings("ignore")
import time

from initialize_agentic_router import agentic_rag_router


def routing_accuracy_test():
    """
    Test routing accuracy with labeled queries
    """
    
    print("\n" + "="*80)
    print("🎯 Router Accuracy Test")
    print("="*80)
    
    labeled_queries = [
        ("What is the subject of document 280?", "KG"),
        ("Who sent the document dated June 11, 1990?", "KG"),
        ("Find documents about flavor development", "VECTOR"),
        ("What is the date in document 279?", "KG"),
        ("Who was CCd on document 280?", "KG"),
        ("Find handwritten forms from 1993", "VECTOR"),
        ("What letterhead does document 280 have?", "KG"),
        ("Find letters discussing tobacco products", "VECTOR"),
    ]
    
    correct = 0
    total = len(labeled_queries)
    
    # Track latencies for each route type
    kg_latencies = []
    vector_latencies = []
    
    print(f"\nTesting {total} labeled queries...\n")
    
    for query, expected_route in labeled_queries:
        # Measure query latency
        start_time = time.time()
        result = agentic_rag_router.route_query(query)
        end_time = time.time()
        
        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        
        actual_route = result['route'].split()[0]  # Get "KG" or "VECTOR" (remove "default" suffix if any)
        
        # Track latency by route type
        if actual_route == "KG":
            kg_latencies.append(latency)
        elif actual_route == "VECTOR":
            vector_latencies.append(latency)
        
        is_correct = actual_route == expected_route
        correct += is_correct
        
        status = "✅" if is_correct else "❌"
        print(f"{status} Query: {query}")
        print(f"   Expected: {expected_route} | Actual: {actual_route}")
        print(f"   Latency: {latency:.2f}ms")
        if not is_correct:
            print(f"   ⚠️  MISMATCH!")
        print()
    
    # Calculate statistics
    accuracy = (correct / total) * 100
    avg_kg_latency = sum(kg_latencies) / len(kg_latencies) if kg_latencies else 0
    avg_vector_latency = sum(vector_latencies) / len(vector_latencies) if vector_latencies else 0
    
    print("="*80)
    print(f"📊 Routing Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    print("="*80)
    
    print("\n" + "="*80)
    print("⏱️  Latency Statistics")
    print("="*80)
    print(f"\n📊 Knowledge Graph (KG) Queries:")
    print(f"   Count: {len(kg_latencies)}")
    if kg_latencies:
        print(f"   Average Latency: {avg_kg_latency:.2f}ms")
        print(f"   Min Latency: {min(kg_latencies):.2f}ms")
        print(f"   Max Latency: {max(kg_latencies):.2f}ms")
    
    print(f"\n🔍 Vector Database Queries:")
    print(f"   Count: {len(vector_latencies)}")
    if vector_latencies:
        print(f"   Average Latency: {avg_vector_latency:.2f}ms")
        print(f"   Min Latency: {min(vector_latencies):.2f}ms")
        print(f"   Max Latency: {max(vector_latencies):.2f}ms")
    
    # Overall statistics
    all_latencies = kg_latencies + vector_latencies
    if all_latencies:
        print(f"\n📈 Overall Statistics:")
        print(f"   Total Queries: {len(all_latencies)}")
        print(f"   Average Latency: {sum(all_latencies)/len(all_latencies):.2f}ms")
        print(f"   Min Latency: {min(all_latencies):.2f}ms")
        print(f"   Max Latency: {max(all_latencies):.2f}ms")
    
    # Performance comparison
    if kg_latencies and vector_latencies:
        print(f"\n⚡ Performance Comparison:")
        if avg_kg_latency < avg_vector_latency:
            speedup = avg_vector_latency / avg_kg_latency
            print(f"   KG is {speedup:.2f}x faster than Vector DB")
        else:
            speedup = avg_kg_latency / avg_vector_latency
            print(f"   Vector DB is {speedup:.2f}x faster than KG")
    
    print("="*80)

if __name__ == "__main__":
    routing_accuracy_test()
