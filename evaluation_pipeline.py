"""
Evaluation Pipeline for Multimodal RAG System using DeepEval

Focuses on end-to-end answer quality evaluation:
1. Answer quality (relevancy, faithfulness, correctness vs ground truth)
2. Retrieval quality (context relevancy, precision, recall)
3. Multimodal performance analysis (text+image vs text-only)
4. Latency tracking

Note: For routing accuracy testing, use routing_accuracy_test.py with manually labeled queries.
Note: DeepEval integration is currently in testing. See evaluation report for details.

Usage:
    python evaluation_pipeline.py --num-samples 50 --output results.json
    
    Or for quick test:
    python evaluation_pipeline.py --quick-test
"""

import warnings
warnings.filterwarnings("ignore")

import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import traceback

# Import the existing system components
from initialize_agentic_router import agentic_rag_router
from LVLM_Generation_with_Multimodal_RAG import LVLMAnswerGenerator

# DeepEval imports
try:
    from deepeval import evaluate
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        FaithfulnessMetric,
        ContextualRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
    )
    from deepeval.models import DeepEvalBaseLLM
    DEEPEVAL_AVAILABLE = True
except ImportError:
    print("⚠️  DeepEval not installed. Install with: pip install deepeval")
    print("   Continuing with basic evaluation only...")
    DEEPEVAL_AVAILABLE = False
    DeepEvalBaseLLM = None

# Ollama model for DeepEval (local evaluation)
class OllamaEvaluator(DeepEvalBaseLLM if DeepEvalBaseLLM else object):
    """Local Ollama model for DeepEval metrics - no API costs!"""
    
    def __init__(self, model_name: str = "llama3.1:8b"):
        self._model_name = model_name
        if DeepEvalBaseLLM:
            try:
                super().__init__(model=model_name)
            except:
                # Fallback if super().__init__ has different signature
                super().__init__()
        
    def load_model(self):
        return self._model_name
    
    def generate(self, prompt: str) -> str:
        """Generate response using Ollama"""
        try:
            import ollama
            response = ollama.generate(model=self._model_name, prompt=prompt)
            return response['response']
        except Exception as e:
            print(f"⚠️  Ollama generation failed: {e}")
            return ""
    
    async def a_generate(self, prompt: str) -> str:
        """Async generation (uses sync for simplicity)"""
        return self.generate(prompt)
    
    def get_model_name(self) -> str:
        return f"Ollama {self._model_name}"


@dataclass
class EvaluationResult:
    """Single evaluation result"""
    question_id: int
    question: str
    doc_id: int
    ground_truth_answers: List[str]
    
    # Retrieval
    route_used: str  # "KG" or "VECTOR" (which route was taken)
    retrieved_context: str
    retrieval_time_ms: float
    
    # Generation
    generated_answer: str
    used_image: bool
    generation_time_ms: float
    total_time_ms: float
    
    # Evaluation metrics (from DeepEval)
    answer_relevancy_score: Optional[float] = None
    faithfulness_score: Optional[float] = None
    contextual_relevancy_score: Optional[float] = None
    contextual_precision_score: Optional[float] = None
    contextual_recall_score: Optional[float] = None
    
    # Simple string matching score
    exact_match: bool = False
    fuzzy_match: bool = False
    
    # Errors
    error: Optional[str] = None


class MultimodalRAGEvaluator:
    """Comprehensive evaluator for the multimodal RAG system"""
    
    def __init__(self, 
                 test_data_path: str = "data/spdocvqa_qas/val_v1.0_withQT.json",
                 use_deepeval: bool = True,
                 use_openai: bool = False):
        """
        Initialize evaluator
        
        Args:
            test_data_path: Path to test/validation dataset
            use_deepeval: Whether to use DeepEval metrics
            use_openai: Use OpenAI/GPT-4 instead of local Ollama (requires API key)
        """
        self.test_data_path = Path(test_data_path)
        self.use_deepeval = use_deepeval and DEEPEVAL_AVAILABLE
        self.use_openai = use_openai
        
        # Initialize LVLM generator
        print("🔧 Initializing LVLM Answer Generator...")
        self.generator = LVLMAnswerGenerator()
        
        # Load test data
        print(f"📂 Loading test data from {self.test_data_path}...")
        self.test_data = self._load_test_data()
        print(f"✅ Loaded {len(self.test_data)} test questions")
        
        if len(self.test_data) == 0:
            raise ValueError("No test data loaded! Check your test data file.")
        
        # Initialize DeepEval metrics if available
        if self.use_deepeval:
            eval_model = "OpenAI GPT-4" if use_openai else "Local Ollama"
            print(f"🎯 Initializing DeepEval metrics with {eval_model}...")
            self._initialize_metrics()
        else:
            print("⚠️  DeepEval metrics disabled, using basic evaluation only")
    
    def _load_test_data(self) -> List[Dict]:
        """Load test data from JSON file"""
        if not self.test_data_path.exists():
            raise FileNotFoundError(f"Test data not found: {self.test_data_path}")
        
        with open(self.test_data_path, 'r') as f:
            data = json.load(f)
        
        # Handle both list and dict formats
        if isinstance(data, dict):
            # If it's a dict, try to find the list of questions
            # Common keys: 'data', 'questions', 'items'
            for key in ['data', 'questions', 'items']:
                if key in data:
                    return data[key]
            # If no common key found, assume the dict itself is one item
            return [data]
        elif isinstance(data, list):
            return data
        else:
            raise ValueError(f"Unexpected data format: {type(data)}")
    
    def _initialize_metrics(self):
        """Initialize DeepEval metrics with Ollama (default) or OpenAI"""
        try:
            # Configure evaluation model
            if not self.use_openai:
                # Use local Ollama - no API costs!
                eval_model = OllamaEvaluator(model_name="llama3.1:8b")
                print("   Using local Ollama (llama3.1:8b) - no API costs!")
            else:
                # Use OpenAI GPT-4 (requires OPENAI_API_KEY)
                eval_model = None  # DeepEval will use default OpenAI
                print("   Using OpenAI GPT-4 (requires OPENAI_API_KEY)")
            
            # Initialize metrics with configured model
            self.answer_relevancy_metric = AnswerRelevancyMetric(
                threshold=0.7,
                model=eval_model
            )
            self.faithfulness_metric = FaithfulnessMetric(
                threshold=0.7,
                model=eval_model
            )
            self.contextual_relevancy_metric = ContextualRelevancyMetric(
                threshold=0.7,
                model=eval_model
            )
            self.contextual_precision_metric = ContextualPrecisionMetric(
                threshold=0.7,
                model=eval_model
            )
            self.contextual_recall_metric = ContextualRecallMetric(
                threshold=0.7,
                model=eval_model
            )
            
            print("✅ DeepEval metrics initialized")
        except Exception as e:
            print(f"⚠️  Error initializing DeepEval metrics: {e}")
            print("   Continuing without DeepEval metrics...")
            self.use_deepeval = False
    
    def _check_answer_match(self, generated: str, ground_truths: List[str]) -> tuple[bool, bool]:
        """
        Check if generated answer matches ground truth
        
        Returns:
            (exact_match, fuzzy_match)
        """
        generated_lower = generated.lower().strip()
        
        # Exact match
        for gt in ground_truths:
            if gt.lower().strip() in generated_lower or generated_lower in gt.lower().strip():
                return True, True
        
        # Fuzzy match: check if key terms appear
        for gt in ground_truths:
            gt_words = set(gt.lower().split())
            gen_words = set(generated_lower.split())
            
            # If 70% of ground truth words appear in generated answer
            if len(gt_words) > 0:
                overlap = len(gt_words & gen_words) / len(gt_words)
                if overlap >= 0.7:
                    return False, True
        
        return False, False
    
    def evaluate_single_query(self, test_item: Dict, verbose: bool = False) -> EvaluationResult:
        """
        Evaluate a single query through the full pipeline
        
        Args:
            test_item: Test data item with question, answers, docId, etc.
            verbose: Print detailed progress
            
        Returns:
            EvaluationResult with all metrics
        """
        question_id = test_item.get('questionId', 0)
        question = test_item['question']
        ground_truth_answers = test_item['answers']
        doc_id = test_item['docId']
        question_types = test_item.get('question_types', [])
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"📝 Evaluating Question {question_id}: {question}")
            print(f"{'='*80}")
        
        try:
            # Run the full pipeline
            start_time = time.time()
            result = self.generator.answer_question(question, verbose=verbose)
            total_time = (time.time() - start_time) * 1000
            
            # Extract results
            route_used = result['rag_result']['route'].split()[0]  # "KG" or "VECTOR"
            retrieved_context = result['rag_result']['response']
            generated_answer = result['answer']
            used_image = result['used_image']
            
            # Check answer match
            exact_match, fuzzy_match = self._check_answer_match(generated_answer, ground_truth_answers)
            
            # Create evaluation result
            eval_result = EvaluationResult(
                question_id=question_id,
                question=question,
                doc_id=doc_id,
                ground_truth_answers=ground_truth_answers,
                route_used=route_used,
                retrieved_context=retrieved_context,
                retrieval_time_ms=result['timing'].get('rag_retrieval', 0),
                generated_answer=generated_answer,
                used_image=used_image,
                generation_time_ms=result['timing'].get('llava_generation', 0),
                total_time_ms=total_time,
                exact_match=exact_match,
                fuzzy_match=fuzzy_match
            )
            
            # Run DeepEval metrics if available
            if self.use_deepeval:
                try:
                    eval_result = self._run_deepeval_metrics(eval_result)
                except Exception as e:
                    if verbose:
                        print(f"⚠️  DeepEval metrics failed: {e}")
            
            if verbose:
                print(f"\n✅ Evaluation complete!")
                print(f"   Route Used: {route_used}")
                print(f"   Answer Match: {'✓ Exact' if exact_match else '✓ Fuzzy' if fuzzy_match else '✗ No match'}")
                print(f"   Total Time: {total_time:.2f}ms")
            
            return eval_result
        
        except Exception as e:
            if verbose:
                print(f"\n❌ Error evaluating query: {e}")
                traceback.print_exc()
            
            return EvaluationResult(
                question_id=question_id,
                question=question,
                doc_id=doc_id,
                ground_truth_answers=ground_truth_answers,
                route_used="ERROR",
                retrieved_context="",
                retrieval_time_ms=0,
                generated_answer="",
                used_image=False,
                generation_time_ms=0,
                total_time_ms=0,
                error=str(e)
            )
    
    def _run_deepeval_metrics(self, eval_result: EvaluationResult) -> EvaluationResult:
        """
        Run DeepEval metrics on the evaluation result
        
        Note: This uses local Ollama by default, or OpenAI if configured
        """
        if not self.use_deepeval:
            return eval_result
        
        # Split retrieved context into sentences for better evaluation
        # DeepEval expects a list of context strings
        context_list = [eval_result.retrieved_context] if eval_result.retrieved_context else []
        
        # Create LLM test case
        test_case = LLMTestCase(
            input=eval_result.question,
            actual_output=eval_result.generated_answer,
            expected_output=eval_result.ground_truth_answers[0],  # Use first ground truth
            retrieval_context=context_list  # For RAG-specific metrics
        )
        
        # Run metrics (silently catch errors to continue evaluation)
        try:
            self.answer_relevancy_metric.measure(test_case)
            eval_result.answer_relevancy_score = self.answer_relevancy_metric.score
        except Exception as e:
            pass  # Silently skip if metric fails
        
        try:
            self.faithfulness_metric.measure(test_case)
            eval_result.faithfulness_score = self.faithfulness_metric.score
        except Exception as e:
            pass
        
        try:
            self.contextual_relevancy_metric.measure(test_case)
            eval_result.contextual_relevancy_score = self.contextual_relevancy_metric.score
        except Exception as e:
            pass
        
        return eval_result
    
    def evaluate_dataset(self, 
                        num_samples: Optional[int] = None,
                        start_idx: int = 0,
                        verbose: bool = True) -> List[EvaluationResult]:
        """
        Evaluate multiple queries from the dataset
        
        Args:
            num_samples: Number of samples to evaluate (None = all)
            start_idx: Starting index in dataset
            verbose: Print progress
            
        Returns:
            List of evaluation results
        """
        # Select samples
        if num_samples is None:
            samples = self.test_data[start_idx:]
        else:
            samples = self.test_data[start_idx:start_idx + num_samples]
        
        print(f"\n{'='*80}")
        print(f"🚀 Starting Evaluation")
        print(f"{'='*80}")
        print(f"Total samples: {len(samples)}")
        print(f"Start index: {start_idx}")
        print(f"DeepEval enabled: {self.use_deepeval}")
        print(f"{'='*80}\n")
        
        results = []
        for i, test_item in enumerate(samples, 1):
            print(f"\n[{i}/{len(samples)}] ", end="")
            result = self.evaluate_single_query(test_item, verbose=verbose)
            results.append(result)
            
            # Progress update
            if not verbose and i % 5 == 0:
                print(f"✓ Completed {i}/{len(samples)} evaluations")
        
        return results
    
    def generate_report(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report
        
        Args:
            results: List of evaluation results
            
        Returns:
            Dictionary with aggregated metrics and statistics
        """
        if not results:
            return {"error": "No results to analyze"}
        
        # Filter out errors
        valid_results = [r for r in results if r.error is None]
        error_count = len(results) - len(valid_results)
        
        if not valid_results:
            return {"error": "All evaluations failed", "error_count": error_count}
        
        # Route distribution (for informational purposes)
        kg_results = [r for r in valid_results if r.route_used == "KG"]
        vector_results = [r for r in valid_results if r.route_used == "VECTOR"]
        
        # Answer quality
        exact_matches = sum(1 for r in valid_results if r.exact_match)
        fuzzy_matches = sum(1 for r in valid_results if r.fuzzy_match)
        exact_match_rate = exact_matches / len(valid_results)
        fuzzy_match_rate = fuzzy_matches / len(valid_results)
        
        # Multimodal usage
        multimodal_results = [r for r in valid_results if r.used_image]
        text_only_results = [r for r in valid_results if not r.used_image]
        
        multimodal_exact_match = sum(1 for r in multimodal_results if r.exact_match) / len(multimodal_results) if multimodal_results else 0
        text_only_exact_match = sum(1 for r in text_only_results if r.exact_match) / len(text_only_results) if text_only_results else 0
        
        # Timing statistics
        avg_total_time = sum(r.total_time_ms for r in valid_results) / len(valid_results)
        avg_retrieval_time = sum(r.retrieval_time_ms for r in valid_results) / len(valid_results)
        avg_generation_time = sum(r.generation_time_ms for r in valid_results) / len(valid_results)
        
        # DeepEval metrics (if available)
        deepeval_metrics = {}
        if self.use_deepeval:
            relevancy_scores = [r.answer_relevancy_score for r in valid_results if r.answer_relevancy_score is not None]
            faithfulness_scores = [r.faithfulness_score for r in valid_results if r.faithfulness_score is not None]
            contextual_relevancy_scores = [r.contextual_relevancy_score for r in valid_results if r.contextual_relevancy_score is not None]
            
            if relevancy_scores:
                deepeval_metrics['answer_relevancy'] = {
                    'mean': sum(relevancy_scores) / len(relevancy_scores),
                    'min': min(relevancy_scores),
                    'max': max(relevancy_scores),
                    'count': len(relevancy_scores)
                }
            
            if faithfulness_scores:
                deepeval_metrics['faithfulness'] = {
                    'mean': sum(faithfulness_scores) / len(faithfulness_scores),
                    'min': min(faithfulness_scores),
                    'max': max(faithfulness_scores),
                    'count': len(faithfulness_scores)
                }
            
            if contextual_relevancy_scores:
                deepeval_metrics['contextual_relevancy'] = {
                    'mean': sum(contextual_relevancy_scores) / len(contextual_relevancy_scores),
                    'min': min(contextual_relevancy_scores),
                    'max': max(contextual_relevancy_scores),
                    'count': len(contextual_relevancy_scores)
                }
        
        # Compile report
        report = {
            'summary': {
                'total_samples': len(results),
                'valid_samples': len(valid_results),
                'error_count': error_count,
                'evaluation_date': time.strftime("%Y-%m-%d %H:%M:%S")
            },
            'route_distribution': {
                'kg_queries': len(kg_results),
                'vector_queries': len(vector_results)
            },
            'answer_quality': {
                'exact_match_rate': exact_match_rate,
                'fuzzy_match_rate': fuzzy_match_rate,
                'exact_matches': exact_matches,
                'fuzzy_matches': fuzzy_matches
            },
            'multimodal_analysis': {
                'multimodal_queries': len(multimodal_results),
                'text_only_queries': len(text_only_results),
                'multimodal_exact_match_rate': multimodal_exact_match,
                'text_only_exact_match_rate': text_only_exact_match
            },
            'timing': {
                'avg_total_time_ms': avg_total_time,
                'avg_retrieval_time_ms': avg_retrieval_time,
                'avg_generation_time_ms': avg_generation_time
            },
            'deepeval_metrics': deepeval_metrics if deepeval_metrics else None
        }
        
        return report
    
    def print_report(self, report: Dict[str, Any]):
        """Print formatted evaluation report"""
        print("\n" + "="*80)
        print("📊 EVALUATION REPORT")
        print("="*80)
        
        # Summary
        print("\n📋 Summary:")
        print(f"   Total Samples: {report['summary']['total_samples']}")
        print(f"   Valid Samples: {report['summary']['valid_samples']}")
        print(f"   Errors: {report['summary']['error_count']}")
        print(f"   Date: {report['summary']['evaluation_date']}")
        
        # Route distribution
        print("\n🔀 Route Distribution:")
        print(f"   KG Queries: {report['route_distribution']['kg_queries']}")
        print(f"   Vector Queries: {report['route_distribution']['vector_queries']}")
        print(f"   Note: For routing accuracy testing, use routing_accuracy_test.py")
        
        # Answer Quality
        print("\n✅ Answer Quality:")
        print(f"   Exact Match Rate: {report['answer_quality']['exact_match_rate']*100:.1f}%")
        print(f"   Fuzzy Match Rate: {report['answer_quality']['fuzzy_match_rate']*100:.1f}%")
        print(f"   Exact Matches: {report['answer_quality']['exact_matches']}")
        print(f"   Fuzzy Matches: {report['answer_quality']['fuzzy_matches']}")
        
        # Multimodal Analysis
        print("\n🖼️  Multimodal Analysis:")
        print(f"   Multimodal Queries: {report['multimodal_analysis']['multimodal_queries']}")
        print(f"   Text-Only Queries: {report['multimodal_analysis']['text_only_queries']}")
        print(f"   Multimodal Exact Match: {report['multimodal_analysis']['multimodal_exact_match_rate']*100:.1f}%")
        print(f"   Text-Only Exact Match: {report['multimodal_analysis']['text_only_exact_match_rate']*100:.1f}%")
        
        # Timing
        print("\n⏱️  Performance:")
        print(f"   Avg Total Time: {report['timing']['avg_total_time_ms']:.2f}ms")
        print(f"   Avg Retrieval Time: {report['timing']['avg_retrieval_time_ms']:.2f}ms")
        print(f"   Avg Generation Time: {report['timing']['avg_generation_time_ms']:.2f}ms")
        
        # DeepEval Metrics
        if report['deepeval_metrics']:
            print("\n🎯 DeepEval Metrics:")
            for metric_name, metric_data in report['deepeval_metrics'].items():
                print(f"   {metric_name.replace('_', ' ').title()}:")
                print(f"      Mean: {metric_data['mean']:.3f}")
                print(f"      Range: [{metric_data['min']:.3f}, {metric_data['max']:.3f}]")
                print(f"      Evaluated: {metric_data['count']} samples")
        
        print("\n" + "="*80)
    
    def save_results(self, results: List[EvaluationResult], report: Dict[str, Any], output_path: str):
        """Save evaluation results and report to JSON file"""
        output_data = {
            'report': report,
            'results': [asdict(r) for r in results]
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_path}")


def main():
    """Main evaluation script"""
    parser = argparse.ArgumentParser(description="Evaluate Multimodal RAG System")
    parser.add_argument('--num-samples', type=int, default=10, 
                       help='Number of samples to evaluate (default: 10)')
    parser.add_argument('--start-idx', type=int, default=0,
                       help='Starting index in dataset (default: 0)')
    parser.add_argument('--test-data', type=str, default='data/spdocvqa_qas/val_v1.0_withQT.json',
                       help='Path to test data JSON file')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                       help='Output file for results (default: evaluation_results.json)')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with 3 samples')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed progress')
    parser.add_argument('--no-deepeval', action='store_true',
                       help='Disable DeepEval metrics (faster)')
    parser.add_argument('--use-openai', action='store_true',
                       help='Use OpenAI GPT-4 instead of local Ollama (requires OPENAI_API_KEY)')
    
    args = parser.parse_args()
    
    # Quick test mode
    if args.quick_test:
        args.num_samples = 3
        args.verbose = True
        print("\n🚀 Running Quick Test Mode (3 samples)")
    
    # Initialize evaluator
    try:
        evaluator = MultimodalRAGEvaluator(
            test_data_path=args.test_data,
            use_deepeval=not args.no_deepeval,
            use_openai=args.use_openai
        )
    except Exception as e:
        print(f"❌ Error initializing evaluator: {e}")
        traceback.print_exc()
        return
    
    # Run evaluation
    try:
        results = evaluator.evaluate_dataset(
            num_samples=args.num_samples,
            start_idx=args.start_idx,
            verbose=args.verbose
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
        return
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        traceback.print_exc()
        return
    
    # Generate and print report
    try:
        report = evaluator.generate_report(results)
        evaluator.print_report(report)
        
        # Save results
        evaluator.save_results(results, report, args.output)
        
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()

