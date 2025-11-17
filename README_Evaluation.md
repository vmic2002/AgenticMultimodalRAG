# Evaluation Pipeline

Comprehensive **end-to-end answer quality evaluation** using DeepEval with **local Ollama models** (no API costs).

> ⚠️ **Note**: DeepEval integration is currently in progress/testing. See evaluation report for details on current status and known issues.

## Quick Start

```bash
# Install DeepEval
pip install deepeval

# Quick test (3 samples, ~2 min)
python evaluation_pipeline.py --quick-test

# Full evaluation (50 samples, ~15 min)
python evaluation_pipeline.py --num-samples 50
```

## What It Evaluates

- **Answer Quality**: Exact/fuzzy match with ground truth (target: >60%)
- **Multimodal Performance**: Text+Image vs Text-only comparison
- **DeepEval Metrics**: Answer Relevancy, Faithfulness, Context Quality (uses local Ollama)
- **Latency**: Retrieval, generation, and total time tracking
- **Route Distribution**: Which queries went to KG vs Vector DB (informational)

**Note**: For routing accuracy testing with manually labeled queries, use `routing_accuracy_test.py`

## Commands

```bash
# Quick test (3 samples)
python evaluation_pipeline.py --quick-test

# Custom number of samples
python evaluation_pipeline.py --num-samples 100

# Custom output file
python evaluation_pipeline.py --num-samples 50 --output my_results.json

# Verbose mode (detailed progress)
python evaluation_pipeline.py --num-samples 10 --verbose

# Different test dataset
python evaluation_pipeline.py --test-data data/spdocvqa_qas/test_v1.0.json

# Skip DeepEval metrics (faster)
python evaluation_pipeline.py --num-samples 50 --no-deepeval
```

**JSON File:** Detailed results in `evaluation_results.json`

## DeepEval Configuration

**Default**: Uses local Ollama (llama3.1:8b) - no API costs!

To use GPT-4 instead:
```bash
export OPENAI_API_KEY="your-key"
python evaluation_pipeline.py --use-openai
```

## Common Use Cases

**Baseline before changes:**
```bash
python evaluation_pipeline.py --num-samples 100 --output baseline.json
```

**Test after changes:**
```bash
python evaluation_pipeline.py --num-samples 100 --output after_changes.json
# Compare results
```

**Analyze failures:**
Review JSON output to identify low-scoring queries and improve prompts/retrieval.