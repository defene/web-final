# RAG System for Enhanced Groundedness

A modular Retrieval-Augmented Generation (RAG) system for evaluating and improving factual accuracy in QA tasks.

## Project Structure

```
web-final/
├── config/
│   └── config.yaml          # Configuration file
├── data/
│   ├── raw/                  # Raw datasets
│   ├── processed/            # Processed data
│   └── indices/              # Vector indices
├── src/
│   ├── retrievers/           # Retrieval modules
│   │   ├── base.py           # Base retriever interface
│   │   ├── sparse.py         # BM25 retriever
│   │   ├── dense.py          # Dense (embedding) retriever
│   │   └── hybrid.py         # Hybrid retriever (RRF fusion)
│   ├── generators/           # Generation modules
│   │   ├── base.py           # Base generator interface
│   │   └── llm_generator.py  # LLM generator (OpenAI + HuggingFace)
│   ├── rerankers/            # Reranking modules
│   │   └── cross_encoder.py  # Cross-encoder reranker
│   ├── pipelines/            # RAG pipelines
│   │   └── base_rag.py       # Baseline, Reranked, Iterative RAG
│   ├── evaluation/           # Evaluation modules
│   │   ├── retrieval_metrics.py  # MRR, Precision@k, Recall@k
│   │   └── rag_metrics.py        # Ragas metrics (Faithfulness, etc.)
│   └── utils/                # Utilities
│       ├── data_loader.py    # Dataset loaders
│       └── indexer.py        # Index builders
├── scripts/
│   ├── prepare_data.py       # Data preparation
│   └── run_experiment.py     # Run experiments
├── notebooks/
│   └── demo.ipynb            # Demo notebook
├── results/                  # Experiment results
├── requirements.txt          # Dependencies
└── README.md
```

## Installation

```bash
# Clone the repository
cd web-final

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set environment variables
# Create .env file with:
# OPENAI_API_KEY=your_key_here
# HF_TOKEN=your_huggingface_token (optional, for gated models)
```

## Quick Start

### 1. Basic Usage

```python
from src.retrievers.sparse import BM25Retriever
from src.retrievers.dense import DenseRetriever
from src.generators.llm_generator import LLMGenerator
from src.pipelines.base_rag import BaseRAGPipeline

# Create documents
documents = [
    "The Eiffel Tower is located in Paris, France.",
    "The Great Wall of China is in China.",
    "The Colosseum is in Rome, Italy."
]

# Create retriever and index documents
retriever = DenseRetriever(model_name="BAAI/bge-base-en-v1.5", device="cuda")
retriever.index(documents)

# Create generator
generator = LLMGenerator(model_name="gpt-4o-mini", backend="openai")

# Create RAG pipeline
rag = BaseRAGPipeline(retriever=retriever, generator=generator, top_k=3)

# Query
result = rag.query("Where is the Eiffel Tower?")
print(result.answer)
```

### 2. Prepare Data

```bash
# Prepare SQuAD dataset (small, good for testing)
python scripts/prepare_data.py --dataset squad --max_examples 1000

# Prepare HotpotQA (multi-hop reasoning)
python scripts/prepare_data.py --dataset hotpotqa --max_examples 500

# Prepare all datasets
python scripts/prepare_data.py --dataset all --max_examples 500
```

### 3. Run Experiments

```bash
# Baseline RAG with dense retriever
python scripts/run_experiment.py --dataset squad --pipeline baseline --retriever dense

# RAG with reranking
python scripts/run_experiment.py --dataset squad --pipeline rerank --retriever dense

# Iterative RAG
python scripts/run_experiment.py --dataset hotpotqa --pipeline iterative --retriever hybrid

# Use local model
python scripts/run_experiment.py --dataset squad --pipeline baseline --generator_backend huggingface --generator_model google/flan-t5-large
```

## Supported Components

### Retrievers
- **BM25Retriever**: Sparse retrieval using BM25 algorithm
- **DenseRetriever**: Dense retrieval using sentence embeddings (BGE, Contriever, etc.)
- **HybridRetriever**: Combines sparse and dense with Reciprocal Rank Fusion (RRF)

### Generators
- **OpenAI API**: GPT-4o-mini, GPT-4o, etc.
- **HuggingFace**: FLAN-T5, Llama-3, Mistral-7B (local inference)

### Pipelines
- **BaseRAGPipeline**: Standard retrieve-then-generate
- **RerankedRAGPipeline**: Adds cross-encoder reranking step
- **IterativeRAGPipeline**: Multi-step retrieval with query refinement

### Evaluation Metrics
- **Retrieval**: MRR, Precision@k, Recall@k, NDCG@k
- **QA**: Exact Match, F1 Score
- **RAG**: Faithfulness, Answer Relevancy, Context Precision (via Ragas)

## Configuration

Edit `config/config.yaml` to customize:
- Model selection (OpenAI vs. local)
- Retrieval parameters (top_k, chunk_size)
- Pipeline settings
- Evaluation metrics

## Datasets

Supported datasets:
- **SQuAD**: Reading comprehension (single-hop)
- **HotpotQA**: Multi-hop reasoning
- **Natural Questions**: Real Google queries

## Authors

- Suwen Wang (sw6359)
- Yiheng Chen (yc7766)

