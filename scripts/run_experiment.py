"""Run RAG experiments.

This script runs RAG experiments with different configurations and evaluates results.

Usage:
    python scripts/run_experiment.py --dataset squad --pipeline baseline --retriever dense
    python scripts/run_experiment.py --dataset hotpotqa --pipeline rerank --num_examples 100
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_loader import DataLoader
from src.utils.indexer import IndexBuilder
from src.retrievers.sparse import BM25Retriever
from src.retrievers.dense import DenseRetriever
from src.retrievers.hybrid import HybridRetriever
from src.generators.llm_generator import LLMGenerator
from src.pipelines.base_rag import BaseRAGPipeline, RerankedRAGPipeline, IterativeRAGPipeline
from src.rerankers.cross_encoder import CrossEncoderReranker
from src.evaluation.retrieval_metrics import RetrievalEvaluator, QAEvaluator


def load_retriever(
    builder: IndexBuilder,
    dataset: str,
    retriever_type: str,
    device: str
):
    """Load a retriever based on type."""
    if retriever_type == "sparse":
        return builder.load_sparse_index(dataset)
    elif retriever_type == "dense":
        return builder.load_dense_index(dataset, device=device)
    elif retriever_type == "hybrid":
        return builder.load_hybrid_index(dataset, device=device)
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")


def create_pipeline(
    retriever,
    generator: LLMGenerator,
    pipeline_type: str,
    device: str,
    top_k: int = 5
):
    """Create a RAG pipeline based on type."""
    if pipeline_type == "baseline":
        return BaseRAGPipeline(
            retriever=retriever,
            generator=generator,
            top_k=top_k
        )
    elif pipeline_type == "rerank":
        reranker = CrossEncoderReranker(device=device)
        return RerankedRAGPipeline(
            retriever=retriever,
            generator=generator,
            reranker=reranker,
            initial_k=top_k * 4,
            final_k=top_k
        )
    elif pipeline_type == "iterative":
        return IterativeRAGPipeline(
            retriever=retriever,
            generator=generator,
            max_iterations=2,
            top_k=top_k
        )
    else:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")


def run_experiment(
    dataset: str,
    pipeline_type: str,
    retriever_type: str,
    generator_backend: str,
    generator_model: str,
    num_examples: int,
    top_k: int,
    device: str,
    data_dir: str,
    results_dir: str
):
    """Run a single experiment."""
    print("\n" + "=" * 60)
    print("RAG Experiment")
    print("=" * 60)
    print(f"Dataset: {dataset}")
    print(f"Pipeline: {pipeline_type}")
    print(f"Retriever: {retriever_type}")
    print(f"Generator: {generator_backend}/{generator_model}")
    print(f"Top-k: {top_k}")
    print(f"Num examples: {num_examples}")
    print("=" * 60)
    
    # Initialize components
    loader = DataLoader(data_dir=data_dir)
    builder = IndexBuilder(index_dir=f"{data_dir}/indices")
    
    # Load data
    print("\nLoading data...")
    qa_examples, _ = loader.load_processed(dataset)
    
    if num_examples:
        qa_examples = qa_examples[:num_examples]
    
    # Load retriever
    print("\nLoading retriever...")
    retriever = load_retriever(builder, dataset, retriever_type, device)
    
    # Create generator
    print("\nInitializing generator...")
    generator = LLMGenerator(
        model_name=generator_model,
        backend=generator_backend,
        device=device
    )
    
    # Create pipeline
    print("\nCreating pipeline...")
    pipeline = create_pipeline(
        retriever=retriever,
        generator=generator,
        pipeline_type=pipeline_type,
        device=device,
        top_k=top_k
    )
    
    # Run queries
    print(f"\nRunning {len(qa_examples)} queries...")
    questions = [ex.question for ex in qa_examples]
    ground_truths = [ex.answer for ex in qa_examples]
    relevant_ids = [ex.relevant_doc_ids or set() for ex in qa_examples]
    
    results = pipeline.batch_query(questions, top_k=top_k)
    
    # Extract results for evaluation
    predictions = [r.answer for r in results]
    retrieved_ids = [[doc.doc_id for doc in r.retrieved_documents] for r in results]
    contexts = [[doc.content for doc in r.retrieved_documents] for r in results]
    latencies = [r.latency_ms for r in results]
    
    # Evaluate retrieval
    print("\nEvaluating retrieval...")
    retrieval_evaluator = RetrievalEvaluator()
    retrieval_metrics = retrieval_evaluator.evaluate(
        retrieved_ids=retrieved_ids,
        relevant_ids=relevant_ids,
        k_values=[1, 3, 5, 10]
    )
    retrieval_evaluator.print_results(retrieval_metrics)
    
    # Evaluate QA
    print("\nEvaluating QA...")
    qa_evaluator = QAEvaluator()
    qa_metrics = qa_evaluator.evaluate(predictions, ground_truths)
    print(f"Exact Match: {qa_metrics['exact_match']:.4f}")
    print(f"F1 Score: {qa_metrics['f1']:.4f}")
    
    # System metrics
    avg_latency = sum(latencies) / len(latencies)
    print(f"\nAverage Latency: {avg_latency:.2f} ms")
    
    # Save results
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = Path(results_dir) / f"{dataset}_{pipeline_type}_{retriever_type}_{timestamp}.json"
    
    experiment_results = {
        "config": {
            "dataset": dataset,
            "pipeline_type": pipeline_type,
            "retriever_type": retriever_type,
            "generator_backend": generator_backend,
            "generator_model": generator_model,
            "top_k": top_k,
            "num_examples": len(qa_examples)
        },
        "retrieval_metrics": retrieval_metrics,
        "qa_metrics": qa_metrics,
        "system_metrics": {
            "avg_latency_ms": avg_latency,
            "total_queries": len(qa_examples)
        },
        "timestamp": timestamp
    }
    
    with open(result_file, "w") as f:
        json.dump(experiment_results, f, indent=2)
    
    print(f"\nResults saved to: {result_file}")
    
    return experiment_results


def main():
    parser = argparse.ArgumentParser(description="Run RAG experiments")
    parser.add_argument(
        "--dataset",
        type=str,
        default="squad",
        help="Dataset to use (squad, hotpotqa, nq)"
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default="baseline",
        choices=["baseline", "rerank", "iterative"],
        help="Pipeline type"
    )
    parser.add_argument(
        "--retriever",
        type=str,
        default="dense",
        choices=["sparse", "dense", "hybrid"],
        help="Retriever type"
    )
    parser.add_argument(
        "--generator_backend",
        type=str,
        default="openai",
        choices=["openai", "huggingface"],
        help="Generator backend"
    )
    parser.add_argument(
        "--generator_model",
        type=str,
        default="gpt-4o-mini",
        help="Generator model name"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=100,
        help="Number of examples to evaluate"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of documents to retrieve"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Data directory"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Results directory"
    )
    
    args = parser.parse_args()
    
    run_experiment(
        dataset=args.dataset,
        pipeline_type=args.pipeline,
        retriever_type=args.retriever,
        generator_backend=args.generator_backend,
        generator_model=args.generator_model,
        num_examples=args.num_examples,
        top_k=args.top_k,
        device=args.device,
        data_dir=args.data_dir,
        results_dir=args.results_dir
    )


if __name__ == "__main__":
    main()


