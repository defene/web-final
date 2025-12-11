"""Data preparation script for RAG system.

This script downloads and processes QA datasets for RAG evaluation.

Usage:
    python scripts/prepare_data.py --dataset squad --max_examples 1000
    python scripts/prepare_data.py --dataset hotpotqa --split validation
    python scripts/prepare_data.py --dataset all --max_examples 500
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_loader import DataLoader
from src.utils.indexer import IndexBuilder, chunk_documents


def prepare_squad(
    loader: DataLoader,
    builder: IndexBuilder,
    split: str = "validation",
    max_examples: int = None,
    chunk_size: int = 512,
    device: str = "cuda"
):
    """Prepare SQuAD dataset."""
    print("\n" + "=" * 60)
    print("Preparing SQuAD Dataset")
    print("=" * 60)
    
    # Load dataset
    qa_examples, documents = loader.load_squad(split=split, max_examples=max_examples)
    
    # Chunk documents
    chunked_docs = chunk_documents(documents, chunk_size=chunk_size)
    
    # Save processed data
    loader.save_processed(qa_examples, chunked_docs, "squad")
    
    # Build indices
    print("\nBuilding BM25 index...")
    builder.build_sparse_index(chunked_docs, "squad")
    
    print("\nBuilding Dense index...")
    builder.build_dense_index(chunked_docs, "squad", device=device)
    
    print("\nSQuAD preparation complete!")
    return qa_examples, chunked_docs


def prepare_hotpotqa(
    loader: DataLoader,
    builder: IndexBuilder,
    split: str = "validation",
    max_examples: int = None,
    chunk_size: int = 512,
    device: str = "cuda"
):
    """Prepare HotpotQA dataset."""
    print("\n" + "=" * 60)
    print("Preparing HotpotQA Dataset")
    print("=" * 60)
    
    # Load dataset
    qa_examples, documents = loader.load_hotpotqa(split=split, max_examples=max_examples)
    
    # Chunk documents
    chunked_docs = chunk_documents(documents, chunk_size=chunk_size)
    
    # Save processed data
    loader.save_processed(qa_examples, chunked_docs, "hotpotqa")
    
    # Build indices
    print("\nBuilding BM25 index...")
    builder.build_sparse_index(chunked_docs, "hotpotqa")
    
    print("\nBuilding Dense index...")
    builder.build_dense_index(chunked_docs, "hotpotqa", device=device)
    
    print("\nHotpotQA preparation complete!")
    return qa_examples, chunked_docs


def prepare_natural_questions(
    loader: DataLoader,
    builder: IndexBuilder,
    split: str = "train",
    max_examples: int = None,
    chunk_size: int = 512,
    device: str = "cuda"
):
    """Prepare Natural Questions dataset."""
    print("\n" + "=" * 60)
    print("Preparing Natural Questions Dataset")
    print("=" * 60)
    
    # Load dataset
    qa_examples, documents = loader.load_natural_questions(split=split, max_examples=max_examples)
    
    # Chunk documents
    chunked_docs = chunk_documents(documents, chunk_size=chunk_size)
    
    # Save processed data
    loader.save_processed(qa_examples, chunked_docs, "nq")
    
    # Build indices
    print("\nBuilding BM25 index...")
    builder.build_sparse_index(chunked_docs, "nq")
    
    print("\nBuilding Dense index...")
    builder.build_dense_index(chunked_docs, "nq", device=device)
    
    print("\nNatural Questions preparation complete!")
    return qa_examples, chunked_docs


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for RAG evaluation")
    parser.add_argument(
        "--dataset",
        type=str,
        default="squad",
        choices=["squad", "hotpotqa", "nq", "all"],
        help="Dataset to prepare"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split to use"
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Maximum number of examples to load"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=512,
        help="Chunk size for document splitting"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for embedding model (cuda/cpu)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Data directory"
    )
    
    args = parser.parse_args()
    
    # Initialize loaders
    loader = DataLoader(data_dir=args.data_dir)
    builder = IndexBuilder(index_dir=f"{args.data_dir}/indices")
    
    # Prepare datasets
    if args.dataset == "squad" or args.dataset == "all":
        prepare_squad(
            loader, builder,
            split=args.split,
            max_examples=args.max_examples,
            chunk_size=args.chunk_size,
            device=args.device
        )
    
    if args.dataset == "hotpotqa" or args.dataset == "all":
        prepare_hotpotqa(
            loader, builder,
            split=args.split,
            max_examples=args.max_examples,
            chunk_size=args.chunk_size,
            device=args.device
        )
    
    if args.dataset == "nq" or args.dataset == "all":
        # NQ uses 'train' split by default (validation is much smaller)
        nq_split = "train" if args.split == "validation" else args.split
        prepare_natural_questions(
            loader, builder,
            split=nq_split,
            max_examples=args.max_examples,
            chunk_size=args.chunk_size,
            device=args.device
        )
    
    print("\n" + "=" * 60)
    print("Data Preparation Complete!")
    print("=" * 60)
    print(f"\nAvailable indices: {builder.list_indices()}")


if __name__ == "__main__":
    main()

