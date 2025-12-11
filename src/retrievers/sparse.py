"""BM25 sparse retriever implementation."""

import json
import pickle
from pathlib import Path
from typing import List, Optional

from rank_bm25 import BM25Okapi

from .base import BaseRetriever, RetrievedDocument


class BM25Retriever(BaseRetriever):
    """BM25-based sparse retriever using rank_bm25."""
    
    def __init__(
        self,
        top_k: int = 5,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Optional[callable] = None
    ):
        """Initialize BM25 retriever.
        
        Args:
            top_k: Number of documents to retrieve.
            k1: BM25 k1 parameter (term frequency saturation).
            b: BM25 b parameter (length normalization).
            tokenizer: Custom tokenizer function. Defaults to whitespace tokenization.
        """
        super().__init__(top_k)
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer or self._default_tokenizer
        
        self.bm25: Optional[BM25Okapi] = None
        self.documents: List[str] = []
        self.doc_ids: List[str] = []
        self.tokenized_corpus: List[List[str]] = []
    
    @staticmethod
    def _default_tokenizer(text: str) -> List[str]:
        """Default tokenizer: lowercase and split by whitespace."""
        return text.lower().split()
    
    def index(self, documents: List[str], doc_ids: Optional[List[str]] = None) -> None:
        """Index documents for BM25 retrieval.
        
        Args:
            documents: List of document texts to index.
            doc_ids: Optional list of document IDs.
        """
        self.documents = documents
        self.doc_ids = doc_ids or [str(i) for i in range(len(documents))]
        
        # Tokenize corpus
        self.tokenized_corpus = [self.tokenizer(doc) for doc in documents]
        
        # Create BM25 index
        self.bm25 = BM25Okapi(
            self.tokenized_corpus,
            k1=self.k1,
            b=self.b
        )
        
        self._is_indexed = True
        print(f"Indexed {len(documents)} documents with BM25")
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedDocument]:
        """Retrieve relevant documents using BM25.
        
        Args:
            query: The query string.
            top_k: Number of documents to retrieve.
            
        Returns:
            List of RetrievedDocument objects sorted by BM25 score.
        """
        if not self._is_indexed:
            raise ValueError("No documents indexed. Call index() first.")
        
        k = top_k or self.top_k
        
        # Tokenize query
        tokenized_query = self.tokenizer(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        
        # Build results
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include documents with positive scores
                results.append(RetrievedDocument(
                    doc_id=self.doc_ids[idx],
                    content=self.documents[idx],
                    score=float(scores[idx]),
                    metadata={"retriever": "bm25", "index": idx}
                ))
        
        return results
    
    def save_index(self, path: str) -> None:
        """Save BM25 index to disk.
        
        Args:
            path: Directory path to save the index.
        """
        if not self._is_indexed:
            raise ValueError("No index to save. Call index() first.")
        
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save BM25 object
        with open(save_dir / "bm25.pkl", "wb") as f:
            pickle.dump(self.bm25, f)
        
        # Save documents and metadata
        metadata = {
            "documents": self.documents,
            "doc_ids": self.doc_ids,
            "k1": self.k1,
            "b": self.b,
            "top_k": self.top_k
        }
        with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"Saved BM25 index to {path}")
    
    def load_index(self, path: str) -> None:
        """Load BM25 index from disk.
        
        Args:
            path: Directory path to load the index from.
        """
        load_dir = Path(path)
        
        # Load BM25 object
        with open(load_dir / "bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)
        
        # Load documents and metadata
        with open(load_dir / "metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        self.documents = metadata["documents"]
        self.doc_ids = metadata["doc_ids"]
        self.k1 = metadata["k1"]
        self.b = metadata["b"]
        self.top_k = metadata["top_k"]
        
        # Reconstruct tokenized corpus
        self.tokenized_corpus = [self.tokenizer(doc) for doc in self.documents]
        
        self._is_indexed = True
        print(f"Loaded BM25 index from {path} ({len(self.documents)} documents)")


