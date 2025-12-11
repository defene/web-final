"""Hybrid retriever combining sparse and dense retrieval."""

from typing import List, Optional, Literal

from .base import BaseRetriever, RetrievedDocument
from .sparse import BM25Retriever
from .dense import DenseRetriever


class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining BM25 and dense retrieval with rank fusion."""
    
    def __init__(
        self,
        sparse_retriever: Optional[BM25Retriever] = None,
        dense_retriever: Optional[DenseRetriever] = None,
        top_k: int = 5,
        fusion_method: Literal["rrf", "weighted"] = "rrf",
        rrf_k: int = 60,
        sparse_weight: float = 0.3,
        dense_weight: float = 0.7
    ):
        """Initialize hybrid retriever.
        
        Args:
            sparse_retriever: BM25 retriever instance.
            dense_retriever: Dense retriever instance.
            top_k: Number of documents to retrieve.
            fusion_method: Fusion method ('rrf' or 'weighted').
            rrf_k: RRF parameter k (only used with rrf method).
            sparse_weight: Weight for sparse retriever scores (only used with weighted method).
            dense_weight: Weight for dense retriever scores (only used with weighted method).
        """
        super().__init__(top_k)
        self.sparse_retriever = sparse_retriever
        self.dense_retriever = dense_retriever
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k
        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight
    
    def index(self, documents: List[str], doc_ids: Optional[List[str]] = None) -> None:
        """Index documents for hybrid retrieval.
        
        Args:
            documents: List of document texts to index.
            doc_ids: Optional list of document IDs.
        """
        if self.sparse_retriever is None:
            self.sparse_retriever = BM25Retriever(top_k=self.top_k * 2)
        if self.dense_retriever is None:
            self.dense_retriever = DenseRetriever(top_k=self.top_k * 2)
        
        # Index both retrievers
        print("Indexing sparse retriever...")
        self.sparse_retriever.index(documents, doc_ids)
        
        print("Indexing dense retriever...")
        self.dense_retriever.index(documents, doc_ids)
        
        self._is_indexed = True
        print(f"Indexed {len(documents)} documents with hybrid retriever")
    
    def _reciprocal_rank_fusion(
        self,
        sparse_results: List[RetrievedDocument],
        dense_results: List[RetrievedDocument],
        k: int
    ) -> List[RetrievedDocument]:
        """Combine results using Reciprocal Rank Fusion.
        
        RRF score = sum(1 / (k + rank)) for each list where the document appears.
        
        Args:
            sparse_results: Results from sparse retriever.
            dense_results: Results from dense retriever.
            k: Number of results to return.
            
        Returns:
            Fused list of RetrievedDocument objects.
        """
        # Build doc_id to document mapping
        doc_map = {}
        for doc in sparse_results + dense_results:
            if doc.doc_id not in doc_map:
                doc_map[doc.doc_id] = doc
        
        # Calculate RRF scores
        rrf_scores = {}
        
        for rank, doc in enumerate(sparse_results):
            rrf_scores[doc.doc_id] = rrf_scores.get(doc.doc_id, 0) + 1 / (self.rrf_k + rank + 1)
        
        for rank, doc in enumerate(dense_results):
            rrf_scores[doc.doc_id] = rrf_scores.get(doc.doc_id, 0) + 1 / (self.rrf_k + rank + 1)
        
        # Sort by RRF score
        sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:k]
        
        # Build results
        results = []
        for doc_id in sorted_doc_ids:
            original_doc = doc_map[doc_id]
            results.append(RetrievedDocument(
                doc_id=doc_id,
                content=original_doc.content,
                score=rrf_scores[doc_id],
                metadata={
                    "retriever": "hybrid",
                    "fusion_method": "rrf",
                    **original_doc.metadata
                }
            ))
        
        return results
    
    def _weighted_fusion(
        self,
        sparse_results: List[RetrievedDocument],
        dense_results: List[RetrievedDocument],
        k: int
    ) -> List[RetrievedDocument]:
        """Combine results using weighted score fusion.
        
        Args:
            sparse_results: Results from sparse retriever.
            dense_results: Results from dense retriever.
            k: Number of results to return.
            
        Returns:
            Fused list of RetrievedDocument objects.
        """
        # Normalize scores to [0, 1] range
        def normalize_scores(results: List[RetrievedDocument]) -> dict:
            if not results:
                return {}
            scores = [doc.score for doc in results]
            min_score, max_score = min(scores), max(scores)
            score_range = max_score - min_score if max_score != min_score else 1
            return {
                doc.doc_id: (doc.score - min_score) / score_range
                for doc in results
            }
        
        sparse_normalized = normalize_scores(sparse_results)
        dense_normalized = normalize_scores(dense_results)
        
        # Build doc_id to document mapping
        doc_map = {}
        for doc in sparse_results + dense_results:
            if doc.doc_id not in doc_map:
                doc_map[doc.doc_id] = doc
        
        # Calculate weighted scores
        weighted_scores = {}
        all_doc_ids = set(sparse_normalized.keys()) | set(dense_normalized.keys())
        
        for doc_id in all_doc_ids:
            sparse_score = sparse_normalized.get(doc_id, 0)
            dense_score = dense_normalized.get(doc_id, 0)
            weighted_scores[doc_id] = (
                self.sparse_weight * sparse_score +
                self.dense_weight * dense_score
            )
        
        # Sort by weighted score
        sorted_doc_ids = sorted(weighted_scores.keys(), key=lambda x: weighted_scores[x], reverse=True)[:k]
        
        # Build results
        results = []
        for doc_id in sorted_doc_ids:
            original_doc = doc_map[doc_id]
            results.append(RetrievedDocument(
                doc_id=doc_id,
                content=original_doc.content,
                score=weighted_scores[doc_id],
                metadata={
                    "retriever": "hybrid",
                    "fusion_method": "weighted",
                    **original_doc.metadata
                }
            ))
        
        return results
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedDocument]:
        """Retrieve relevant documents using hybrid retrieval.
        
        Args:
            query: The query string.
            top_k: Number of documents to retrieve.
            
        Returns:
            List of RetrievedDocument objects sorted by fused relevance.
        """
        if not self._is_indexed:
            raise ValueError("No documents indexed. Call index() first.")
        
        k = top_k or self.top_k
        
        # Retrieve from both retrievers (get more results for better fusion)
        retrieve_k = k * 3
        sparse_results = self.sparse_retriever.retrieve(query, retrieve_k)
        dense_results = self.dense_retriever.retrieve(query, retrieve_k)
        
        # Fuse results
        if self.fusion_method == "rrf":
            return self._reciprocal_rank_fusion(sparse_results, dense_results, k)
        else:
            return self._weighted_fusion(sparse_results, dense_results, k)
    
    def save_index(self, path: str) -> None:
        """Save hybrid index to disk.
        
        Args:
            path: Directory path to save the index.
        """
        from pathlib import Path
        import json
        
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save both retriever indices
        self.sparse_retriever.save_index(str(save_dir / "sparse"))
        self.dense_retriever.save_index(str(save_dir / "dense"))
        
        # Save hybrid config
        config = {
            "fusion_method": self.fusion_method,
            "rrf_k": self.rrf_k,
            "sparse_weight": self.sparse_weight,
            "dense_weight": self.dense_weight,
            "top_k": self.top_k
        }
        with open(save_dir / "hybrid_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"Saved hybrid index to {path}")
    
    def load_index(self, path: str) -> None:
        """Load hybrid index from disk.
        
        Args:
            path: Directory path to load the index from.
        """
        from pathlib import Path
        import json
        
        load_dir = Path(path)
        
        # Load hybrid config
        with open(load_dir / "hybrid_config.json", "r") as f:
            config = json.load(f)
        
        self.fusion_method = config["fusion_method"]
        self.rrf_k = config["rrf_k"]
        self.sparse_weight = config["sparse_weight"]
        self.dense_weight = config["dense_weight"]
        self.top_k = config["top_k"]
        
        # Initialize and load both retrievers
        if self.sparse_retriever is None:
            self.sparse_retriever = BM25Retriever()
        if self.dense_retriever is None:
            self.dense_retriever = DenseRetriever()
        
        self.sparse_retriever.load_index(str(load_dir / "sparse"))
        self.dense_retriever.load_index(str(load_dir / "dense"))
        
        self._is_indexed = True
        print(f"Loaded hybrid index from {path}")


