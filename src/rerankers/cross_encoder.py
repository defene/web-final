"""Cross-encoder reranker implementation."""

from typing import List, Optional

from ..retrievers.base import RetrievedDocument


class CrossEncoderReranker:
    """Cross-encoder based reranker using sentence-transformers."""
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cuda"
    ):
        """Initialize cross-encoder reranker.
        
        Args:
            model_name: HuggingFace cross-encoder model name.
            device: Device to run the model on ('cuda' or 'cpu').
        """
        self.model_name = model_name
        self.device = device
        self._model = None
    
    @property
    def model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, device=self.device)
            print(f"Loaded cross-encoder model: {self.model_name}")
        return self._model
    
    def rerank(
        self,
        query: str,
        documents: List[RetrievedDocument],
        top_k: Optional[int] = None
    ) -> List[RetrievedDocument]:
        """Rerank documents using cross-encoder.
        
        Args:
            query: The query string.
            documents: List of documents to rerank.
            top_k: Number of documents to return after reranking.
            
        Returns:
            List of RetrievedDocument objects sorted by cross-encoder score.
        """
        if not documents:
            return []
        
        k = top_k or len(documents)
        k = min(k, len(documents))
        
        # Prepare query-document pairs
        pairs = [(query, doc.content) for doc in documents]
        
        # Get cross-encoder scores
        scores = self.model.predict(pairs)
        
        # Create reranked documents with new scores
        reranked = []
        for doc, score in zip(documents, scores):
            reranked.append(RetrievedDocument(
                doc_id=doc.doc_id,
                content=doc.content,
                score=float(score),
                metadata={
                    **doc.metadata,
                    "original_score": doc.score,
                    "reranker": "cross_encoder",
                    "reranker_model": self.model_name
                }
            ))
        
        # Sort by new scores and return top-k
        reranked.sort(key=lambda x: x.score, reverse=True)
        
        return reranked[:k]
    
    def batch_rerank(
        self,
        queries: List[str],
        documents_list: List[List[RetrievedDocument]],
        top_k: Optional[int] = None
    ) -> List[List[RetrievedDocument]]:
        """Rerank documents for multiple queries.
        
        Args:
            queries: List of query strings.
            documents_list: List of document lists (one per query).
            top_k: Number of documents to return per query.
            
        Returns:
            List of lists of reranked RetrievedDocument objects.
        """
        return [
            self.rerank(query, docs, top_k)
            for query, docs in zip(queries, documents_list)
        ]


