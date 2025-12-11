"""Base retriever interface for RAG system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RetrievedDocument:
    """A retrieved document with metadata."""
    
    doc_id: str
    content: str
    score: float
    metadata: Optional[dict] = None
    
    def __repr__(self) -> str:
        content_preview = self.content[:100] + "..." if len(self.content) > 100 else self.content
        return f"RetrievedDocument(id={self.doc_id}, score={self.score:.4f}, content='{content_preview}')"


class BaseRetriever(ABC):
    """Abstract base class for all retrievers."""
    
    def __init__(self, top_k: int = 5):
        """Initialize retriever.
        
        Args:
            top_k: Number of documents to retrieve.
        """
        self.top_k = top_k
        self._is_indexed = False
    
    @abstractmethod
    def index(self, documents: List[str], doc_ids: Optional[List[str]] = None) -> None:
        """Index documents for retrieval.
        
        Args:
            documents: List of document texts to index.
            doc_ids: Optional list of document IDs. If not provided, will use indices.
        """
        pass
    
    @abstractmethod
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedDocument]:
        """Retrieve relevant documents for a query.
        
        Args:
            query: The query string.
            top_k: Number of documents to retrieve. Uses self.top_k if not specified.
            
        Returns:
            List of RetrievedDocument objects sorted by relevance.
        """
        pass
    
    def batch_retrieve(
        self, queries: List[str], top_k: Optional[int] = None
    ) -> List[List[RetrievedDocument]]:
        """Retrieve documents for multiple queries.
        
        Args:
            queries: List of query strings.
            top_k: Number of documents to retrieve per query.
            
        Returns:
            List of lists of RetrievedDocument objects.
        """
        return [self.retrieve(query, top_k) for query in queries]
    
    @abstractmethod
    def save_index(self, path: str) -> None:
        """Save the index to disk.
        
        Args:
            path: Path to save the index.
        """
        pass
    
    @abstractmethod
    def load_index(self, path: str) -> None:
        """Load the index from disk.
        
        Args:
            path: Path to load the index from.
        """
        pass
    
    @property
    def is_indexed(self) -> bool:
        """Check if documents have been indexed."""
        return self._is_indexed


