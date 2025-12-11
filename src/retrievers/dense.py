"""Dense retriever implementation using sentence transformers and FAISS."""

import json
from pathlib import Path
from typing import List, Optional, Literal

import numpy as np

from .base import BaseRetriever, RetrievedDocument


class DenseRetriever(BaseRetriever):
    """Dense retriever using sentence embeddings and FAISS/numpy for similarity search."""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        top_k: int = 5,
        device: str = "cuda",
        use_faiss: bool = True,
        similarity_metric: Literal["cosine", "l2", "ip"] = "cosine"
    ):
        """Initialize dense retriever.
        
        Args:
            model_name: HuggingFace model name for embeddings.
            top_k: Number of documents to retrieve.
            device: Device to run the model on ('cuda' or 'cpu').
            use_faiss: Whether to use FAISS for efficient similarity search.
            similarity_metric: Similarity metric ('cosine', 'l2', 'ip').
        """
        super().__init__(top_k)
        self.model_name = model_name
        self.device = device
        self.use_faiss = use_faiss
        self.similarity_metric = similarity_metric
        
        # Lazy loading
        self._model = None
        self._index = None
        self.embeddings: Optional[np.ndarray] = None
        self.documents: List[str] = []
        self.doc_ids: List[str] = []
    
    @property
    def model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
            print(f"Loaded embedding model: {self.model_name}")
        return self._model
    
    def _create_faiss_index(self, embeddings: np.ndarray):
        """Create FAISS index from embeddings."""
        import faiss
        
        dim = embeddings.shape[1]
        
        if self.similarity_metric == "cosine":
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            index = faiss.IndexFlatIP(dim)  # Inner product after normalization = cosine
        elif self.similarity_metric == "l2":
            index = faiss.IndexFlatL2(dim)
        else:  # ip (inner product)
            index = faiss.IndexFlatIP(dim)
        
        index.add(embeddings)
        return index
    
    def index(self, documents: List[str], doc_ids: Optional[List[str]] = None) -> None:
        """Index documents for dense retrieval.
        
        Args:
            documents: List of document texts to index.
            doc_ids: Optional list of document IDs.
        """
        self.documents = documents
        self.doc_ids = doc_ids or [str(i) for i in range(len(documents))]
        
        # Generate embeddings
        print(f"Generating embeddings for {len(documents)} documents...")
        self.embeddings = self.model.encode(
            documents,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=(self.similarity_metric == "cosine")
        )
        
        # Create index
        if self.use_faiss:
            self._index = self._create_faiss_index(self.embeddings.copy())
            print(f"Created FAISS index with {len(documents)} vectors")
        
        self._is_indexed = True
        print(f"Indexed {len(documents)} documents with dense retriever")
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedDocument]:
        """Retrieve relevant documents using dense embeddings.
        
        Args:
            query: The query string.
            top_k: Number of documents to retrieve.
            
        Returns:
            List of RetrievedDocument objects sorted by similarity.
        """
        if not self._is_indexed:
            raise ValueError("No documents indexed. Call index() first.")
        
        k = top_k or self.top_k
        k = min(k, len(self.documents))
        
        # Generate query embedding
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=(self.similarity_metric == "cosine")
        )
        
        if self.use_faiss and self._index is not None:
            # Use FAISS for search
            scores, indices = self._index.search(query_embedding, k)
            scores = scores[0]
            indices = indices[0]
        else:
            # Use numpy for search
            if self.similarity_metric == "cosine":
                scores = np.dot(self.embeddings, query_embedding.T).flatten()
            elif self.similarity_metric == "l2":
                scores = -np.linalg.norm(self.embeddings - query_embedding, axis=1)
            else:  # ip
                scores = np.dot(self.embeddings, query_embedding.T).flatten()
            
            indices = np.argsort(scores)[::-1][:k]
            scores = scores[indices]
        
        # Build results
        results = []
        for idx, score in zip(indices, scores):
            if idx >= 0:  # FAISS may return -1 for empty results
                results.append(RetrievedDocument(
                    doc_id=self.doc_ids[idx],
                    content=self.documents[idx],
                    score=float(score),
                    metadata={"retriever": "dense", "model": self.model_name, "index": int(idx)}
                ))
        
        return results
    
    def save_index(self, path: str) -> None:
        """Save dense index to disk.
        
        Args:
            path: Directory path to save the index.
        """
        if not self._is_indexed:
            raise ValueError("No index to save. Call index() first.")
        
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        np.save(save_dir / "embeddings.npy", self.embeddings)
        
        # Save FAISS index if used
        if self.use_faiss and self._index is not None:
            import faiss
            faiss.write_index(self._index, str(save_dir / "faiss.index"))
        
        # Save metadata
        metadata = {
            "documents": self.documents,
            "doc_ids": self.doc_ids,
            "model_name": self.model_name,
            "top_k": self.top_k,
            "similarity_metric": self.similarity_metric,
            "use_faiss": self.use_faiss
        }
        with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"Saved dense index to {path}")
    
    def load_index(self, path: str) -> None:
        """Load dense index from disk.
        
        Args:
            path: Directory path to load the index from.
        """
        load_dir = Path(path)
        
        # Load metadata
        with open(load_dir / "metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        self.documents = metadata["documents"]
        self.doc_ids = metadata["doc_ids"]
        self.model_name = metadata["model_name"]
        self.top_k = metadata["top_k"]
        self.similarity_metric = metadata["similarity_metric"]
        self.use_faiss = metadata["use_faiss"]
        
        # Load embeddings
        self.embeddings = np.load(load_dir / "embeddings.npy")
        
        # Load FAISS index if used
        if self.use_faiss and (load_dir / "faiss.index").exists():
            import faiss
            self._index = faiss.read_index(str(load_dir / "faiss.index"))
        
        self._is_indexed = True
        print(f"Loaded dense index from {path} ({len(self.documents)} documents)")


