"""Index building utilities for RAG system."""

from pathlib import Path
from typing import List, Optional
import json

from .data_loader import Document


class IndexBuilder:
    """Builder for creating and managing retrieval indices."""
    
    def __init__(self, index_dir: str = "data/indices"):
        """Initialize index builder.
        
        Args:
            index_dir: Directory to store indices.
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
    
    def build_sparse_index(
        self,
        documents: List[Document],
        name: str,
        k1: float = 1.5,
        b: float = 0.75
    ) -> "BM25Retriever":
        """Build and save a BM25 index.
        
        Args:
            documents: List of documents to index.
            name: Name for the index.
            k1: BM25 k1 parameter.
            b: BM25 b parameter.
            
        Returns:
            Configured BM25Retriever.
        """
        from ..retrievers.sparse import BM25Retriever
        
        retriever = BM25Retriever(k1=k1, b=b)
        
        # Index documents
        doc_texts = [doc.content for doc in documents]
        doc_ids = [doc.doc_id for doc in documents]
        retriever.index(doc_texts, doc_ids)
        
        # Save index
        save_path = self.index_dir / f"{name}_bm25"
        retriever.save_index(str(save_path))
        
        return retriever
    
    def build_dense_index(
        self,
        documents: List[Document],
        name: str,
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: str = "cuda",
        use_faiss: bool = True
    ) -> "DenseRetriever":
        """Build and save a dense embedding index.
        
        Args:
            documents: List of documents to index.
            name: Name for the index.
            model_name: Embedding model name.
            device: Device to run on.
            use_faiss: Whether to use FAISS.
            
        Returns:
            Configured DenseRetriever.
        """
        from ..retrievers.dense import DenseRetriever
        
        retriever = DenseRetriever(
            model_name=model_name,
            device=device,
            use_faiss=use_faiss
        )
        
        # Index documents
        doc_texts = [doc.content for doc in documents]
        doc_ids = [doc.doc_id for doc in documents]
        retriever.index(doc_texts, doc_ids)
        
        # Save index
        save_path = self.index_dir / f"{name}_dense"
        retriever.save_index(str(save_path))
        
        return retriever
    
    def build_hybrid_index(
        self,
        documents: List[Document],
        name: str,
        dense_model_name: str = "BAAI/bge-base-en-v1.5",
        device: str = "cuda",
        fusion_method: str = "rrf"
    ) -> "HybridRetriever":
        """Build and save a hybrid index.
        
        Args:
            documents: List of documents to index.
            name: Name for the index.
            dense_model_name: Embedding model name.
            device: Device to run on.
            fusion_method: Fusion method ('rrf' or 'weighted').
            
        Returns:
            Configured HybridRetriever.
        """
        from ..retrievers.hybrid import HybridRetriever
        
        retriever = HybridRetriever(fusion_method=fusion_method)
        
        # Index documents
        doc_texts = [doc.content for doc in documents]
        doc_ids = [doc.doc_id for doc in documents]
        retriever.index(doc_texts, doc_ids)
        
        # Save index
        save_path = self.index_dir / f"{name}_hybrid"
        retriever.save_index(str(save_path))
        
        return retriever
    
    def load_sparse_index(self, name: str) -> "BM25Retriever":
        """Load a saved BM25 index.
        
        Args:
            name: Name of the index.
            
        Returns:
            Loaded BM25Retriever.
        """
        from ..retrievers.sparse import BM25Retriever
        
        retriever = BM25Retriever()
        load_path = self.index_dir / f"{name}_bm25"
        retriever.load_index(str(load_path))
        
        return retriever
    
    def load_dense_index(
        self,
        name: str,
        device: str = "cuda"
    ) -> "DenseRetriever":
        """Load a saved dense index.
        
        Args:
            name: Name of the index.
            device: Device to run on.
            
        Returns:
            Loaded DenseRetriever.
        """
        from ..retrievers.dense import DenseRetriever
        
        retriever = DenseRetriever(device=device)
        load_path = self.index_dir / f"{name}_dense"
        retriever.load_index(str(load_path))
        
        return retriever
    
    def load_hybrid_index(
        self,
        name: str,
        device: str = "cuda"
    ) -> "HybridRetriever":
        """Load a saved hybrid index.
        
        Args:
            name: Name of the index.
            device: Device to run on.
            
        Returns:
            Loaded HybridRetriever.
        """
        from ..retrievers.hybrid import HybridRetriever
        from ..retrievers.dense import DenseRetriever
        
        retriever = HybridRetriever(
            dense_retriever=DenseRetriever(device=device)
        )
        load_path = self.index_dir / f"{name}_hybrid"
        retriever.load_index(str(load_path))
        
        return retriever
    
    def list_indices(self) -> List[str]:
        """List all available indices.
        
        Returns:
            List of index names.
        """
        indices = []
        for path in self.index_dir.iterdir():
            if path.is_dir():
                indices.append(path.name)
        return sorted(indices)


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 50
) -> List[Document]:
    """Split documents into smaller chunks.
    
    Args:
        documents: List of documents to chunk.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between chunks.
        
    Returns:
        List of chunked documents.
    """
    chunked_docs = []
    
    for doc in documents:
        text = doc.content
        
        if len(text) <= chunk_size:
            chunked_docs.append(doc)
            continue
        
        # Split into chunks
        start = 0
        chunk_idx = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence ending
                for punct in [". ", "! ", "? ", "\n"]:
                    last_punct = text[start:end].rfind(punct)
                    if last_punct > chunk_size // 2:
                        end = start + last_punct + len(punct)
                        break
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunked_docs.append(Document(
                    doc_id=f"{doc.doc_id}_chunk{chunk_idx}",
                    content=chunk_text,
                    title=doc.title,
                    metadata={
                        **(doc.metadata or {}),
                        "parent_doc_id": doc.doc_id,
                        "chunk_index": chunk_idx
                    }
                ))
                chunk_idx += 1
            
            start = end - chunk_overlap
    
    print(f"Chunked {len(documents)} documents into {len(chunked_docs)} chunks")
    return chunked_docs


