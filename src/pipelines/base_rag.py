"""Base RAG Pipeline implementation."""

from dataclasses import dataclass
from typing import List, Optional, Union
import time

from ..retrievers.base import BaseRetriever, RetrievedDocument
from ..generators.base import BaseGenerator, GenerationResult


@dataclass
class RAGResult:
    """Result from a RAG query."""
    
    query: str
    answer: str
    retrieved_documents: List[RetrievedDocument]
    generation_result: GenerationResult
    latency_ms: float
    metadata: Optional[dict] = None
    
    def __repr__(self) -> str:
        answer_preview = self.answer[:100] + "..." if len(self.answer) > 100 else self.answer
        return f"RAGResult(answer='{answer_preview}', docs={len(self.retrieved_documents)}, latency={self.latency_ms:.2f}ms)"
    
    def get_context_string(self) -> str:
        """Get concatenated context from retrieved documents."""
        return "\n\n".join(doc.content for doc in self.retrieved_documents)


class BaseRAGPipeline:
    """Base RAG pipeline combining retrieval and generation."""
    
    def __init__(
        self,
        retriever: BaseRetriever,
        generator: BaseGenerator,
        top_k: int = 5,
        prompt_template: Optional[str] = None
    ):
        """Initialize RAG pipeline.
        
        Args:
            retriever: Retriever instance for document retrieval.
            generator: Generator instance for answer generation.
            top_k: Number of documents to retrieve.
            prompt_template: Optional custom prompt template.
        """
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k
        self.prompt_template = prompt_template
    
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        max_tokens: int = 512,
        temperature: float = 0.0
    ) -> RAGResult:
        """Execute a RAG query.
        
        Args:
            question: The user's question.
            top_k: Number of documents to retrieve (overrides default).
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature.
            
        Returns:
            RAGResult object containing answer and metadata.
        """
        start_time = time.time()
        
        k = top_k or self.top_k
        
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(question, k)
        
        # Step 2: Extract context from retrieved documents
        context = [doc.content for doc in retrieved_docs]
        
        # Step 3: Generate answer
        generation_result = self.generator.generate(
            query=question,
            context=context,
            max_tokens=max_tokens,
            temperature=temperature,
            template=self.prompt_template
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        return RAGResult(
            query=question,
            answer=generation_result.answer,
            retrieved_documents=retrieved_docs,
            generation_result=generation_result,
            latency_ms=latency_ms,
            metadata={
                "pipeline": "base_rag",
                "top_k": k,
                "retriever": type(self.retriever).__name__,
                "generator": self.generator.model_name
            }
        )
    
    def batch_query(
        self,
        questions: List[str],
        top_k: Optional[int] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        show_progress: bool = True
    ) -> List[RAGResult]:
        """Execute RAG queries for multiple questions.
        
        Args:
            questions: List of questions.
            top_k: Number of documents to retrieve per question.
            max_tokens: Maximum tokens per response.
            temperature: Sampling temperature.
            show_progress: Whether to show progress bar.
            
        Returns:
            List of RAGResult objects.
        """
        results = []
        
        if show_progress:
            from tqdm import tqdm
            questions = tqdm(questions, desc="Processing queries")
        
        for question in questions:
            result = self.query(question, top_k, max_tokens, temperature)
            results.append(result)
        
        return results
    
    def index_documents(
        self,
        documents: List[str],
        doc_ids: Optional[List[str]] = None
    ) -> None:
        """Index documents for the retriever.
        
        Args:
            documents: List of document texts to index.
            doc_ids: Optional list of document IDs.
        """
        self.retriever.index(documents, doc_ids)
    
    def save(self, path: str) -> None:
        """Save the pipeline's index to disk.
        
        Args:
            path: Directory path to save to.
        """
        self.retriever.save_index(path)
    
    def load(self, path: str) -> None:
        """Load the pipeline's index from disk.
        
        Args:
            path: Directory path to load from.
        """
        self.retriever.load_index(path)


class RerankedRAGPipeline(BaseRAGPipeline):
    """RAG Pipeline with reranking step."""
    
    def __init__(
        self,
        retriever: BaseRetriever,
        generator: BaseGenerator,
        reranker,  # CrossEncoderReranker
        initial_k: int = 20,
        final_k: int = 5,
        prompt_template: Optional[str] = None
    ):
        """Initialize Reranked RAG pipeline.
        
        Args:
            retriever: Retriever instance for initial retrieval.
            generator: Generator instance for answer generation.
            reranker: Reranker instance for reranking retrieved documents.
            initial_k: Number of documents to retrieve initially.
            final_k: Number of documents after reranking.
            prompt_template: Optional custom prompt template.
        """
        super().__init__(retriever, generator, initial_k, prompt_template)
        self.reranker = reranker
        self.initial_k = initial_k
        self.final_k = final_k
    
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        max_tokens: int = 512,
        temperature: float = 0.0
    ) -> RAGResult:
        """Execute a RAG query with reranking.
        
        Args:
            question: The user's question.
            top_k: Number of documents after reranking (overrides default).
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature.
            
        Returns:
            RAGResult object containing answer and metadata.
        """
        start_time = time.time()
        
        final_k = top_k or self.final_k
        
        # Step 1: Retrieve initial set of documents
        initial_docs = self.retriever.retrieve(question, self.initial_k)
        
        # Step 2: Rerank documents
        reranked_docs = self.reranker.rerank(question, initial_docs, final_k)
        
        # Step 3: Extract context from reranked documents
        context = [doc.content for doc in reranked_docs]
        
        # Step 4: Generate answer
        generation_result = self.generator.generate(
            query=question,
            context=context,
            max_tokens=max_tokens,
            temperature=temperature,
            template=self.prompt_template
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        return RAGResult(
            query=question,
            answer=generation_result.answer,
            retrieved_documents=reranked_docs,
            generation_result=generation_result,
            latency_ms=latency_ms,
            metadata={
                "pipeline": "reranked_rag",
                "initial_k": self.initial_k,
                "final_k": final_k,
                "retriever": type(self.retriever).__name__,
                "reranker": type(self.reranker).__name__,
                "generator": self.generator.model_name
            }
        )


class IterativeRAGPipeline(BaseRAGPipeline):
    """Iterative RAG Pipeline with multi-step retrieval."""
    
    def __init__(
        self,
        retriever: BaseRetriever,
        generator: BaseGenerator,
        max_iterations: int = 2,
        top_k: int = 5,
        prompt_template: Optional[str] = None
    ):
        """Initialize Iterative RAG pipeline.
        
        Args:
            retriever: Retriever instance for document retrieval.
            generator: Generator instance for answer generation.
            max_iterations: Maximum number of retrieval iterations.
            top_k: Number of documents to retrieve per iteration.
            prompt_template: Optional custom prompt template.
        """
        super().__init__(retriever, generator, top_k, prompt_template)
        self.max_iterations = max_iterations
        
        self._refinement_template = """Based on the current context and partial answer, generate a follow-up query to find more relevant information.

Current context:
{context}

Question: {question}

Partial answer: {partial_answer}

What additional information would help answer the question more completely? Generate a focused follow-up query:"""
    
    def _generate_followup_query(
        self,
        question: str,
        context: str,
        partial_answer: str
    ) -> str:
        """Generate a follow-up query for the next retrieval iteration."""
        prompt = self._refinement_template.format(
            context=context[:2000],  # Truncate to avoid too long prompts
            question=question,
            partial_answer=partial_answer
        )
        
        result = self.generator.generate(
            query=prompt,
            context=[],  # No additional context for query generation
            max_tokens=100,
            temperature=0.3
        )
        
        return result.answer.strip()
    
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        max_tokens: int = 512,
        temperature: float = 0.0
    ) -> RAGResult:
        """Execute an iterative RAG query.
        
        Args:
            question: The user's question.
            top_k: Number of documents to retrieve per iteration.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature.
            
        Returns:
            RAGResult object containing answer and metadata.
        """
        start_time = time.time()
        
        k = top_k or self.top_k
        all_docs = []
        seen_doc_ids = set()
        current_query = question
        
        for iteration in range(self.max_iterations):
            # Retrieve documents
            retrieved_docs = self.retriever.retrieve(current_query, k)
            
            # Add new unique documents
            for doc in retrieved_docs:
                if doc.doc_id not in seen_doc_ids:
                    all_docs.append(doc)
                    seen_doc_ids.add(doc.doc_id)
            
            # Generate partial answer
            context = [doc.content for doc in all_docs]
            generation_result = self.generator.generate(
                query=question,  # Always use original question
                context=context,
                max_tokens=max_tokens,
                temperature=temperature,
                template=self.prompt_template
            )
            
            # Check if we should continue
            if iteration < self.max_iterations - 1:
                # Generate follow-up query for next iteration
                current_query = self._generate_followup_query(
                    question,
                    "\n".join(context[:3]),
                    generation_result.answer
                )
        
        latency_ms = (time.time() - start_time) * 1000
        
        return RAGResult(
            query=question,
            answer=generation_result.answer,
            retrieved_documents=all_docs,
            generation_result=generation_result,
            latency_ms=latency_ms,
            metadata={
                "pipeline": "iterative_rag",
                "iterations": self.max_iterations,
                "top_k_per_iteration": k,
                "total_docs": len(all_docs),
                "retriever": type(self.retriever).__name__,
                "generator": self.generator.model_name
            }
        )


