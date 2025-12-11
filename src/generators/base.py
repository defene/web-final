"""Base generator interface for RAG system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class GenerationResult:
    """Result from a generation."""
    
    answer: str
    model: str
    prompt: str
    context_used: List[str]
    metadata: Optional[dict] = None
    
    def __repr__(self) -> str:
        answer_preview = self.answer[:100] + "..." if len(self.answer) > 100 else self.answer
        return f"GenerationResult(model={self.model}, answer='{answer_preview}')"


class BaseGenerator(ABC):
    """Abstract base class for all generators."""
    
    def __init__(self, model_name: str):
        """Initialize generator.
        
        Args:
            model_name: Name of the model to use.
        """
        self.model_name = model_name
    
    @abstractmethod
    def generate(
        self,
        query: str,
        context: List[str],
        max_tokens: int = 512,
        temperature: float = 0.0
    ) -> GenerationResult:
        """Generate an answer based on query and context.
        
        Args:
            query: The user's question.
            context: List of relevant context passages.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature.
            
        Returns:
            GenerationResult object containing the answer.
        """
        pass
    
    def format_context(self, context: List[str], separator: str = "\n\n") -> str:
        """Format context passages into a single string.
        
        Args:
            context: List of context passages.
            separator: Separator between passages.
            
        Returns:
            Formatted context string.
        """
        formatted = []
        for i, passage in enumerate(context, 1):
            formatted.append(f"[{i}] {passage}")
        return separator.join(formatted)
    
    def create_prompt(
        self,
        query: str,
        context: List[str],
        template: Optional[str] = None
    ) -> str:
        """Create a prompt from query and context.
        
        Args:
            query: The user's question.
            context: List of relevant context passages.
            template: Optional custom prompt template.
            
        Returns:
            Formatted prompt string.
        """
        if template is None:
            template = """Answer the question based on the following context. If the answer cannot be found in the context, say "I don't know."

Context:
{context}

Question: {question}

Answer:"""
        
        formatted_context = self.format_context(context)
        return template.format(context=formatted_context, question=query)


