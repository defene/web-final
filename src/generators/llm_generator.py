"""LLM Generator supporting both OpenAI API and local HuggingFace models."""

from typing import List, Optional, Literal

from .base import BaseGenerator, GenerationResult


class LLMGenerator(BaseGenerator):
    """LLM Generator supporting multiple backends."""
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        backend: Literal["openai", "huggingface"] = "openai",
        device: str = "cuda",
        api_key: Optional[str] = None
    ):
        """Initialize LLM Generator.
        
        Args:
            model_name: Name of the model to use.
            backend: Backend to use ('openai' or 'huggingface').
            device: Device for local models ('cuda' or 'cpu').
            api_key: OpenAI API key (optional, can use env var).
        """
        super().__init__(model_name)
        self.backend = backend
        self.device = device
        self.api_key = api_key
        
        # Lazy loading
        self._client = None
        self._model = None
        self._tokenizer = None
    
    def _init_openai(self):
        """Initialize OpenAI client."""
        if self._client is None:
            from openai import OpenAI
            import os
            
            api_key = self.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY env var or pass api_key.")
            
            self._client = OpenAI(api_key=api_key)
            print(f"Initialized OpenAI client with model: {self.model_name}")
    
    def _init_huggingface(self):
        """Initialize HuggingFace model and tokenizer."""
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Determine model type
            if "t5" in self.model_name.lower() or "flan" in self.model_name.lower():
                self._model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto"
                )
                self._model_type = "seq2seq"
            else:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto"
                )
                self._model_type = "causal"
            
            # Set pad token if not set
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            
            print(f"Loaded HuggingFace model: {self.model_name} ({self._model_type})")
    
    def _generate_openai(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """Generate using OpenAI API."""
        self._init_openai()
        
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. Be concise and accurate."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return response.choices[0].message.content.strip()
    
    def _generate_huggingface(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """Generate using local HuggingFace model."""
        self._init_huggingface()
        
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        
        # Generation parameters
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature if temperature > 0 else None,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        # Remove None values
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
        
        outputs = self._model.generate(**inputs, **gen_kwargs)
        
        if self._model_type == "seq2seq":
            # For T5/FLAN models, decode the output directly
            response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            # For causal models, remove the input prompt from output
            response = self._tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        return response.strip()
    
    def generate(
        self,
        query: str,
        context: List[str],
        max_tokens: int = 512,
        temperature: float = 0.0,
        template: Optional[str] = None
    ) -> GenerationResult:
        """Generate an answer based on query and context.
        
        Args:
            query: The user's question.
            context: List of relevant context passages.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature.
            template: Optional custom prompt template.
            
        Returns:
            GenerationResult object containing the answer.
        """
        # Create prompt
        prompt = self.create_prompt(query, context, template)
        
        # Generate based on backend
        if self.backend == "openai":
            answer = self._generate_openai(prompt, max_tokens, temperature)
        else:
            answer = self._generate_huggingface(prompt, max_tokens, temperature)
        
        return GenerationResult(
            answer=answer,
            model=self.model_name,
            prompt=prompt,
            context_used=context,
            metadata={
                "backend": self.backend,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
        )
    
    def batch_generate(
        self,
        queries: List[str],
        contexts: List[List[str]],
        max_tokens: int = 512,
        temperature: float = 0.0,
        template: Optional[str] = None
    ) -> List[GenerationResult]:
        """Generate answers for multiple queries.
        
        Args:
            queries: List of user questions.
            contexts: List of context lists (one per query).
            max_tokens: Maximum tokens per response.
            temperature: Sampling temperature.
            template: Optional custom prompt template.
            
        Returns:
            List of GenerationResult objects.
        """
        results = []
        for query, context in zip(queries, contexts):
            result = self.generate(query, context, max_tokens, temperature, template)
            results.append(result)
        return results


