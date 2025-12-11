"""RAG evaluation metrics using Ragas framework."""

from typing import List, Dict, Optional
import warnings


class RAGEvaluator:
    """Evaluator for RAG-specific metrics using Ragas."""
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model_name: str = "gpt-4o-mini"
    ):
        """Initialize RAG evaluator.
        
        Args:
            openai_api_key: OpenAI API key for LLM-as-judge evaluation.
            model_name: Model to use for evaluation.
        """
        self.openai_api_key = openai_api_key
        self.model_name = model_name
        self._ragas_initialized = False
    
    def _init_ragas(self):
        """Initialize Ragas with OpenAI."""
        if self._ragas_initialized:
            return
        
        import os
        if self.openai_api_key:
            os.environ["OPENAI_API_KEY"] = self.openai_api_key
        
        self._ragas_initialized = True
    
    def evaluate(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Evaluate RAG outputs using Ragas metrics.
        
        Args:
            questions: List of questions.
            answers: List of generated answers.
            contexts: List of context lists (retrieved documents per question).
            ground_truths: Optional list of ground truth answers.
            metrics: List of metrics to compute. Options:
                     - 'faithfulness': Answer grounded in context
                     - 'answer_relevancy': Answer addresses the question
                     - 'context_precision': Retrieved context is relevant
                     - 'context_recall': Context covers ground truth
                     
        Returns:
            Dictionary of metric names to scores.
        """
        self._init_ragas()
        
        try:
            from ragas import evaluate as ragas_evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            )
            from datasets import Dataset
            
            # Default metrics
            if metrics is None:
                metrics = ["faithfulness", "answer_relevancy", "context_precision"]
            
            # Map metric names to Ragas metric objects
            metric_map = {
                "faithfulness": faithfulness,
                "answer_relevancy": answer_relevancy,
                "context_precision": context_precision,
                "context_recall": context_recall
            }
            
            ragas_metrics = [metric_map[m] for m in metrics if m in metric_map]
            
            # Prepare data for Ragas
            data = {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
            }
            
            if ground_truths and "context_recall" in metrics:
                data["ground_truth"] = ground_truths
            
            dataset = Dataset.from_dict(data)
            
            # Run evaluation
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = ragas_evaluate(dataset, metrics=ragas_metrics)
            
            return dict(results)
            
        except ImportError as e:
            print(f"Warning: Ragas not fully configured. Error: {e}")
            print("Falling back to simple heuristic evaluation.")
            return self._fallback_evaluate(questions, answers, contexts)
        except Exception as e:
            print(f"Warning: Ragas evaluation failed. Error: {e}")
            return self._fallback_evaluate(questions, answers, contexts)
    
    def _fallback_evaluate(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]]
    ) -> Dict[str, float]:
        """Simple fallback evaluation when Ragas is not available.
        
        Uses heuristic measures:
        - Context relevance: Word overlap between question and context
        - Answer coverage: Word overlap between answer and context
        """
        def word_overlap(text1: str, text2: str) -> float:
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            if not words1 or not words2:
                return 0.0
            return len(words1 & words2) / len(words1 | words2)
        
        context_relevance_scores = []
        answer_coverage_scores = []
        
        for question, answer, context_list in zip(questions, answers, contexts):
            context_text = " ".join(context_list)
            
            # Context relevance: how much the context relates to the question
            context_relevance_scores.append(word_overlap(question, context_text))
            
            # Answer coverage: how much the answer is grounded in context
            answer_coverage_scores.append(word_overlap(answer, context_text))
        
        return {
            "context_relevance_heuristic": sum(context_relevance_scores) / len(context_relevance_scores) if context_relevance_scores else 0.0,
            "answer_coverage_heuristic": sum(answer_coverage_scores) / len(answer_coverage_scores) if answer_coverage_scores else 0.0
        }
    
    def print_results(self, results: Dict[str, float]) -> None:
        """Pretty print evaluation results.
        
        Args:
            results: Dictionary of metric names to scores.
        """
        print("\n" + "=" * 50)
        print("RAG Evaluation Results")
        print("=" * 50)
        
        for metric, score in results.items():
            print(f"{metric:.<30} {score:.4f}")
        
        print("=" * 50 + "\n")


