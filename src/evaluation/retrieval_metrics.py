"""Retrieval evaluation metrics."""

from typing import List, Dict, Optional, Set
import numpy as np


class RetrievalEvaluator:
    """Evaluator for retrieval performance metrics."""
    
    def __init__(self):
        """Initialize retrieval evaluator."""
        pass
    
    def mean_reciprocal_rank(
        self,
        retrieved_ids: List[List[str]],
        relevant_ids: List[Set[str]]
    ) -> float:
        """Calculate Mean Reciprocal Rank (MRR).
        
        Args:
            retrieved_ids: List of retrieved document ID lists (one per query).
            relevant_ids: List of sets of relevant document IDs (one per query).
            
        Returns:
            MRR score.
        """
        rr_scores = []
        
        for retrieved, relevant in zip(retrieved_ids, relevant_ids):
            rr = 0.0
            for rank, doc_id in enumerate(retrieved, 1):
                if doc_id in relevant:
                    rr = 1.0 / rank
                    break
            rr_scores.append(rr)
        
        return np.mean(rr_scores) if rr_scores else 0.0
    
    def precision_at_k(
        self,
        retrieved_ids: List[List[str]],
        relevant_ids: List[Set[str]],
        k: int
    ) -> float:
        """Calculate Precision@k.
        
        Args:
            retrieved_ids: List of retrieved document ID lists.
            relevant_ids: List of sets of relevant document IDs.
            k: Number of top documents to consider.
            
        Returns:
            Precision@k score.
        """
        precisions = []
        
        for retrieved, relevant in zip(retrieved_ids, relevant_ids):
            top_k = retrieved[:k]
            relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant)
            precisions.append(relevant_in_top_k / k if k > 0 else 0.0)
        
        return np.mean(precisions) if precisions else 0.0
    
    def recall_at_k(
        self,
        retrieved_ids: List[List[str]],
        relevant_ids: List[Set[str]],
        k: int
    ) -> float:
        """Calculate Recall@k.
        
        Args:
            retrieved_ids: List of retrieved document ID lists.
            relevant_ids: List of sets of relevant document IDs.
            k: Number of top documents to consider.
            
        Returns:
            Recall@k score.
        """
        recalls = []
        
        for retrieved, relevant in zip(retrieved_ids, relevant_ids):
            if not relevant:
                continue
            top_k = retrieved[:k]
            relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant)
            recalls.append(relevant_in_top_k / len(relevant))
        
        return np.mean(recalls) if recalls else 0.0
    
    def ndcg_at_k(
        self,
        retrieved_ids: List[List[str]],
        relevant_ids: List[Set[str]],
        k: int
    ) -> float:
        """Calculate Normalized Discounted Cumulative Gain (NDCG@k).
        
        Args:
            retrieved_ids: List of retrieved document ID lists.
            relevant_ids: List of sets of relevant document IDs.
            k: Number of top documents to consider.
            
        Returns:
            NDCG@k score.
        """
        def dcg(relevances: List[int], k: int) -> float:
            relevances = relevances[:k]
            return sum(rel / np.log2(rank + 2) for rank, rel in enumerate(relevances))
        
        ndcg_scores = []
        
        for retrieved, relevant in zip(retrieved_ids, relevant_ids):
            # Calculate relevance scores (binary: 1 if relevant, 0 otherwise)
            relevances = [1 if doc_id in relevant else 0 for doc_id in retrieved[:k]]
            
            # Calculate DCG
            dcg_score = dcg(relevances, k)
            
            # Calculate ideal DCG (all relevant documents at the top)
            ideal_relevances = sorted(relevances, reverse=True)
            idcg_score = dcg(ideal_relevances, k)
            
            if idcg_score > 0:
                ndcg_scores.append(dcg_score / idcg_score)
            else:
                ndcg_scores.append(0.0)
        
        return np.mean(ndcg_scores) if ndcg_scores else 0.0
    
    def evaluate(
        self,
        retrieved_ids: List[List[str]],
        relevant_ids: List[Set[str]],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, float]:
        """Run comprehensive retrieval evaluation.
        
        Args:
            retrieved_ids: List of retrieved document ID lists.
            relevant_ids: List of sets of relevant document IDs.
            k_values: List of k values for Precision@k and Recall@k.
            
        Returns:
            Dictionary of metric names to scores.
        """
        results = {
            "mrr": self.mean_reciprocal_rank(retrieved_ids, relevant_ids)
        }
        
        for k in k_values:
            results[f"precision@{k}"] = self.precision_at_k(retrieved_ids, relevant_ids, k)
            results[f"recall@{k}"] = self.recall_at_k(retrieved_ids, relevant_ids, k)
            results[f"ndcg@{k}"] = self.ndcg_at_k(retrieved_ids, relevant_ids, k)
        
        return results
    
    def print_results(self, results: Dict[str, float]) -> None:
        """Pretty print evaluation results.
        
        Args:
            results: Dictionary of metric names to scores.
        """
        print("\n" + "=" * 50)
        print("Retrieval Evaluation Results")
        print("=" * 50)
        
        # Group metrics
        print(f"\nMRR: {results['mrr']:.4f}")
        
        # Find all k values
        k_values = sorted(set(
            int(key.split("@")[1])
            for key in results
            if "@" in key
        ))
        
        print("\n{:<12} {:<12} {:<12} {:<12}".format("K", "Precision", "Recall", "NDCG"))
        print("-" * 48)
        for k in k_values:
            p = results.get(f"precision@{k}", 0)
            r = results.get(f"recall@{k}", 0)
            n = results.get(f"ndcg@{k}", 0)
            print(f"{k:<12} {p:<12.4f} {r:<12.4f} {n:<12.4f}")
        
        print("=" * 50 + "\n")


class QAEvaluator:
    """Evaluator for QA metrics (Exact Match and F1)."""
    
    def __init__(self):
        """Initialize QA evaluator."""
        pass
    
    @staticmethod
    def normalize_answer(text: str) -> str:
        """Normalize answer text for comparison."""
        import re
        import string
        
        # Lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))
        # Remove articles
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        # Remove extra whitespace
        text = " ".join(text.split())
        
        return text
    
    def exact_match(
        self,
        predictions: List[str],
        references: List[str]
    ) -> float:
        """Calculate Exact Match score.
        
        Args:
            predictions: List of predicted answers.
            references: List of reference answers.
            
        Returns:
            Exact Match score.
        """
        matches = 0
        for pred, ref in zip(predictions, references):
            if self.normalize_answer(pred) == self.normalize_answer(ref):
                matches += 1
        
        return matches / len(predictions) if predictions else 0.0
    
    def f1_score(
        self,
        predictions: List[str],
        references: List[str]
    ) -> float:
        """Calculate token-level F1 score.
        
        Args:
            predictions: List of predicted answers.
            references: List of reference answers.
            
        Returns:
            Average F1 score.
        """
        def compute_f1(pred: str, ref: str) -> float:
            pred_tokens = set(self.normalize_answer(pred).split())
            ref_tokens = set(self.normalize_answer(ref).split())
            
            if not pred_tokens and not ref_tokens:
                return 1.0
            if not pred_tokens or not ref_tokens:
                return 0.0
            
            common = pred_tokens & ref_tokens
            precision = len(common) / len(pred_tokens)
            recall = len(common) / len(ref_tokens)
            
            if precision + recall == 0:
                return 0.0
            
            return 2 * precision * recall / (precision + recall)
        
        f1_scores = [compute_f1(pred, ref) for pred, ref in zip(predictions, references)]
        return np.mean(f1_scores) if f1_scores else 0.0
    
    def evaluate(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """Run QA evaluation.
        
        Args:
            predictions: List of predicted answers.
            references: List of reference answers.
            
        Returns:
            Dictionary with exact_match and f1 scores.
        """
        return {
            "exact_match": self.exact_match(predictions, references),
            "f1": self.f1_score(predictions, references)
        }


