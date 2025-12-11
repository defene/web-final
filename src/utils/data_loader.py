"""Data loading utilities for RAG system."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Set
import json


@dataclass
class QAExample:
    """A single QA example with optional context."""
    
    question: str
    answer: str
    question_id: str
    relevant_doc_ids: Optional[Set[str]] = None
    metadata: Optional[Dict] = None


@dataclass
class Document:
    """A document for the knowledge base."""
    
    doc_id: str
    content: str
    title: Optional[str] = None
    metadata: Optional[Dict] = None


class DataLoader:
    """Data loader for QA datasets."""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize data loader.
        
        Args:
            data_dir: Root data directory.
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Ensure directories exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def load_natural_questions(
        self,
        split: str = "train",
        max_examples: Optional[int] = None
    ) -> Tuple[List[QAExample], List[Document]]:
        """Load Natural Questions dataset from HuggingFace.
        
        Args:
            split: Dataset split ('train', 'validation').
            max_examples: Maximum number of examples to load.
            
        Returns:
            Tuple of (QA examples, documents).
        """
        from datasets import load_dataset
        
        print(f"Loading Natural Questions ({split} split)...")
        
        # Load dataset
        dataset = load_dataset("natural_questions", "default", split=split, trust_remote_code=True)
        
        if max_examples:
            dataset = dataset.select(range(min(max_examples, len(dataset))))
        
        qa_examples = []
        documents = []
        doc_set = set()
        
        for idx, item in enumerate(dataset):
            # Extract question
            question = item["question"]["text"]
            
            # Extract answer (short answer if available, otherwise long answer)
            annotations = item["annotations"]
            answer = ""
            
            if annotations["short_answers"][0]["text"]:
                answer = annotations["short_answers"][0]["text"][0] if annotations["short_answers"][0]["text"] else ""
            elif annotations["long_answer"][0]["start_token"] >= 0:
                # Extract long answer from document
                doc_tokens = item["document"]["tokens"]
                start = annotations["long_answer"][0]["start_token"]
                end = annotations["long_answer"][0]["end_token"]
                answer = " ".join(doc_tokens["token"][start:end])
            
            if not answer:
                continue
            
            # Create document from the Wikipedia article
            doc_id = f"nq_doc_{idx}"
            doc_content = " ".join(item["document"]["tokens"]["token"])
            
            if doc_id not in doc_set:
                documents.append(Document(
                    doc_id=doc_id,
                    content=doc_content[:5000],  # Truncate long documents
                    title=item["document"].get("title", ""),
                    metadata={"source": "natural_questions"}
                ))
                doc_set.add(doc_id)
            
            qa_examples.append(QAExample(
                question=question,
                answer=answer,
                question_id=f"nq_{idx}",
                relevant_doc_ids={doc_id},
                metadata={"source": "natural_questions"}
            ))
        
        print(f"Loaded {len(qa_examples)} QA examples and {len(documents)} documents")
        return qa_examples, documents
    
    def load_hotpotqa(
        self,
        split: str = "train",
        max_examples: Optional[int] = None
    ) -> Tuple[List[QAExample], List[Document]]:
        """Load HotpotQA dataset from HuggingFace.
        
        Args:
            split: Dataset split ('train', 'validation').
            max_examples: Maximum number of examples to load.
            
        Returns:
            Tuple of (QA examples, documents).
        """
        from datasets import load_dataset
        
        print(f"Loading HotpotQA ({split} split)...")
        
        # Load dataset
        dataset = load_dataset("hotpot_qa", "fullwiki", split=split, trust_remote_code=True)
        
        if max_examples:
            dataset = dataset.select(range(min(max_examples, len(dataset))))
        
        qa_examples = []
        documents = []
        doc_set = set()
        
        for idx, item in enumerate(dataset):
            question = item["question"]
            answer = item["answer"]
            
            # Process supporting documents
            relevant_doc_ids = set()
            for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
                doc_id = f"hotpot_{hash(title) % 1000000}"
                
                if doc_id not in doc_set:
                    documents.append(Document(
                        doc_id=doc_id,
                        content=" ".join(sentences),
                        title=title,
                        metadata={"source": "hotpotqa"}
                    ))
                    doc_set.add(doc_id)
                
                relevant_doc_ids.add(doc_id)
            
            qa_examples.append(QAExample(
                question=question,
                answer=answer,
                question_id=f"hotpot_{idx}",
                relevant_doc_ids=relevant_doc_ids,
                metadata={
                    "source": "hotpotqa",
                    "type": item.get("type", "unknown"),
                    "level": item.get("level", "unknown")
                }
            ))
        
        print(f"Loaded {len(qa_examples)} QA examples and {len(documents)} documents")
        return qa_examples, documents
    
    def load_squad(
        self,
        split: str = "train",
        max_examples: Optional[int] = None
    ) -> Tuple[List[QAExample], List[Document]]:
        """Load SQuAD dataset from HuggingFace.
        
        Args:
            split: Dataset split ('train', 'validation').
            max_examples: Maximum number of examples to load.
            
        Returns:
            Tuple of (QA examples, documents).
        """
        from datasets import load_dataset
        
        print(f"Loading SQuAD ({split} split)...")
        
        # Load dataset
        dataset = load_dataset("squad", split=split)
        
        if max_examples:
            dataset = dataset.select(range(min(max_examples, len(dataset))))
        
        qa_examples = []
        documents = []
        doc_set = set()
        
        for idx, item in enumerate(dataset):
            question = item["question"]
            answer = item["answers"]["text"][0] if item["answers"]["text"] else ""
            
            if not answer:
                continue
            
            # Use context as document
            context = item["context"]
            doc_id = f"squad_{hash(context) % 1000000}"
            
            if doc_id not in doc_set:
                documents.append(Document(
                    doc_id=doc_id,
                    content=context,
                    title=item.get("title", ""),
                    metadata={"source": "squad"}
                ))
                doc_set.add(doc_id)
            
            qa_examples.append(QAExample(
                question=question,
                answer=answer,
                question_id=f"squad_{idx}",
                relevant_doc_ids={doc_id},
                metadata={"source": "squad"}
            ))
        
        print(f"Loaded {len(qa_examples)} QA examples and {len(documents)} documents")
        return qa_examples, documents
    
    def save_processed(
        self,
        qa_examples: List[QAExample],
        documents: List[Document],
        name: str
    ) -> None:
        """Save processed data to disk.
        
        Args:
            qa_examples: List of QA examples.
            documents: List of documents.
            name: Name for the saved files.
        """
        save_dir = self.processed_dir / name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save QA examples
        qa_data = [
            {
                "question": ex.question,
                "answer": ex.answer,
                "question_id": ex.question_id,
                "relevant_doc_ids": list(ex.relevant_doc_ids) if ex.relevant_doc_ids else [],
                "metadata": ex.metadata
            }
            for ex in qa_examples
        ]
        with open(save_dir / "qa_examples.json", "w", encoding="utf-8") as f:
            json.dump(qa_data, f, ensure_ascii=False, indent=2)
        
        # Save documents
        doc_data = [
            {
                "doc_id": doc.doc_id,
                "content": doc.content,
                "title": doc.title,
                "metadata": doc.metadata
            }
            for doc in documents
        ]
        with open(save_dir / "documents.json", "w", encoding="utf-8") as f:
            json.dump(doc_data, f, ensure_ascii=False, indent=2)
        
        print(f"Saved processed data to {save_dir}")
    
    def load_processed(self, name: str) -> Tuple[List[QAExample], List[Document]]:
        """Load processed data from disk.
        
        Args:
            name: Name of the saved data.
            
        Returns:
            Tuple of (QA examples, documents).
        """
        load_dir = self.processed_dir / name
        
        # Load QA examples
        with open(load_dir / "qa_examples.json", "r", encoding="utf-8") as f:
            qa_data = json.load(f)
        
        qa_examples = [
            QAExample(
                question=item["question"],
                answer=item["answer"],
                question_id=item["question_id"],
                relevant_doc_ids=set(item["relevant_doc_ids"]) if item["relevant_doc_ids"] else None,
                metadata=item.get("metadata")
            )
            for item in qa_data
        ]
        
        # Load documents
        with open(load_dir / "documents.json", "r", encoding="utf-8") as f:
            doc_data = json.load(f)
        
        documents = [
            Document(
                doc_id=item["doc_id"],
                content=item["content"],
                title=item.get("title"),
                metadata=item.get("metadata")
            )
            for item in doc_data
        ]
        
        print(f"Loaded {len(qa_examples)} QA examples and {len(documents)} documents from {load_dir}")
        return qa_examples, documents


