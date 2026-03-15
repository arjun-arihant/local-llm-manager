"""Dataset loaders for TruthfulQA and MMLU evaluation subsets."""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
from dataclasses import dataclass, field


DATA_DIR = Path(__file__).parent / "data"


@dataclass
class EvalQuestion:
    """A single evaluation question."""
    id: str
    question: str
    choices: Dict[str, str]          # {"A": "...", "B": "...", ...}
    answer: str                       # correct choice key, e.g. "B"
    category: str = ""                # TruthfulQA: misconceptions, health, ...
    subject: str = ""                 # MMLU: anatomy, machine_learning, ...
    group: str = ""                   # MMLU: STEM, Humanities, ...


class BaseDataset:
    """Base class for evaluation datasets."""

    name: str = "base"
    description: str = ""

    def __init__(self, questions: Optional[List[EvalQuestion]] = None):
        self._questions: List[EvalQuestion] = questions or []

    def __len__(self) -> int:
        return len(self._questions)

    def __iter__(self) -> Iterator[EvalQuestion]:
        return iter(self._questions)

    def __getitem__(self, idx: int) -> EvalQuestion:
        return self._questions[idx]

    def sample(self, n: int, seed: Optional[int] = None) -> List[EvalQuestion]:
        """Return a random sample of n questions."""
        rng = random.Random(seed)
        n = min(n, len(self._questions))
        return rng.sample(self._questions, n)

    def get_subjects(self) -> List[str]:
        """Return unique subjects / categories."""
        seen = set()
        result = []
        for q in self._questions:
            key = q.subject or q.category
            if key and key not in seen:
                seen.add(key)
                result.append(key)
        return result

    def get_groups(self) -> List[str]:
        """Return unique groups (MMLU subject groups)."""
        seen = set()
        result = []
        for q in self._questions:
            if q.group and q.group not in seen:
                seen.add(q.group)
                result.append(q.group)
        return result

    def filter_by_subject(self, subject: str) -> "BaseDataset":
        """Return a new dataset filtered by subject / category."""
        filtered = [
            q for q in self._questions
            if q.subject == subject or q.category == subject
        ]
        ds = self.__class__.__new__(self.__class__)
        ds._questions = filtered
        return ds

    def filter_by_group(self, group: str) -> "BaseDataset":
        """Return a new dataset filtered by group."""
        filtered = [q for q in self._questions if q.group == group]
        ds = self.__class__.__new__(self.__class__)
        ds._questions = filtered
        return ds

    @property
    def stats(self) -> Dict[str, Any]:
        """Return dataset statistics."""
        subjects: Dict[str, int] = {}
        groups: Dict[str, int] = {}
        for q in self._questions:
            key = q.subject or q.category
            if key:
                subjects[key] = subjects.get(key, 0) + 1
            if q.group:
                groups[q.group] = groups.get(q.group, 0) + 1
        return {
            "name": self.name,
            "total_questions": len(self._questions),
            "subjects": subjects,
            "groups": groups,
        }


class TruthfulQADataset(BaseDataset):
    """TruthfulQA evaluation subset — tests the model's ability to avoid
    common misconceptions and provide truthful answers."""

    name = "TruthfulQA"
    description = "Tests model truthfulness on common misconceptions"

    def __init__(self):
        super().__init__()
        self._load()

    def _load(self):
        path = DATA_DIR / "truthfulqa_subset.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._questions = [
            EvalQuestion(
                id=item["id"],
                question=item["question"],
                choices=item["choices"],
                answer=item["answer"],
                category=item.get("category", ""),
            )
            for item in data
        ]


class MMLUDataset(BaseDataset):
    """MMLU (Massive Multitask Language Understanding) evaluation subset —
    tests the model across STEM, Humanities, Social Sciences, and Other."""

    name = "MMLU"
    description = "Tests broad knowledge across academic subjects"

    def __init__(self):
        super().__init__()
        self._load()

    def _load(self):
        path = DATA_DIR / "mmlu_subset.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._questions = [
            EvalQuestion(
                id=item["id"],
                question=item["question"],
                choices=item["choices"],
                answer=item["answer"],
                subject=item.get("subject", ""),
                group=item.get("group", ""),
            )
            for item in data
        ]


class DatasetManager:
    """Convenience loader for all available datasets."""

    AVAILABLE = {
        "truthfulqa": TruthfulQADataset,
        "mmlu": MMLUDataset,
    }

    @classmethod
    def load(cls, name: str) -> BaseDataset:
        """Load a dataset by name."""
        name_lower = name.lower()
        if name_lower not in cls.AVAILABLE:
            raise ValueError(
                f"Unknown dataset: {name}. "
                f"Available: {', '.join(cls.AVAILABLE.keys())}"
            )
        return cls.AVAILABLE[name_lower]()

    @classmethod
    def load_all(cls) -> Dict[str, BaseDataset]:
        """Load all available datasets."""
        return {name: loader() for name, loader in cls.AVAILABLE.items()}

    @classmethod
    def list_datasets(cls) -> List[Dict[str, Any]]:
        """Return metadata about all available datasets."""
        result = []
        for name, loader in cls.AVAILABLE.items():
            ds = loader()
            result.append({
                "name": name,
                "display_name": ds.name,
                "description": ds.description,
                "total_questions": len(ds),
                "subjects": ds.get_subjects(),
                "groups": ds.get_groups(),
            })
        return result
