"""Evaluation harness engine for running benchmarks against Ollama models."""

import re
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

from .datasets import BaseDataset, EvalQuestion, DatasetManager
from .ollama_client import OllamaClient
from .hardware import get_hardware_profile


@dataclass
class QuestionResult:
    """Result for a single question evaluation."""
    question_id: str
    question_text: str
    expected_answer: str
    model_answer: str
    is_correct: bool
    latency_ms: float
    raw_response: str
    subject: str = ""
    group: str = ""


@dataclass
class EvalResult:
    """Aggregated evaluation result for one model on one dataset."""
    model_name: str
    dataset_name: str
    timestamp: datetime
    total_questions: int
    correct_count: int
    accuracy: float

    # Subject / group breakdown
    per_subject_accuracy: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    per_group_accuracy: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Latency stats
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0

    # Throughput
    total_tokens: int = 0
    tokens_per_second: float = 0.0
    total_eval_time_s: float = 0.0

    # Hardware context
    hardware_info: Dict[str, Any] = field(default_factory=dict)

    # Detailed results
    question_results: List[QuestionResult] = field(default_factory=list)


@dataclass
class EvalConfig:
    """Configuration for an evaluation run."""
    models: List[str]
    datasets: List[str] = field(default_factory=lambda: ["truthfulqa", "mmlu"])
    num_samples: Optional[int] = None      # None = run all questions
    temperature: float = 0.0               # Deterministic by default
    seed: Optional[int] = 42               # For reproducible sampling
    max_retries: int = 2
    timeout_seconds: int = 120


def _build_mc_prompt(question: EvalQuestion) -> str:
    """Build a multiple-choice prompt designed for reliable answer extraction."""
    choices_text = "\n".join(
        f"  {key}. {value}" for key, value in sorted(question.choices.items())
    )
    return (
        f"Answer the following multiple-choice question by responding with ONLY "
        f"the letter (A, B, C, or D) of the correct answer. "
        f"Do not explain.\n\n"
        f"Question: {question.question}\n"
        f"{choices_text}\n\n"
        f"Answer:"
    )


def _extract_answer(response: str) -> str:
    """Extract the answer letter from model response.

    Tries multiple strategies:
    1. Exact single-letter match at the start
    2. Pattern like "A." or "(A)" or "A)"
    3. First capital letter A-D in the response
    """
    text = response.strip()

    # Strategy 1: single letter
    if len(text) == 1 and text.upper() in "ABCD":
        return text.upper()

    # Strategy 2: starts with letter followed by punctuation
    match = re.match(r"^[(\s]*([A-Da-d])[).:\s]", text)
    if match:
        return match.group(1).upper()

    # Strategy 3: first occurrence of a standalone A-D
    match = re.search(r"\b([A-Da-d])\b", text)
    if match:
        return match.group(1).upper()

    # Strategy 4: any A-D character
    for ch in text:
        if ch.upper() in "ABCD":
            return ch.upper()

    return ""


class EvalRunner:
    """Runs evaluations against Ollama models."""

    def __init__(
        self,
        client: Optional[OllamaClient] = None,
        config: Optional[EvalConfig] = None,
    ):
        self.client = client or OllamaClient()
        self.config = config or EvalConfig(models=[])

    def _run_single_question(
        self, model: str, question: EvalQuestion
    ) -> QuestionResult:
        """Evaluate a single question with retry logic."""
        prompt = _build_mc_prompt(question)
        raw_response = ""
        model_answer = ""
        latency_ms = 0.0

        for attempt in range(self.config.max_retries + 1):
            try:
                start = time.perf_counter()
                result = self.client.generate(
                    model,
                    prompt,
                    stream=False,
                )
                elapsed = time.perf_counter() - start
                latency_ms = elapsed * 1000

                raw_response = result.get("response", "")
                model_answer = _extract_answer(raw_response)

                if model_answer:
                    break  # Got a valid answer
            except Exception:
                if attempt == self.config.max_retries:
                    break
                time.sleep(1)

        is_correct = model_answer == question.answer

        return QuestionResult(
            question_id=question.id,
            question_text=question.question,
            expected_answer=question.answer,
            model_answer=model_answer,
            is_correct=is_correct,
            latency_ms=latency_ms,
            raw_response=raw_response[:500],  # Truncate for storage
            subject=question.subject or question.category,
            group=question.group,
        )

    def evaluate(
        self,
        model: str,
        dataset: BaseDataset,
        num_samples: Optional[int] = None,
        progress_callback=None,
    ) -> EvalResult:
        """Run full evaluation of a model on a dataset.

        Args:
            model: Ollama model name.
            dataset: Dataset to evaluate on.
            num_samples: Limit number of questions (None = all).
            progress_callback: Optional callable(current, total, question_result).

        Returns:
            EvalResult with accuracy and latency breakdown.
        """
        # Select questions
        n = num_samples or self.config.num_samples
        if n and n < len(dataset):
            questions = dataset.sample(n, seed=self.config.seed)
        else:
            questions = list(dataset)

        total = len(questions)
        question_results: List[QuestionResult] = []
        start_time = time.perf_counter()

        for i, question in enumerate(questions):
            qr = self._run_single_question(model, question)
            question_results.append(qr)
            if progress_callback:
                progress_callback(i + 1, total, qr)

        total_time = time.perf_counter() - start_time

        # Compute aggregate metrics
        correct = sum(1 for qr in question_results if qr.is_correct)
        accuracy = correct / total if total > 0 else 0.0

        # Latency stats
        latencies = sorted(qr.latency_ms for qr in question_results)
        avg_lat = sum(latencies) / len(latencies) if latencies else 0
        p50 = latencies[len(latencies) // 2] if latencies else 0
        p95_idx = int(len(latencies) * 0.95)
        p95 = latencies[min(p95_idx, len(latencies) - 1)] if latencies else 0

        # Per-subject breakdown
        per_subject: Dict[str, Dict[str, Any]] = {}
        for qr in question_results:
            key = qr.subject
            if not key:
                continue
            if key not in per_subject:
                per_subject[key] = {"total": 0, "correct": 0}
            per_subject[key]["total"] += 1
            if qr.is_correct:
                per_subject[key]["correct"] += 1
        for key in per_subject:
            s = per_subject[key]
            s["accuracy"] = round(s["correct"] / s["total"], 4) if s["total"] > 0 else 0

        # Per-group breakdown
        per_group: Dict[str, Dict[str, Any]] = {}
        for qr in question_results:
            key = qr.group
            if not key:
                continue
            if key not in per_group:
                per_group[key] = {"total": 0, "correct": 0}
            per_group[key]["total"] += 1
            if qr.is_correct:
                per_group[key]["correct"] += 1
        for key in per_group:
            g = per_group[key]
            g["accuracy"] = round(g["correct"] / g["total"], 4) if g["total"] > 0 else 0

        # Hardware info
        try:
            hw = get_hardware_profile()
            hardware_info = {
                "cpu": hw.cpu.brand,
                "cores": hw.cpu.cores,
                "ram_gb": round(hw.ram.total_gb, 1),
                "gpu": hw.gpu.name if hw.gpu else None,
                "vram_mb": hw.gpu.vram_total_mb if hw.gpu else 0,
            }
        except Exception:
            hardware_info = {}

        return EvalResult(
            model_name=model,
            dataset_name=dataset.name,
            timestamp=datetime.now(),
            total_questions=total,
            correct_count=correct,
            accuracy=round(accuracy, 4),
            per_subject_accuracy=per_subject,
            per_group_accuracy=per_group,
            avg_latency_ms=round(avg_lat, 1),
            p50_latency_ms=round(p50, 1),
            p95_latency_ms=round(p95, 1),
            min_latency_ms=round(min(latencies) if latencies else 0, 1),
            max_latency_ms=round(max(latencies) if latencies else 0, 1),
            total_eval_time_s=round(total_time, 2),
            hardware_info=hardware_info,
            question_results=question_results,
        )

    def evaluate_multiple(
        self,
        models: List[str],
        dataset_names: Optional[List[str]] = None,
        num_samples: Optional[int] = None,
        progress_callback=None,
    ) -> Dict[str, Dict[str, EvalResult]]:
        """Evaluate multiple models on multiple datasets.

        Returns:
            Nested dict: {model_name: {dataset_name: EvalResult}}
        """
        dataset_names = dataset_names or self.config.datasets
        results: Dict[str, Dict[str, EvalResult]] = {}

        for model in models:
            results[model] = {}
            for ds_name in dataset_names:
                dataset = DatasetManager.load(ds_name)
                result = self.evaluate(
                    model, dataset, num_samples, progress_callback
                )
                results[model][ds_name] = result

        return results
