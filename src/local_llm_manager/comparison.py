"""Comparison engine for quantized vs full-precision model analysis."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from .eval_harness import EvalResult


@dataclass
class ModelComparison:
    """Comparison between two models (e.g. full vs quantized)."""
    model_a: str                  # Baseline (typically full-precision)
    model_b: str                  # Challenger (typically quantized)
    dataset_name: str

    # Accuracy
    accuracy_a: float
    accuracy_b: float
    accuracy_delta: float         # B - A (negative = regression)
    accuracy_pct_change: float    # % change from A to B

    # Latency
    avg_latency_a_ms: float
    avg_latency_b_ms: float
    latency_speedup: float        # A / B (> 1 = B is faster)

    # P95 latency
    p95_latency_a_ms: float
    p95_latency_b_ms: float

    # Per-subject deltas
    per_subject_delta: Dict[str, float] = field(default_factory=dict)
    per_group_delta: Dict[str, float] = field(default_factory=dict)

    # Quality-efficiency score
    quality_efficiency_score: float = 0.0

    @property
    def summary(self) -> str:
        """One-line summary of the comparison."""
        direction = "faster" if self.latency_speedup > 1 else "slower"
        acc_dir = "higher" if self.accuracy_delta > 0 else "lower"
        return (
            f"{self.model_b} is {abs(self.latency_speedup):.1f}x {direction} "
            f"with {abs(self.accuracy_delta)*100:.1f}% {acc_dir} accuracy "
            f"vs {self.model_a} on {self.dataset_name}"
        )


def compare_models(result_a: EvalResult, result_b: EvalResult) -> ModelComparison:
    """Compare two evaluation results.

    Typically result_a is the full-precision baseline and result_b
    is the quantized / alternative model.
    """
    acc_delta = result_b.accuracy - result_a.accuracy
    acc_pct = (acc_delta / result_a.accuracy * 100) if result_a.accuracy > 0 else 0.0

    lat_speedup = (
        result_a.avg_latency_ms / result_b.avg_latency_ms
        if result_b.avg_latency_ms > 0 else 0.0
    )

    # Per-subject delta
    all_subjects = set(result_a.per_subject_accuracy.keys()) | set(result_b.per_subject_accuracy.keys())
    per_subject_delta: Dict[str, float] = {}
    for subj in all_subjects:
        acc_a = result_a.per_subject_accuracy.get(subj, {}).get("accuracy", 0)
        acc_b = result_b.per_subject_accuracy.get(subj, {}).get("accuracy", 0)
        per_subject_delta[subj] = round(acc_b - acc_a, 4)

    # Per-group delta
    all_groups = set(result_a.per_group_accuracy.keys()) | set(result_b.per_group_accuracy.keys())
    per_group_delta: Dict[str, float] = {}
    for grp in all_groups:
        acc_a = result_a.per_group_accuracy.get(grp, {}).get("accuracy", 0)
        acc_b = result_b.per_group_accuracy.get(grp, {}).get("accuracy", 0)
        per_group_delta[grp] = round(acc_b - acc_a, 4)

    # Quality-efficiency score: harmonic mean of accuracy retention and speed gain
    acc_retention = result_b.accuracy / result_a.accuracy if result_a.accuracy > 0 else 1.0
    speed_gain = min(lat_speedup, 5.0) / 5.0  # Normalize speed gain to 0-1 range
    if acc_retention + speed_gain > 0:
        qe_score = 2 * (acc_retention * speed_gain) / (acc_retention + speed_gain)
    else:
        qe_score = 0.0

    return ModelComparison(
        model_a=result_a.model_name,
        model_b=result_b.model_name,
        dataset_name=result_a.dataset_name,
        accuracy_a=result_a.accuracy,
        accuracy_b=result_b.accuracy,
        accuracy_delta=round(acc_delta, 4),
        accuracy_pct_change=round(acc_pct, 2),
        avg_latency_a_ms=result_a.avg_latency_ms,
        avg_latency_b_ms=result_b.avg_latency_ms,
        latency_speedup=round(lat_speedup, 2),
        p95_latency_a_ms=result_a.p95_latency_ms,
        p95_latency_b_ms=result_b.p95_latency_ms,
        per_subject_delta=per_subject_delta,
        per_group_delta=per_group_delta,
        quality_efficiency_score=round(qe_score, 4),
    )


@dataclass
class LeaderboardEntry:
    """Single row in the leaderboard."""
    rank: int
    model_name: str
    dataset_name: str
    accuracy: float
    avg_latency_ms: float
    p95_latency_ms: float
    total_questions: int
    correct_count: int
    timestamp: str


def build_leaderboard(
    results: List[EvalResult],
    sort_by: str = "accuracy",
) -> List[LeaderboardEntry]:
    """Build a ranked leaderboard from multiple eval results.

    Args:
        results: List of EvalResult objects.
        sort_by: 'accuracy', 'latency', or 'efficiency'.

    Returns:
        Sorted list of LeaderboardEntry objects.
    """
    entries = []
    for r in results:
        entries.append(LeaderboardEntry(
            rank=0,  # Will be set after sorting
            model_name=r.model_name,
            dataset_name=r.dataset_name,
            accuracy=r.accuracy,
            avg_latency_ms=r.avg_latency_ms,
            p95_latency_ms=r.p95_latency_ms,
            total_questions=r.total_questions,
            correct_count=r.correct_count,
            timestamp=r.timestamp.strftime("%Y-%m-%d %H:%M"),
        ))

    # Sort
    if sort_by == "latency":
        entries.sort(key=lambda e: e.avg_latency_ms)
    elif sort_by == "efficiency":
        # Composite: accuracy / (latency in seconds)
        entries.sort(
            key=lambda e: e.accuracy / (e.avg_latency_ms / 1000) if e.avg_latency_ms > 0 else 0,
            reverse=True
        )
    else:  # accuracy (default)
        entries.sort(key=lambda e: e.accuracy, reverse=True)

    # Assign ranks
    for i, entry in enumerate(entries):
        entry.rank = i + 1

    return entries
