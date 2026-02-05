"""Benchmarking module for testing model performance."""

import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from .ollama_client import OllamaClient
from .hardware import get_hardware_profile, format_hardware_profile
from .database import BenchmarkDB, BenchmarkResult


# Standard benchmark prompts of varying complexity
BENCHMARK_PROMPTS = {
    "short": "What is 2+2? Answer with just the number.",
    "medium": "Explain quantum computing in simple terms, keeping your answer under 100 words.",
    "long": "Write a Python function to calculate the Fibonacci sequence with memoization. Include docstring and example usage.",
    "creative": "Write a haiku about artificial intelligence.",
}


@dataclass
class BenchmarkStats:
    """Benchmark statistics."""
    model_name: str
    tokens_per_second: float
    prompt_eval_rate: float
    total_duration_ms: int
    load_duration_ms: int
    prompt_eval_count: int
    eval_count: int
    prompt_type: str


def benchmark_model(
    model_name: str,
    client: Optional[OllamaClient] = None,
    prompt_type: str = "medium",
    save: bool = True
) -> BenchmarkStats:
    """Benchmark a model and optionally save results."""
    if client is None:
        client = OllamaClient()
    
    prompt = BENCHMARK_PROMPTS.get(prompt_type, BENCHMARK_PROMPTS["medium"])
    
    # Run benchmark
    print(f"Running benchmark for {model_name}...")
    print(f"Prompt: {prompt[:50]}...")
    
    result = client.generate(model_name, prompt, stream=False)
    
    # Extract metrics
    eval_count = result.get("eval_count", 0)
    eval_duration_ns = result.get("eval_duration", 1)
    prompt_eval_count = result.get("prompt_eval_count", 0)
    prompt_eval_duration_ns = result.get("prompt_eval_duration", 1)
    total_duration_ns = result.get("total_duration", 1)
    load_duration_ns = result.get("load_duration", 0)
    
    # Calculate tokens per second
    eval_duration_s = eval_duration_ns / 1e9
    tokens_per_second = eval_count / eval_duration_s if eval_duration_s > 0 else 0
    
    # Calculate prompt eval rate
    prompt_eval_duration_s = prompt_eval_duration_ns / 1e9
    prompt_eval_rate = prompt_eval_count / prompt_eval_duration_s if prompt_eval_duration_s > 0 else 0
    
    stats = BenchmarkStats(
        model_name=model_name,
        tokens_per_second=tokens_per_second,
        prompt_eval_rate=prompt_eval_rate,
        total_duration_ms=int(total_duration_ns / 1e6),
        load_duration_ms=int(load_duration_ns / 1e6),
        prompt_eval_count=prompt_eval_count,
        eval_count=eval_count,
        prompt_type=prompt_type
    )
    
    # Save to database
    if save:
        db = BenchmarkDB()
        hardware = get_hardware_profile()
        hardware_info = {
            "cpu": hardware.cpu.brand,
            "cores": hardware.cpu.cores,
            "ram_gb": hardware.ram.total_gb,
            "gpu": hardware.gpu.name if hardware.gpu else None,
            "vram_mb": hardware.gpu.vram_total_mb if hardware.gpu else 0,
        }
        
        benchmark_result = BenchmarkResult(
            id=None,
            model_name=model_name,
            timestamp=datetime.now(),
            tokens_per_second=tokens_per_second,
            prompt_eval_rate=prompt_eval_rate,
            total_duration_ms=stats.total_duration_ms,
            load_duration_ms=stats.load_duration_ms,
            prompt_eval_count=prompt_eval_count,
            eval_count=eval_count,
            quantize_level=None,
            hardware_info=hardware_info
        )
        
        db.save_benchmark(benchmark_result)
        print(f"✓ Benchmark saved to database")
    
    return stats


def format_benchmark_stats(stats: BenchmarkStats) -> Dict[str, str]:
    """Format benchmark stats for display."""
    return {
        "Model": stats.model_name,
        "Tokens/sec": f"{stats.tokens_per_second:.2f}",
        "Prompt eval rate": f"{stats.prompt_eval_rate:.2f} tok/s",
        "Total time": f"{stats.total_duration_ms}ms",
        "Load time": f"{stats.load_duration_ms}ms",
        "Tokens generated": str(stats.eval_count),
        "Prompt tokens": str(stats.prompt_eval_count),
    }


def run_full_benchmark(model_name: str, client: Optional[OllamaClient] = None) -> Dict[str, BenchmarkStats]:
    """Run benchmarks with multiple prompt types."""
    if client is None:
        client = OllamaClient()
    
    results = {}
    for prompt_type in ["short", "medium", "long"]:
        print(f"\n--- Testing with {prompt_type} prompt ---")
        stats = benchmark_model(model_name, client, prompt_type, save=False)
        results[prompt_type] = stats
        print(f"Tokens/sec: {stats.tokens_per_second:.2f}")
    
    # Save average of all runs
    if results:
        avg_tps = sum(r.tokens_per_second for r in results.values()) / len(results)
        db = BenchmarkDB()
        hardware = get_hardware_profile()
        hardware_info = {
            "cpu": hardware.cpu.brand,
            "cores": hardware.cpu.cores,
            "ram_gb": hardware.ram.total_gb,
            "gpu": hardware.gpu.name if hardware.gpu else None,
            "vram_mb": hardware.gpu.vram_total_mb if hardware.gpu else 0,
        }
        
        benchmark_result = BenchmarkResult(
            id=None,
            model_name=model_name,
            timestamp=datetime.now(),
            tokens_per_second=avg_tps,
            prompt_eval_rate=sum(r.prompt_eval_rate for r in results.values()) / len(results),
            total_duration_ms=int(sum(r.total_duration_ms for r in results.values()) / len(results)),
            load_duration_ms=int(sum(r.load_duration_ms for r in results.values()) / len(results)),
            prompt_eval_count=int(sum(r.prompt_eval_count for r in results.values()) / len(results)),
            eval_count=int(sum(r.eval_count for r in results.values()) / len(results)),
            quantize_level=None,
            hardware_info=hardware_info
        )
        db.save_benchmark(benchmark_result)
        print(f"\n✓ Average benchmark saved to database")
    
    return results
