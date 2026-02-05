"""Model recommendations based on hardware capabilities."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from .hardware import HardwareProfile, GPUInfo


@dataclass
class ModelRecommendation:
    """Model recommendation data class."""
    name: str
    description: str
    size_gb: float
    vram_required_mb: int
    ram_required_gb: float
    suitable_for: str
    tags: List[str]
    difficulty: str  # easy, medium, hard


# Model database with hardware requirements
MODELS_DB = [
    # Tiny models (4GB VRAM or CPU only)
    ModelRecommendation(
        name="tinyllama",
        description="TinyLlama 1.1B - Fast, lightweight for basic tasks",
        size_gb=0.6,
        vram_required_mb=1500,
        ram_required_gb=2.0,
        suitable_for="Low-end GPU or CPU",
        tags=["fast", "lightweight", "chat"],
        difficulty="easy"
    ),
    ModelRecommendation(
        name="phi3:mini",
        description="Phi-3 Mini 3.8B - Microsoft's efficient small model",
        size_gb=2.3,
        vram_required_mb=3000,
        ram_required_gb=4.0,
        suitable_for="4GB VRAM GPUs",
        tags=["efficient", "reasoning", "microsoft"],
        difficulty="easy"
    ),
    ModelRecommendation(
        name="qwen2.5:3b",
        description="Qwen2.5 3B - Alibaba's efficient small model",
        size_gb=1.9,
        vram_required_mb=2500,
        ram_required_gb=4.0,
        suitable_for="4GB VRAM GPUs",
        tags=["multilingual", "efficient", "coding"],
        difficulty="easy"
    ),
    
    # Small models (6-8GB VRAM)
    ModelRecommendation(
        name="llama3.2",
        description="Llama 3.2 3B - Meta's latest efficient model",
        size_gb=2.0,
        vram_required_mb=3500,
        ram_required_gb=6.0,
        suitable_for="6GB+ VRAM GPUs",
        tags=["general", "efficient", "meta"],
        difficulty="easy"
    ),
    ModelRecommendation(
        name="gemma:2b",
        description="Gemma 2B - Google's lightweight model",
        size_gb=1.6,
        vram_required_mb=2500,
        ram_required_gb=4.0,
        suitable_for="4GB+ VRAM GPUs",
        tags=["google", "lightweight", "general"],
        difficulty="easy"
    ),
    
    # Medium models (8GB+ VRAM)
    ModelRecommendation(
        name="mistral",
        description="Mistral 7B - High quality 7B parameter model",
        size_gb=4.1,
        vram_required_mb=7000,
        ram_required_gb=10.0,
        suitable_for="8GB+ VRAM GPUs",
        tags=["high-quality", "general", "popular"],
        difficulty="medium"
    ),
    ModelRecommendation(
        name="llama3.1:8b",
        description="Llama 3.1 8B - Meta's powerful 8B model",
        size_gb=4.7,
        vram_required_mb=7500,
        ram_required_gb=12.0,
        suitable_for="8GB+ VRAM GPUs",
        tags=["general", "powerful", "meta"],
        difficulty="medium"
    ),
    
    # Large models (16GB+ VRAM)
    ModelRecommendation(
        name="llama3.1:70b",
        description="Llama 3.1 70B - State-of-the-art large model",
        size_gb=40.0,
        vram_required_mb=48000,
        ram_required_gb=64.0,
        suitable_for="High-end GPUs (RTX 3090/4090)",
        tags=["state-of-the-art", "powerful", "enterprise"],
        difficulty="hard"
    ),
    ModelRecommendation(
        name="mixtral",
        description="Mixtral 8x7B - MoE model with high performance",
        size_gb=26.0,
        vram_required_mb=32000,
        ram_required_gb=48.0,
        suitable_for="High-end GPUs or multi-GPU",
        tags=["moe", "high-performance", "advanced"],
        difficulty="hard"
    ),
    
    # Code models
    ModelRecommendation(
        name="codellama:7b",
        description="CodeLlama 7B - Specialized for coding tasks",
        size_gb=3.8,
        vram_required_mb=6000,
        ram_required_gb=8.0,
        suitable_for="6GB+ VRAM GPUs",
        tags=["coding", "specialized", "meta"],
        difficulty="medium"
    ),
    ModelRecommendation(
        name="qwen2.5-coder:7b",
        description="Qwen2.5 Coder 7B - Excellent coding model",
        size_gb=4.4,
        vram_required_mb=7500,
        ram_required_gb=10.0,
        suitable_for="8GB+ VRAM GPUs",
        tags=["coding", "chinese", "alibaba"],
        difficulty="medium"
    ),
]


def get_vram_mb(profile: HardwareProfile) -> int:
    """Get available VRAM in MB, or 0 if no GPU."""
    if profile.gpu:
        return profile.gpu.vram_total_mb
    return 0


def get_ram_gb(profile: HardwareProfile) -> float:
    """Get available RAM in GB."""
    return profile.ram.total_gb


def score_model_for_hardware(model: ModelRecommendation, profile: HardwareProfile) -> tuple[float, str]:
    """Score how well a model fits the hardware. Returns (score, reason)."""
    vram = get_vram_mb(profile)
    ram = get_ram_gb(profile)
    
    # No GPU - only CPU-suitable models
    if vram == 0:
        if model.vram_required_mb <= 2000:  # Can run on CPU
            return (0.8, "Can run on CPU (slower)")
        return (0.0, "Requires GPU")
    
    # Check VRAM requirements
    if model.vram_required_mb > vram:
        return (0.0, f"Requires {model.vram_required_mb}MB VRAM, you have {vram}MB")
    
    # Check RAM requirements
    if model.ram_required_gb > ram:
        return (0.0, f"Requires {model.ram_required_gb}GB RAM, you have {ram:.1f}GB")
    
    # Calculate fit score
    vram_ratio = model.vram_required_mb / vram
    ram_ratio = model.ram_required_gb / ram
    
    # Models using 50-80% of resources are ideal
    utilization_score = 1.0 - abs(0.65 - vram_ratio)
    
    return (utilization_score, "Good fit for your hardware")


def get_recommendations(profile: HardwareProfile, limit: int = 5) -> List[Dict[str, Any]]:
    """Get model recommendations based on hardware profile."""
    scored_models = []
    
    for model in MODELS_DB:
        score, reason = score_model_for_hardware(model, profile)
        if score > 0:
            scored_models.append((score, model, reason))
    
    # Sort by score (descending)
    scored_models.sort(key=lambda x: x[0], reverse=True)
    
    # Format results
    results = []
    for score, model, reason in scored_models[:limit]:
        results.append({
            "name": model.name,
            "description": model.description,
            "size": f"{model.size_gb:.1f} GB",
            "requirements": f"{model.vram_required_mb}MB VRAM, {model.ram_required_gb}GB RAM",
            "fit_score": f"{score*100:.0f}%",
            "reason": reason,
            "tags": model.tags,
            "difficulty": model.difficulty,
        })
    
    return results


def get_all_models() -> List[Dict[str, Any]]:
    """Get all available models."""
    return [
        {
            "name": m.name,
            "description": m.description,
            "size": f"{m.size_gb:.1f} GB",
            "requirements": f"{m.vram_required_mb}MB VRAM, {m.ram_required_gb}GB RAM",
            "tags": m.tags,
            "difficulty": m.difficulty,
        }
        for m in MODELS_DB
    ]
