"""Quantization helper for optimizing models."""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from .hardware import HardwareProfile, get_hardware_profile
from .ollama_client import OllamaClient


@dataclass
class QuantizeConfig:
    """Quantization configuration."""
    bits: int
    name_suffix: str
    vram_reduction: float  # Percentage of VRAM saved
    quality_impact: str  # minimal, moderate, significant
    description: str


# Quantization options ordered by efficiency
QUANTIZE_OPTIONS = [
    QuantizeConfig(
        bits=8,
        name_suffix="q8_0",
        vram_reduction=0.5,
        quality_impact="minimal",
        description="8-bit quantization - nearly lossless, 50% smaller"
    ),
    QuantizeConfig(
        bits=4,
        name_suffix="q4_k_m",
        vram_reduction=0.75,
        quality_impact="minimal",
        description="4-bit (K-means medium) - best balance, 75% smaller"
    ),
    QuantizeConfig(
        bits=4,
        name_suffix="q4_0",
        vram_reduction=0.75,
        quality_impact="moderate",
        description="4-bit standard - faster, slightly lower quality"
    ),
    QuantizeConfig(
        bits=3,
        name_suffix="q3_k_m",
        vram_reduction=0.81,
        quality_impact="moderate",
        description="3-bit (K-means medium) - aggressive compression"
    ),
    QuantizeConfig(
        bits=2,
        name_suffix="q2_k",
        vram_reduction=0.87,
        quality_impact="significant",
        description="2-bit (K-means) - maximum compression"
    ),
]


def estimate_vram_usage(base_size_gb: float, quantize_bits: int) -> float:
    """Estimate VRAM usage after quantization."""
    # Rough estimate: 4-bit is ~75% smaller than 16-bit (fp16)
    ratio = quantize_bits / 16.0
    return base_size_gb * ratio


def recommend_quantization(profile: HardwareProfile, model_size_gb: float) -> Optional[QuantizeConfig]:
    """Recommend best quantization for hardware."""
    vram_mb = profile.gpu.vram_total_mb if profile.gpu else 0
    vram_gb = vram_mb / 1024
    
    # Find best quantization that fits in VRAM with some headroom (80% usage max)
    target_vram = vram_gb * 0.8
    
    for config in QUANTIZE_OPTIONS:
        estimated = estimate_vram_usage(model_size_gb, config.bits)
        if estimated <= target_vram:
            return config
    
    # If nothing fits, return the most aggressive
    return QUANTIZE_OPTIONS[-1]


def quantize_model(
    source_model: str,
    quantize_level: str,
    client: Optional[OllamaClient] = None
) -> str:
    """Create a quantized version of a model."""
    if client is None:
        client = OllamaClient()
    
    # Get model info
    model_info = client.show_model(source_model)
    if not model_info:
        raise ValueError(f"Model {source_model} not found")
    
    # Create new model name
    new_name = f"{source_model}-{quantize_level}"
    
    # Build Modelfile with quantization
    modelfile = f"""FROM {source_model}
PARAMETER quantization {quantize_level}
"""
    
    # Add metadata if available
    if "parameters" in model_info:
        modelfile += f'PARAMETER num_ctx 4096\n'
    
    # Create the quantized model
    print(f"Creating quantized model: {new_name}")
    print(f"Using quantization: {quantize_level}")
    
    for progress in client.create_model(new_name, modelfile):
        status = progress.get("status", "")
        if status:
            print(f"  {status}")
    
    return new_name


def auto_quantize(
    model_name: str,
    profile: Optional[HardwareProfile] = None,
    client: Optional[OllamaClient] = None
) -> str:
    """Automatically quantize model for hardware."""
    if client is None:
        client = OllamaClient()
    
    if profile is None:
        profile = get_hardware_profile()
    
    # Get model info to estimate size
    model_info = client.show_model(model_name)
    if not model_info:
        raise ValueError(f"Model {model_name} not found")
    
    # Estimate model size from parameters (rough estimate)
    # 7B model ~ 4GB in 4-bit, 8B ~ 4.7GB, etc.
    param_count = model_info.get("details", {}).get("parameter_size", "")
    if "7B" in param_count or "8B" in param_count:
        model_size_gb = 4.5
    elif "3B" in param_count:
        model_size_gb = 2.0
    elif "70B" in param_count:
        model_size_gb = 40.0
    else:
        model_size_gb = 4.0  # Default estimate
    
    config = recommend_quantization(profile, model_size_gb)
    
    print(f"Hardware: {profile.gpu.name if profile.gpu else 'CPU'} with {profile.gpu.vram_total_mb if profile.gpu else 0}MB VRAM")
    print(f"Recommended quantization: {config.description}")
    
    return quantize_model(model_name, config.name_suffix, client)


def get_quantize_options() -> List[Dict[str, Any]]:
    """Get all quantization options."""
    return [
        {
            "level": opt.name_suffix,
            "bits": opt.bits,
            "vram_reduction": f"{opt.vram_reduction*100:.0f}%",
            "quality_impact": opt.quality_impact,
            "description": opt.description,
        }
        for opt in QUANTIZE_OPTIONS
    ]
