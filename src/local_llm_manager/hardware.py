"""Hardware detection module for GPU, CPU, and RAM."""

import subprocess
import psutil
import cpuinfo
from typing import Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class GPUInfo:
    """GPU information data class."""
    name: str
    vram_total_mb: int
    vram_used_mb: int
    driver: str
    cuda_available: bool
    cuda_version: Optional[str] = None


@dataclass
class CPUInfo:
    """CPU information data class."""
    brand: str
    cores: int
    threads: int
    architecture: str
    frequency_mhz: float


@dataclass
class RAMInfo:
    """RAM information data class."""
    total_gb: float
    available_gb: float
    used_gb: float
    percent_used: float


@dataclass
class HardwareProfile:
    """Complete hardware profile."""
    gpu: Optional[GPUInfo]
    cpu: CPUInfo
    ram: RAMInfo


def detect_nvidia_gpu() -> Optional[GPUInfo]:
    """Detect NVIDIA GPU using nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            if len(parts) >= 4:
                # Parse VRAM (convert MiB to MB roughly)
                vram_total = int(float(parts[1].strip()))
                vram_used = int(float(parts[2].strip()))
                
                # Check CUDA version
                cuda_version = None
                try:
                    cuda_result = subprocess.run(
                        ["nvcc", "--version"],
                        capture_output=True,
                        text=True,
                        timeout=2
                    )
                    if cuda_result.returncode == 0:
                        for line in cuda_result.stdout.split("\n"):
                            if "release" in line:
                                cuda_version = line.split("release")[-1].split(",")[0].strip()
                                break
                except:
                    pass
                
                return GPUInfo(
                    name=parts[0].strip(),
                    vram_total_mb=vram_total,
                    vram_used_mb=vram_used,
                    driver=parts[3].strip(),
                    cuda_available=cuda_version is not None,
                    cuda_version=cuda_version
                )
    except:
        pass
    return None


def detect_amd_gpu() -> Optional[GPUInfo]:
    """Detect AMD GPU (basic support)."""
    try:
        # Try rocm-smi for AMD GPUs
        result = subprocess.run(
            ["rocm-smi", "--showproductname", "--showmeminfo", "vram"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            for line in lines:
                if "GPU" in line and ":" in line:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        return GPUInfo(
                            name=parts[1].strip(),
                            vram_total_mb=8192,  # Default estimate
                            vram_used_mb=0,
                            driver="ROCm",
                            cuda_available=False
                        )
    except:
        pass
    return None


def detect_gpu() -> Optional[GPUInfo]:
    """Detect GPU (NVIDIA or AMD)."""
    gpu = detect_nvidia_gpu()
    if gpu:
        return gpu
    return detect_amd_gpu()


def detect_cpu() -> CPUInfo:
    """Detect CPU information."""
    info = cpuinfo.get_cpu_info()
    
    return CPUInfo(
        brand=info.get("brand_raw", "Unknown CPU"),
        cores=psutil.cpu_count(logical=False) or 1,
        threads=psutil.cpu_count(logical=True) or 1,
        architecture=info.get("arch", "unknown"),
        frequency_mhz=info.get("hz_advertised_friendly", "0 MHz").split()[0] if info.get("hz_advertised_friendly") else psutil.cpu_freq().max if psutil.cpu_freq() else 0
    )


def detect_ram() -> RAMInfo:
    """Detect RAM information."""
    mem = psutil.virtual_memory()
    return RAMInfo(
        total_gb=mem.total / (1024**3),
        available_gb=mem.available / (1024**3),
        used_gb=mem.used / (1024**3),
        percent_used=mem.percent
    )


def get_hardware_profile() -> HardwareProfile:
    """Get complete hardware profile."""
    return HardwareProfile(
        gpu=detect_gpu(),
        cpu=detect_cpu(),
        ram=detect_ram()
    )


def format_hardware_profile(profile: HardwareProfile) -> Dict[str, Any]:
    """Format hardware profile for display."""
    data = {
        "CPU": {
            "Model": profile.cpu.brand,
            "Cores/Threads": f"{profile.cpu.cores}/{profile.cpu.threads}",
            "Architecture": profile.cpu.architecture,
        },
        "RAM": {
            "Total": f"{profile.ram.total_gb:.1f} GB",
            "Available": f"{profile.ram.available_gb:.1f} GB",
            "Used": f"{profile.ram.percent_used:.1f}%",
        }
    }
    
    if profile.gpu:
        data["GPU"] = {
            "Model": profile.gpu.name,
            "VRAM": f"{profile.gpu.vram_used_mb}/{profile.gpu.vram_total_mb} MB",
            "Driver": profile.gpu.driver,
            "CUDA": profile.gpu.cuda_version if profile.gpu.cuda_available else "Not available",
        }
    else:
        data["GPU"] = {"Status": "No dedicated GPU detected"}
    
    return data
