"""Health check and benchmark results for AI Model Providers in VeriFind AI."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from enum import Enum


class ProviderType(str, Enum):
    GEMINI = "gemini"
    AMD_ROCM = "amd_rocm"
    OPENAI_COMPATIBLE = "openai_compatible"
    MOCK = "mock"


@dataclass
class ProviderHealthStatus:
    provider: ProviderType
    model_name: str
    is_available: bool
    endpoint: Optional[str] = None
    latency_ms: Optional[float] = None
    device_info: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceBenchmarkResult:
    provider: ProviderType
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    ttft_ms: float  # Time To First Token in ms
    total_latency_ms: float  # Total latency in ms
    tokens_per_second: float
    gpu_memory_allocated_mb: Optional[float] = None
    gpu_memory_reserved_mb: Optional[float] = None
    device: str = "cpu"
    success: bool = True
    error: Optional[str] = None

