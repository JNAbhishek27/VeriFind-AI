"""Model Provider Factory for VeriFind AI.

Configures and instantiates the active inference provider based on environment
variables or explicit configuration:
- 'gemini': Google Gemini API (gemini-3.8-flash, etc.)
- 'amd_rocm': Open-source Qwen model served on AMD Developer Cloud / local ROCm (MI300X, MI250, Radeon)
- 'openai_compatible': Any standard OpenAI-compatible vLLM/TGI server
"""

import os
import logging
from typing import Optional

from backend.ai_engine_base import BaseAIEngine
from backend.provider_models import ProviderType
from backend.gemini_engine import GeminiAIEngine
from backend.rocm_provider import AmdROCmProvider

logger = logging.getLogger(__name__)


class ModelProviderFactory:
    """Factory resolving the active AI model inference provider."""

    @staticmethod
    def get_provider(
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        endpoint_url: Optional[str] = None
    ) -> BaseAIEngine:
        """Instantiates the specified or configured model provider.

        Args:
            provider_name: 'gemini', 'amd_rocm', or 'openai_compatible'.
                           If omitted, reads from MODEL_PROVIDER env var (defaults to 'gemini').
            model_name: Optional override for the target model name.
            endpoint_url: Optional override for the inference server URL.

        Returns:
            An instance conforming to BaseAIEngine.
        """
        raw_provider = (
            provider_name
            or os.getenv("MODEL_PROVIDER")
            or "gemini"
        ).strip().lower()

        logger.info("Initializing Model Provider: '%s'", raw_provider)

        if raw_provider in ("amd_rocm", "rocm", "amd"):
            endpoint = endpoint_url or os.getenv("ROCM_INFERENCE_ENDPOINT") or "http://127.0.0.1:8000/v1"
            model = model_name or os.getenv("ROCM_MODEL_NAME") or "Qwen/Qwen2.5-7B-Instruct"
            logger.info("Configured AMD ROCm Provider -> Endpoint: %s, Model: %s", endpoint, model)
            return AmdROCmProvider(
                endpoint_url=endpoint,
                model_name=model,
                provider_type=ProviderType.AMD_ROCM
            )

        elif raw_provider in ("openai_compatible", "openai", "vllm", "tgi"):
            endpoint = endpoint_url or os.getenv("OPENAI_COMPATIBLE_ENDPOINT") or "http://127.0.0.1:8000/v1"
            model = model_name or os.getenv("OPENAI_MODEL_NAME") or "Qwen/Qwen2.5-7B-Instruct"
            logger.info("Configured OpenAI-Compatible Provider -> Endpoint: %s, Model: %s", endpoint, model)
            return AmdROCmProvider(
                endpoint_url=endpoint,
                model_name=model,
                provider_type=ProviderType.OPENAI_COMPATIBLE
            )

        else:
            # Default development provider: Google Gemini
            model = model_name or os.getenv("GEMINI_MODEL") or "gemini-3.8-flash"
            logger.info("Configured Google Gemini Provider -> Model: %s", model)
            return GeminiAIEngine(model_name=model)

