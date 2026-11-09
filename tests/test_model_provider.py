"""Tests for model provider factory and AMD ROCm provider.

Covers:
- ModelProviderFactory provider resolution from string / env var
- AmdROCmProvider health_check with mocked HTTP (server down / server up)
- AmdROCmProvider benchmark with mocked streaming response
"""

import asyncio
import json
import os
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.gemini_engine import GeminiAIEngine
from backend.model_factory import ModelProviderFactory
from backend.provider_models import InferenceBenchmarkResult, ProviderHealthStatus, ProviderType
from backend.rocm_provider import AmdROCmProvider


# ---------------------------------------------------------------------------
# ModelProviderFactory tests
# ---------------------------------------------------------------------------


def test_factory_returns_gemini_for_explicit_name() -> None:
    """ModelProviderFactory.get_provider('gemini') must return GeminiAIEngine."""
    provider = ModelProviderFactory.get_provider("gemini")
    assert isinstance(provider, GeminiAIEngine)


def test_factory_returns_amd_rocm_for_explicit_name() -> None:
    """ModelProviderFactory.get_provider('amd_rocm') must return AmdROCmProvider."""
    provider = ModelProviderFactory.get_provider("amd_rocm")
    assert isinstance(provider, AmdROCmProvider)
    assert provider.provider_type == ProviderType.AMD_ROCM


def test_factory_returns_openai_compatible() -> None:
    """ModelProviderFactory.get_provider('openai_compatible') must return AmdROCmProvider."""
    provider = ModelProviderFactory.get_provider("openai_compatible")
    assert isinstance(provider, AmdROCmProvider)
    assert provider.provider_type == ProviderType.OPENAI_COMPATIBLE


def test_factory_defaults_to_gemini_when_env_unset() -> None:
    """When MODEL_PROVIDER env var is absent, factory must default to GeminiAIEngine."""
    env_backup = os.environ.pop("MODEL_PROVIDER", None)
    try:
        provider = ModelProviderFactory.get_provider()
        assert isinstance(provider, GeminiAIEngine)
    finally:
        if env_backup is not None:
            os.environ["MODEL_PROVIDER"] = env_backup


def test_factory_reads_model_provider_env_var() -> None:
    """When MODEL_PROVIDER='amd_rocm' is set, factory must return AmdROCmProvider."""
    with patch.dict(os.environ, {"MODEL_PROVIDER": "amd_rocm"}):
        provider = ModelProviderFactory.get_provider()
        assert isinstance(provider, AmdROCmProvider)


def test_factory_accepts_rocm_alias() -> None:
    """'rocm' and 'amd' are accepted aliases for AMD ROCm provider."""
    for alias in ("rocm", "amd"):
        provider = ModelProviderFactory.get_provider(alias)
        assert isinstance(provider, AmdROCmProvider), f"alias '{alias}' did not resolve to AmdROCmProvider"


def test_factory_accepts_vllm_tgi_aliases() -> None:
    """'vllm' and 'tgi' are accepted aliases for the OpenAI-compatible provider."""
    for alias in ("vllm", "tgi"):
        provider = ModelProviderFactory.get_provider(alias)
        assert isinstance(provider, AmdROCmProvider), f"alias '{alias}' did not resolve to AmdROCmProvider"


# ---------------------------------------------------------------------------
# AmdROCmProvider construction tests
# ---------------------------------------------------------------------------


def test_rocm_provider_stores_configuration() -> None:
    """AmdROCmProvider must expose the endpoint, model name, and provider type."""
    provider = AmdROCmProvider(
        endpoint_url="http://rocm-host:8000/v1",
        model_name="Qwen/Qwen2.5-7B-Instruct",
        provider_type=ProviderType.AMD_ROCM,
    )
    assert provider.endpoint_url == "http://rocm-host:8000/v1"
    assert provider.model_name == "Qwen/Qwen2.5-7B-Instruct"
    assert provider.provider_type == ProviderType.AMD_ROCM


# ---------------------------------------------------------------------------
# AmdROCmProvider.health_check tests  (mocked HTTP)
# ---------------------------------------------------------------------------


def _make_mock_response(status_code: int, json_data: dict) -> MagicMock:
    """Build a minimal mock httpx.Response."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_data
    return mock_resp


def _make_mock_async_client(get_return=None, get_side_effect=None) -> MagicMock:
    """Build a mock httpx.AsyncClient that works as an async context manager."""
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    if get_side_effect is not None:
        mock_client.get = AsyncMock(side_effect=get_side_effect)
    else:
        mock_client.get = AsyncMock(return_value=get_return)
    return mock_client


def test_rocm_health_check_server_down() -> None:
    """health_check must return is_available=False when the endpoint is unreachable."""
    import httpx

    provider = AmdROCmProvider(
        endpoint_url="http://127.0.0.1:19999/v1",
        model_name="Qwen/Qwen2.5-7B-Instruct",
    )

    async def _run() -> ProviderHealthStatus:
        mock_client = _make_mock_async_client(
            get_side_effect=httpx.ConnectError("connection refused")
        )
        with patch("backend.rocm_provider.httpx.AsyncClient", return_value=mock_client):
            return await provider.health_check()

    status = asyncio.run(_run())
    assert isinstance(status, ProviderHealthStatus)
    assert status.is_available is False
    assert status.provider == ProviderType.AMD_ROCM
    assert status.error_message is not None


def test_rocm_health_check_server_up() -> None:
    """health_check must return is_available=True when /v1/models replies 200."""
    provider = AmdROCmProvider(
        endpoint_url="http://rocm-host:8000/v1",
        model_name="Qwen/Qwen2.5-7B-Instruct",
    )

    mock_response = _make_mock_response(
        200,
        {"data": [{"id": "Qwen/Qwen2.5-7B-Instruct"}]},
    )

    async def _run() -> ProviderHealthStatus:
        mock_client = _make_mock_async_client(get_return=mock_response)
        with patch("backend.rocm_provider.httpx.AsyncClient", return_value=mock_client):
            return await provider.health_check()

    status = asyncio.run(_run())
    assert isinstance(status, ProviderHealthStatus)
    assert status.is_available is True
    assert status.error_message is None


# ---------------------------------------------------------------------------
# AmdROCmProvider.benchmark test  (mocked streaming via aiter_lines)
# ---------------------------------------------------------------------------


def _make_sse_lines(tokens: list[str]) -> list[str]:
    """Build SSE data lines mimicking vLLM streaming format (aiter_lines yields str lines)."""
    lines: list[str] = []
    for tok in tokens:
        payload = {
            "choices": [{"delta": {"content": tok}, "finish_reason": None}]
        }
        lines.append(f"data: {json.dumps(payload)}")
    lines.append("data: [DONE]")
    return lines


async def _async_line_iter(lines: list[str]) -> AsyncIterator[str]:
    for line in lines:
        yield line


def test_rocm_provider_benchmark_returns_result() -> None:
    """benchmark() must return an InferenceBenchmarkResult with positive latency."""
    provider = AmdROCmProvider(
        endpoint_url="http://rocm-host:8000/v1",
        model_name="Qwen/Qwen2.5-7B-Instruct",
    )

    tokens = ["Hello", ",", " world", "!"]
    sse_lines = _make_sse_lines(tokens)

    async def _run() -> InferenceBenchmarkResult:
        # Mock the streaming response context manager
        mock_stream_resp = MagicMock()
        mock_stream_resp.status_code = 200
        mock_stream_resp.__aenter__ = AsyncMock(return_value=mock_stream_resp)
        mock_stream_resp.__aexit__ = AsyncMock(return_value=False)
        mock_stream_resp.aiter_lines = MagicMock(return_value=_async_line_iter(sse_lines))

        # Mock the outer AsyncClient context manager
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.stream = MagicMock(return_value=mock_stream_resp)

        with patch("backend.rocm_provider.httpx.AsyncClient", return_value=mock_client):
            return await provider.benchmark()

    result = asyncio.run(_run())
    assert isinstance(result, InferenceBenchmarkResult)
    assert result.provider == ProviderType.AMD_ROCM
    assert result.total_latency_ms >= 0
    assert result.completion_tokens >= 0
    assert result.success is True
