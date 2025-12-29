"""Local LLM client for OpenAI-compatible APIs.

Supports Ollama, vLLM, LM Studio, llama.cpp, and any other
OpenAI-compatible local inference server.

Example usage:
    1. Run Ollama with Qwen3
    2. Set env vars:
        LOCAL_LLM_BASE_URL=http://localhost:11434/v1
        LOCAL_LLM_MODEL=qwen3:14b
    3. Use the client
"""

import os
from typing import Optional, List, Dict, Any

from openai import OpenAI


# Model-specific settings for optimal performance
MODEL_CONFIGS = {
    "gemma": {
        "temperature": 0.5,
        "top_p": 0.9,
        "repeat_penalty": 1.05,
        "reasoning_model": False,
    },
    "qwen2.5": {
        "temperature": 0.4,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
        "reasoning_model": False,
    },
    "qwen3": {
        "temperature": 0.6,
        "top_p": 0.95,
        "repeat_penalty": 1.0,
        "reasoning_model": True,
    },
    "nemotron": {
        "temperature": 0.7,
        "top_p": 0.95,
        "repeat_penalty": 1.0,
        "reasoning_model": True,
    },
    "deepseek-r1": {
        "temperature": 0.6,
        "top_p": 0.95,
        "repeat_penalty": 1.0,
        "reasoning_model": True,
    },
    "llama": {
        "temperature": 0.5,
        "top_p": 0.9,
        "repeat_penalty": 1.05,
        "reasoning_model": False,
    },
    "default": {
        "temperature": 0.5,
        "top_p": 0.9,
        "repeat_penalty": 1.0,
        "reasoning_model": False,
    },
}


def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get optimal settings for a specific model."""
    model_lower = model_name.lower()
    for prefix, config in MODEL_CONFIGS.items():
        if prefix in model_lower:
            return config
    return MODEL_CONFIGS["default"]


class LocalLLMClient:
    """Client for OpenAI-compatible local LLM servers.

    Works with:
    - Ollama (http://localhost:11434/v1)
    - llama.cpp (http://localhost:8000/v1)
    - vLLM (http://localhost:8000/v1)
    - LM Studio (http://localhost:1234/v1)
    - Any OpenAI-compatible API

    Automatically detects model type and applies optimal settings.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = "not-needed",
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ):
        """Initialize the local LLM client.

        Args:
            base_url: Base URL for the API. Defaults to LOCAL_LLM_BASE_URL env var.
            model: Model name to use. Defaults to LOCAL_LLM_MODEL env var.
            api_key: API key if required. Defaults to "not-needed".
            max_tokens: Maximum tokens in response (default 4096).
            temperature: Override auto-configured temperature.
            top_p: Override auto-configured top_p.
        """
        self.base_url = base_url or os.environ.get("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1")
        self.model = model or os.environ.get("LOCAL_LLM_MODEL", "qwen3:14b")
        self.api_key = api_key or os.environ.get("LOCAL_LLM_API_KEY", "not-needed")
        self.max_tokens = max_tokens
        self._client: Optional[OpenAI] = None

        config = get_model_config(self.model)
        self.temperature = temperature if temperature is not None else config["temperature"]
        self.top_p = top_p if top_p is not None else config["top_p"]
        self.repeat_penalty = config.get("repeat_penalty", 1.0)
        self._is_reasoning_model = config["reasoning_model"]

    def _ensure_client(self) -> OpenAI:
        """Lazily initialize the OpenAI client."""
        if self._client is None:
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=300.0)
        return self._client

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate a response from the local LLM.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.
            max_tokens: Override default max_tokens.
            temperature: Override default temperature.

        Returns:
            The generated text response.
        """
        try:
            client = self._ensure_client()

            messages: List[Dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            temp = temperature if temperature is not None else self.temperature

            request_params = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens or self.max_tokens,
                "temperature": temp,
                "top_p": self.top_p,
            }

            # Ollama-specific optimizations
            request_params["extra_body"] = {
                "num_ctx": 8192,
                "num_batch": 512,
            }

            if self.repeat_penalty != 1.0:
                request_params["extra_body"]["repeat_penalty"] = self.repeat_penalty

            response = client.chat.completions.create(**request_params)

            message = response.choices[0].message
            content = message.content

            if self._is_reasoning_model:
                reasoning = getattr(message, "reasoning_content", None)
                if not content and reasoning:
                    content = reasoning

            if content is None:
                return ""

            return content
        except Exception as e:
            print(f"LLM generation error: {str(e)}")
            raise e

    def test_connection(self) -> Dict[str, Any]:
        """Test the connection to the local LLM server.

        Returns:
            Dictionary with connection status and available models.
        """
        try:
            client = self._ensure_client()
            models = client.models.list()
            model_names = [m.id for m in models.data]

            return {
                "status": "connected",
                "base_url": self.base_url,
                "available_models": model_names,
                "configured_model": self.model,
            }
        except Exception as e:
            return {
                "status": "error",
                "base_url": self.base_url,
                "error": str(e),
            }
