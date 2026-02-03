"""
LLM Client for Executive Assistant
Supports Azure OpenAI (Microsoft Copilot backend) and OpenAI
"""

import os
from typing import Optional
import requests


class LLMClient:
    """Client for LLM completions."""

    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        deployment: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize LLM client.

        Supports:
        - Azure OpenAI (default, for Microsoft 365 Copilot integration)
        - OpenAI
        - Local/Custom endpoints

        Environment variables:
        - LLM_PROVIDER: "azure", "openai", or "custom"
        - AZURE_OPENAI_KEY: API key for Azure OpenAI
        - AZURE_OPENAI_ENDPOINT: Azure OpenAI endpoint URL
        - AZURE_OPENAI_DEPLOYMENT: Deployment name
        - OPENAI_API_KEY: API key for OpenAI
        - LLM_MODEL: Model name (for OpenAI)
        """
        self.provider = provider or os.getenv("LLM_PROVIDER", "azure")
        self.api_key = api_key
        self.endpoint = endpoint
        self.deployment = deployment
        self.model = model

        self._setup_provider()

    def _setup_provider(self):
        """Set up provider-specific configuration."""
        if self.provider == "azure":
            self.api_key = self.api_key or os.getenv("AZURE_OPENAI_KEY")
            self.endpoint = self.endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
            self.deployment = self.deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
            self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

        elif self.provider == "openai":
            self.api_key = self.api_key or os.getenv("OPENAI_API_KEY")
            self.endpoint = "https://api.openai.com/v1"
            self.model = self.model or os.getenv("LLM_MODEL", "gpt-4")

        elif self.provider == "custom":
            self.api_key = self.api_key or os.getenv("LLM_API_KEY")
            self.endpoint = self.endpoint or os.getenv("LLM_ENDPOINT")
            self.model = self.model or os.getenv("LLM_MODEL")

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.7
    ) -> str:
        """
        Get completion from LLM.

        Args:
            prompt: User prompt
            system: System prompt (optional)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature

        Returns:
            Generated text response
        """
        messages = []

        if system:
            messages.append({"role": "system", "content": system})

        messages.append({"role": "user", "content": prompt})

        if self.provider == "azure":
            return self._azure_complete(messages, max_tokens, temperature)
        elif self.provider == "openai":
            return self._openai_complete(messages, max_tokens, temperature)
        elif self.provider == "custom":
            return self._custom_complete(messages, max_tokens, temperature)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _azure_complete(self, messages: list, max_tokens: int, temperature: float) -> str:
        """Complete using Azure OpenAI."""
        url = f"{self.endpoint}/openai/deployments/{self.deployment}/chat/completions?api-version={self.api_version}"

        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }

        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()

        result = response.json()
        return result["choices"][0]["message"]["content"]

    def _openai_complete(self, messages: list, max_tokens: int, temperature: float) -> str:
        """Complete using OpenAI."""
        url = f"{self.endpoint}/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()

        result = response.json()
        return result["choices"][0]["message"]["content"]

    def _custom_complete(self, messages: list, max_tokens: int, temperature: float) -> str:
        """Complete using custom endpoint (OpenAI-compatible)."""
        url = f"{self.endpoint}/chat/completions"

        headers = {
            "Content-Type": "application/json"
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()

        result = response.json()
        return result["choices"][0]["message"]["content"]


class MockLLMClient(LLMClient):
    """Mock LLM client for testing."""

    def __init__(self):
        self.provider = "mock"

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.7
    ) -> str:
        """Return mock response."""
        return f"""
## Daily Briefing

### Top Priorities
1. Review Q4 budget proposal
2. Prepare for board meeting
3. Follow up with sales team

### Today's Meetings
- 9:00 AM: Team standup
- 11:00 AM: Product review
- 2:00 PM: Client call

### Emails Requiring Attention
- CFO request for budget review (High Priority)
- HR policy update (FYI)

### Suggested Focus Time
Block 3:00-5:00 PM for deep work on strategic planning.
"""
