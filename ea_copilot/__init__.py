"""
Executive Assistant for Microsoft 365 Copilot

Automated AI-powered executive assistant that runs on schedule,
not chat. Configure once, run continuously.
"""

__version__ = "0.1.0"

from .scheduler import EAScheduler
from .ms_graph import MSGraphClient
from .llm_client import LLMClient

__all__ = ["EAScheduler", "MSGraphClient", "LLMClient"]
