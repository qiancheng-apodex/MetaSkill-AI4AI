"""Direct OpenAI Responses and Anthropic Messages adapters.

Both adapters expose the small Chat-Completions-shaped contract consumed by the
fixed Target runner. Credentials are read only from official environment names.
"""

from .official import (
    AuthenticationFailure, ChatCompletionError, ClaudeProvider,
    InfrastructureFailure, ModelContentFailure, OpenAIProvider,
    RequestCompatibilityFailure, provider_for,
)

__all__ = [
    "AuthenticationFailure", "ChatCompletionError", "ClaudeProvider",
    "InfrastructureFailure", "ModelContentFailure", "OpenAIProvider",
    "RequestCompatibilityFailure", "provider_for",
]
