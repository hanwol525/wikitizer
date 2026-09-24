"""The thin model-client adapter the FC adjudicator calls.

It deliberately REUSES the pipeline's existing OpenAI-compatible ``.env`` path
(``LLM_OPENAI_BASE_URL`` + ``LLM_OPENAI_API_KEY``, the OpenRouter gateway) and the already-
pinned ``openai`` SDK -- it does NOT stand up a new endpoint or auth scheme. A dedicated
adapter (rather than reusing ``agents.llm_client.AnthropicCompatClient``) is unavoidable
only because the FC judge needs two things that surface can't pass per call: a
``response_format`` (forced JSON) and Qwen3's thinking-off ``extra_body``.

The client is INJECTABLE -- the adjudicator takes any object with a ``complete(system,
user) -> str`` method, so the offline tests pass a ``FakeModelClient`` and never touch the
network or need a key (the same dependency-injection shape ``BaseAgent`` uses).

Default judge: Qwen3-8B, thinking OFF, non-greedy (Qwen3 discourages greedy decoding),
``response_format={"type": "json_object"}``. FIX #6: not every OpenRouter provider honors
``response_format`` -- if it's REJECTED (HTTP 400) we retry once without it (the tolerant
parser upstream then handles a plain reply); if it's merely IGNORED the tolerant parser
already covers it.
"""

import logging
import os
import re
from typing import Optional

from evals.fc import (
    DEFAULT_FC_MODEL,
    DEFAULT_FC_PROVIDER,
    DEFAULT_TEMPERATURE,
    ENV_MODEL,
    ENV_OPENAI_API_KEY,
    ENV_OPENAI_BASE_URL,
    ENV_TEMPERATURE,
)

logger = logging.getLogger("evals.fc.model_client")

# The extra_body that turns a reasoning model's thinking OFF -- sent BOTH the OpenRouter
# unified param and the provider-native chat_template toggle, for the best odds one is
# honored (mirrors agents.llm_client._reasoning_extra_body("off")).
_THINKING_OFF = {"reasoning": {"enabled": False},
                 "chat_template_kwargs": {"enable_thinking": False}}

# A leading "<think>...</think>" block Qwen3 can still emit even with thinking nominally off.
_THINK_BLOCK = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


def strip_think(text: str) -> str:
    """Defensively drop a leading ``<think>...</think>`` block before JSON parsing."""
    return _THINK_BLOCK.sub("", text or "")


class OpenAICompatModelClient:
    """An OpenAI-compatible chat client specialized for the FC judge. ``complete`` sends one
    system+user turn and returns the assistant text (think-block stripped)."""

    def __init__(self, model: str, temperature: float, thinking: bool = False,
                 provider: str = DEFAULT_FC_PROVIDER, base_url: Optional[str] = None,
                 api_key: Optional[str] = None, timeout: float = 60.0):
        import openai  # lazy: an Anthropic-only user (or an offline test) never imports it
        self._openai = openai
        self.model = model
        self.temperature = temperature
        self.thinking = thinking
        self.provider = provider
        self.base_url = base_url
        self._client = openai.OpenAI(base_url=base_url, api_key=api_key,
                                     max_retries=3, timeout=timeout)

    def _create(self, messages, use_response_format: bool):
        kwargs = dict(model=self.model, temperature=self.temperature, messages=messages)
        if not self.thinking:
            kwargs["extra_body"] = _THINKING_OFF
        if use_response_format:
            kwargs["response_format"] = {"type": "json_object"}
        return self._client.chat.completions.create(**kwargs)

    def complete(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        try:
            resp = self._create(messages, use_response_format=True)
        except self._openai.BadRequestError:
            # FIX #6: the provider rejected response_format -> retry once without it.
            logger.warning("[REVIEW] provider rejected response_format; retrying without it")
            resp = self._create(messages, use_response_format=False)
        choices = getattr(resp, "choices", None) or []
        if not choices:
            logger.warning("[REVIEW] model response had no choices; returning empty string")
            return ""
        content = getattr(choices[0].message, "content", None) or ""
        return strip_think(content)


def build_model_client(env=None) -> Optional[OpenAICompatModelClient]:
    """Build the FC judge client from the environment, or return None when the OpenAI-compat
    credentials aren't set (which drives the runner's SKIPPED path -- same spirit as the
    integration tests' ``skipif`` on a missing key). The model/temperature are read from
    ``WIKITIZER_FC_MODEL`` / ``WIKITIZER_FC_TEMPERATURE`` so the judge is swappable via .env."""
    env = os.environ if env is None else env
    base_url = env.get(ENV_OPENAI_BASE_URL)
    api_key = env.get(ENV_OPENAI_API_KEY)
    if not base_url or not api_key:
        return None
    model = env.get(ENV_MODEL) or DEFAULT_FC_MODEL
    raw_temp = env.get(ENV_TEMPERATURE)
    try:
        temperature = float(raw_temp) if raw_temp else DEFAULT_TEMPERATURE
    except ValueError:
        logger.warning("%s=%r is not a number; using %.2f", ENV_TEMPERATURE, raw_temp,
                       DEFAULT_TEMPERATURE)
        temperature = DEFAULT_TEMPERATURE
    return OpenAICompatModelClient(model=model, temperature=temperature, thinking=False,
                                   base_url=base_url, api_key=api_key)
