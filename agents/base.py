"""Phase 3.1: the shared base class every LLM agent inherits from.

All five Phase 3 agents (noise filter + the locations/characters/history/other
extractors) need the exact same boilerplate: make one low-temperature Claude
call, pull the text out of the response, and parse JSON out of it -- coping with
the fact that Claude sometimes wraps that JSON in a markdown code fence even when
told not to. Writing it once here keeps each specialized agent down to just a
prompt and a thin ``parse`` method.

Design notes / the "why" behind the shapes here:

  * The Anthropic ``client`` is injected (defaulting to a real one) so tests can
    pass a fake and never touch the network or need an API key. We do NOT read
    the API key here: the SDK picks it up from the ``ANTHROPIC_API_KEY`` env var.
  * Two *different* failure modes need two *different* retry strategies, so they
    live at two different layers:
      - HTTP-level failures (timeouts, 429, 5xx) are transient and identical to
        retry, so we let the SDK's own ``max_retries`` handle them -- no manual
        loop in :meth:`BaseAgent.call_claude`.
      - A *successful* call that returns non-JSON text is a different beast: the
        request worked, so the SDK won't retry it, but a re-ask might get valid
        JSON. That retry lives in :meth:`BaseAgent.call_claude_json`.
  * No streaming / tool-use / async: the orchestrator will fan agents out over
    threads later, and the synchronous client is thread-safe, so none of that
    belongs in this foundation.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional, Union

from json_repair import repair_json

from agents.llm_client import DEFAULT_BIG_MODEL, build_llm_client

logger = logging.getLogger(__name__)


def strip_code_fences(text: str) -> str:
    """Remove a wrapping markdown code fence, if one is present.

    Claude occasionally returns its JSON wrapped in a fence -- ``` ```json ```
    ... ``` ``` ``` or a bare ``` ``` ``` ... ``` ``` ``` -- even when asked for
    raw JSON. This peels off a leading fence line (three backticks, optionally
    followed by a language hint like ``json``) and the matching trailing fence
    line, returning the inner content. Anything that is not so wrapped is
    returned unchanged.

    Pure and side-effect-free, so it can be unit-tested on its own.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text

    lines = stripped.splitlines()
    # Need both an opening fence (line 0) and a *closing* fence on its own line.
    # If the closing fence isn't the last line, this isn't a clean wrapper (e.g.
    # a fenced block followed by prose) -- leave the text untouched rather than
    # mangle it.
    if len(lines) < 2 or lines[-1].strip() != "```":
        return text

    inner = "\n".join(lines[1:-1])
    return inner.strip()


def loads_tolerant(text: str) -> Union[dict, list]:
    """Parse JSON out of an LLM reply, tolerating the two malformations Claude
    produces that plain ``json.loads`` chokes on:

      * a conversational preamble before the JSON ("Looking at these messages...")
        -- the characters extractor especially likes to reason out loud; and
      * **unescaped double-quotes inside a string value** -- when Claude copies a
        verbatim quote it straightens the source's curly quotes to ``"`` and forgets
        to escape them, so ``"quote": "The other "countries" are..."`` is invalid JSON.

    Either one used to fail all three retries and drop the ENTIRE 20-message batch
    (``_extract_batch`` returns ``[]``), which is where most of the missing lore went.

    Strategy: try strict ``json.loads(strip_code_fences(...))`` first -- the happy
    path is unchanged and stays exact -- and only on failure fall back to
    ``json_repair`` (which both peels a preamble and re-escapes stray quotes).
    Raises ``json.JSONDecodeError`` if even the repair can't produce a dict/list, so
    every caller's existing ``except JSONDecodeError`` retry/contain logic is
    untouched.

    SAFETY: repair is deliberately aggressive, but it can only ever *recover* a
    batch, never fabricate lore, because everything downstream re-validates -- the
    extractor's ``isinstance(list)`` guard, ``_build_entry``'s per-field checks, and
    above all the verbatim-quote check, which drops any quote that isn't a real
    substring of its cited message. A mis-repaired quote is simply dropped, exactly
    as today.
    """
    stripped = strip_code_fences(text)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        # return_objects=True hands back the parsed Python value directly; a reply
        # with no recoverable JSON comes back as "" (or similar), caught below.
        parsed = repair_json(stripped, return_objects=True)
    if not isinstance(parsed, (dict, list)):
        raise json.JSONDecodeError("JSON must be an object or array", stripped or text, 0)
    return parsed


class ClaudeJSONError(ValueError):
    """Raised when Claude never returns parseable JSON within the retry budget.

    Carries the last raw response text on ``.raw`` (and in the message) so a
    failed extraction is debuggable without re-running the call.
    """

    def __init__(self, message: str, raw: Optional[str]):
        super().__init__(message)
        self.raw = raw


class BaseAgent:
    """Shared Claude plumbing for the Phase 3 LLM agents.

    Subclasses supply a system prompt and a thin parse step; everything about
    talking to Claude lives here.
    """

    def __init__(
        self,
        client=None,
        model: str = DEFAULT_BIG_MODEL,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        max_json_retries: int = 3,
        cache: Optional[bool] = None,
        cache_dir: str = ".llm_cache",
    ):
        # Injected client wins; otherwise build the default (Anthropic) client. The
        # orchestrator normally injects a per-role client (Anthropic or an
        # OpenAI-compatible wrapper) via build_llm_client; this fallback covers direct
        # instantiation. ``max_retries=3`` (set inside build_llm_client) lets the SDK
        # transparently retry transient HTTP failures. Keys come from the environment
        # via the SDK -- never read or stored here.
        self.client = client if client is not None else build_llm_client("anthropic")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_json_retries = max_json_retries
        if self.max_json_retries < 1:
            raise ValueError("max_json_retries must be >= 1")
        # Dev-only on-disk response cache (see _cache_*). Off unless explicitly
        # turned on, so the real pipeline never silently serves stale answers.
        # The env var lets you flip it on for a whole prompt-tuning session
        # without editing call sites.
        if cache is None:
            cache = os.environ.get("WIKITIZER_LLM_CACHE", "").strip().lower() in (
                "1", "true", "yes", "on",
            )
        self._cache_enabled = bool(cache)
        self._cache_dir = Path(cache_dir)

    def call_claude(self, system_prompt: str, user_message: str) -> str:
        """One Messages API call; returns the concatenated text of the response.

        A single user-role message, the instance's model/temperature/max_tokens,
        and ``system_prompt`` as the system prompt. We deliberately do NOT retry
        here -- the SDK's ``max_retries`` already covers HTTP-level failures, and
        a content-level (JSON) retry belongs in :meth:`call_claude_json`.

        Deliberately cache-free: the dev cache lives in :meth:`call_claude_json`
        and is written only after a response parses, so it can't poison the
        retry loop (see that method for the reasoning).

        **Anthropic prompt caching:** the system prompt is passed as a single
        content block marked ``cache_control: {"type": "ephemeral"}``. Every agent's
        system prompt is byte-stable across a run (module constants / class attrs;
        the characters roster is injected once, ``sorted()`` for stability), and
        these calls carry no tools and a per-call *user* message, so the system
        prompt is exactly the cacheable prefix. The many batch calls in a run then
        bill it once (+write) and read it back at ~0.1x. NB: a prefix under the
        model's minimum (Sonnet 4.6 = 2048 tokens, Haiku 4.5 = 4096) silently won't
        cache -- no error, it just no-ops -- so the short prompts are free to mark
        and the long ones (reconciler, characters, timeline) are where it pays off.
        Default (5-min) TTL is right here: a run's batches all fire within minutes.

        The signature stays ``system_prompt: str`` on purpose -- wrapping happens
        only at the ``create`` call, so the dev cache key (which hashes the string)
        is untouched.
        """
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=[{"type": "text", "text": system_prompt,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user_message}],
        )
        # A response can carry several content blocks (text, thinking, ...);
        # concatenate the text of every text-type block and ignore the rest.
        return "".join(block.text for block in response.content if block.type == "text")

    def call_claude_json(self, system_prompt: str, user_message: str) -> Union[dict, list]:
        """Call Claude and return parsed JSON (a ``dict`` or ``list``).

        Strips any code fence, then ``json.loads``. A ``JSONDecodeError`` means
        the call succeeded but the *content* wasn't valid JSON -- which the SDK
        won't have retried -- so we re-ask up to ``max_json_retries`` times.
        After that, raise :class:`ClaudeJSONError` carrying the last raw text.

        The dev cache lives here (not in :meth:`call_claude`) and is written
        only *after* a response parses. Caching pre-validation text would be a
        trap: the retry below re-calls ``call_claude`` with the same key, so a
        cached malformed response would be replayed on every attempt and turn
        the content-retry into a guaranteed failure. Caching only parsed-good
        text means a cache hit is always usable and a miss lets the retry
        genuinely re-ask Claude.
        """
        if self._cache_enabled:
            cached = self._cache_get(system_prompt, user_message)
            if cached is not None:
                # Only parsed-good text is ever stored, but the on-disk cache file
                # can still be corrupted/truncated; treat that as a cache miss.
                try:
                    return loads_tolerant(cached)
                except json.JSONDecodeError:
                    logger.warning("Ignoring corrupted LLM cache entry; re-calling Claude")
        last_raw: Optional[str] = None
        for attempt in range(1, self.max_json_retries + 1):
            last_raw = self.call_claude(system_prompt, user_message)
            try:
                parsed = loads_tolerant(last_raw)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Claude returned invalid JSON (attempt %d/%d): %s",
                    attempt, self.max_json_retries, exc,
                )
                continue
            if self._cache_enabled:
                self._cache_put(system_prompt, user_message, last_raw)
            return parsed

        raise ClaudeJSONError(
            "Claude did not return valid JSON after "
            f"{self.max_json_retries} attempt(s). Last raw response was:\n{last_raw}",
            raw=last_raw,
        )

    # --- dev-only response cache ------------------------------------------------
    # A tiny on-disk cache keyed on (model, system_prompt, user_message) so that
    # re-running the pipeline during prompt-tuning doesn't re-bill identical
    # calls. Intentionally minimal: plain files, no eviction, no expiry.

    def _cache_key(self, system_prompt: str, user_message: str) -> str:
        h = hashlib.sha256()
        # NUL separators so different field boundaries can't collide.
        for part in (self.model, str(self.temperature), str(self.max_tokens), system_prompt, user_message):
            h.update(part.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    def _cache_get(self, system_prompt: str, user_message: str) -> Optional[str]:
        path = self._cache_dir / (self._cache_key(system_prompt, user_message) + ".txt")
        if path.exists():
            return path.read_text(encoding="utf-8")
        return None

    def _cache_put(self, system_prompt: str, user_message: str, text: str) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_dir / (self._cache_key(system_prompt, user_message) + ".txt")
        path.write_text(text, encoding="utf-8")
