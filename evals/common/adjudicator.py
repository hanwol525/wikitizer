"""The self-consistency voting engine shared by every eval's LLM adjudicator.

An eval's adjudicator subclasses ``BaseAdjudicator`` and adds its own tiny single-purpose
prompts + candidate payloads + label vocab; the voting machinery lives here. Each check
judges its whole (small) candidate set ``votes`` times and takes a per-candidate majority;
the verdicts fold into ONE ``Item`` (``engine="model"``).

FIX #5 (vote hygiene): a vote that returns the wrong number of verdicts is discarded as an
abstain, so misaligned indices can never silently corrupt the per-candidate majority; if
EVERY vote is malformed the check fails CLOSED with a ``[REVIEW]`` note (never a false pass);
``votes`` is odd so a per-candidate majority rarely ties, and a tie is resolved toward the
"needs a human" (bad) label and flagged ``[REVIEW]``.
"""

import json
import logging
from collections import Counter
from typing import List, Optional, Tuple

from agents.base import loads_tolerant

from evals.common.models import Engine, Evidence, Item, Status

logger = logging.getLogger("evals.common.adjudicator")


def _safe_parse(raw: str):
    """Parse a model reply into a dict, tolerating fences/preambles; None on failure."""
    try:
        parsed = loads_tolerant(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


class BaseAdjudicator:
    """Runs voted model checks against an injected ``model_client`` (anything with
    ``complete(system, user) -> str``). Subclasses add the eval-specific ``judge_*`` methods;
    they get ``_vote`` / ``_judge`` for free and override ``describe`` to attach criterion
    prose to the emitted items."""

    def __init__(self, model_client, votes: int = 3):
        self.model_client = model_client
        self.votes = max(1, votes)

    # --- item construction -------------------------------------------------- #

    def describe(self, cid: str) -> Optional[str]:
        """The criterion prose for an item id. Default: none; a subclass overrides this to
        inject its own id->prose map so the folded ``Item`` carries a description."""
        return None

    def _model_item(self, cid: str, status: Status, evidence: Evidence) -> Item:
        return Item(id=cid, description=self.describe(cid), engine=Engine.MODEL,
                    status=status, evidence=evidence)

    # --- voting ------------------------------------------------------------- #

    def _vote(self, system: str, user: str, n: int, allowed: set,
              bad_label: str) -> Tuple[Optional[List[str]], List[int]]:
        """Return (per-candidate majority labels, disagreement indices). ``labels`` is None
        when EVERY vote was malformed (caller fails closed). FIX #5 lives here."""
        valid_votes: List[List[str]] = []
        for _ in range(self.votes):
            raw = self.model_client.complete(system, user)
            parsed = _safe_parse(raw)
            verdicts = parsed.get("verdicts") if isinstance(parsed, dict) else None
            if not isinstance(verdicts, list) or len(verdicts) != n:
                logger.debug("discarding a malformed vote (verdicts=%r)", verdicts)
                continue
            valid_votes.append([str(v).strip() for v in verdicts])
        if not valid_votes:
            return None, list(range(n))

        labels: List[str] = []
        disagreements: List[int] = []
        for i in range(n):
            col = [v[i] for v in valid_votes if v[i] in allowed]
            if not col:
                labels.append(bad_label)
                disagreements.append(i)
                continue
            counts = Counter(col).most_common()
            if len(counts) > 1 and counts[0][1] == counts[1][1]:
                labels.append(bad_label)          # tie -> surface for review
                disagreements.append(i)
            else:
                labels.append(counts[0][0])
                if len(set(col)) > 1:
                    disagreements.append(i)       # a non-unanimous but decided split
        return labels, disagreements

    def _judge(self, cid: str, system: str, user: str, candidates: List[dict],
               allowed: set, bad_label: str, offender_label: str) -> Item:
        """Fold a voted judgment over ``candidates`` into ONE ``Item``: fail-closed when the
        model returned nothing parseable, pass when no candidate is an offender, else fail
        with the offenders in the evidence."""
        labels, disagreements = self._vote(system, user, len(candidates), allowed, bad_label)
        lines = [c.get("lineno") for c in candidates if c.get("lineno")]
        if labels is None:
            return self._model_item(cid, Status.FAIL, Evidence(
                lines=lines, detail="[REVIEW] model returned no parseable verdicts; failing closed"))
        offenders = [c for c, lab in zip(candidates, labels) if lab == offender_label]
        note = ""
        if disagreements:
            note = f" ({len(disagreements)} candidate(s) had split votes -- [REVIEW])"
            logger.warning("[REVIEW] %s: %d candidate(s) had split votes across %d votes",
                           cid, len(disagreements), self.votes)
        if not offenders:
            return self._model_item(cid, Status.PASS, Evidence(
                lines=[], detail=f"{len(candidates)} candidate(s) judged, none flagged" + note))
        off_lines = [c.get("lineno") for c in offenders if c.get("lineno")]
        names = [c.get("label") or c.get("name") or c.get("figure") or "?" for c in offenders]
        return self._model_item(cid, Status.FAIL, Evidence(
            lines=off_lines,
            detail=f"{len(offenders)} of {len(candidates)} candidate(s) flagged" + note,
            found="; ".join(str(x) for x in names[:8])))
