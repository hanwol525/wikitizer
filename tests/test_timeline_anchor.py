"""Phase B: the timeline reference-year anchor -- resolving present-relative dates.

A "200 years ago" event used to fall to Undated Events because nothing anchored it
to a year. Now:
  * a reference year is threaded in (config/CLI override) and/or auto-detected by
    Call 1 from the lore;
  * Call 1 resolves an offset ("200 years ago", ref 1424 -> parts [1224]) and flags
    the event `anchor_relative`;
  * `_sanity_guard_parts` SKIPS its digit-match check for such events (1224 shares no
    digit with "200"), so they land on the dated spine instead of being dropped.

Offline: a FakeClient returns canned Call-1/Call-2 JSON; no network, no key. Self-
contained fakes (mirrors test_reconciler.py's harness) per the repo convention.
"""

import json

from agents.reconciler import Reconciler, _sanity_guard_parts, UNDATED
from models.lore import HistoryEvent, Scope
from models.reconcile import DateDecision, DatedEvent
from main import parse_args


# --- self-contained fake client ------------------------------------------- #
class _Block:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _Resp:
    def __init__(self, blocks):
        self.content = blocks


class _Messages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        spec = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        return _Resp([_Block(spec)] if isinstance(spec, str) else spec)


class FakeClient:
    def __init__(self, responses):
        self.messages = _Messages(responses)

    @property
    def call_count(self):
        return len(self.messages.calls)


def hev(name, date_text=None, description="An event happened."):
    return HistoryEvent(name=name, description=description, scope=Scope.WORLD,
                        date_text=date_text)


def _date_json(*entries, reference_year=None, reference_system=None):
    """Each entry is (index, system, parts) or (index, system, parts, anchor_relative)."""
    dated = []
    for e in entries:
        idx, sys, parts = e[0], e[1], e[2]
        ar = e[3] if len(e) > 3 else False
        dated.append({"index": idx, "system": sys, "parts": parts, "anchor_relative": ar})
    return json.dumps({"dated": dated, "reference_year": reference_year,
                       "reference_system": reference_system})


def _place_json(*pairs):
    return json.dumps({"placements": [{"gap": g, "events": e} for g, e in pairs]})


def make_reconciler(responses):
    return Reconciler(client=FakeClient(responses))


# ======================================================================= #
# B4: the _sanity_guard_parts relaxation
# ======================================================================= #
def test_guard_skips_digit_check_for_anchor_relative():
    # 1224 shares NO digit with "200 years ago" -- but it's a resolved offset, so the
    # guard must accept it.
    assert _sanity_guard_parts([1224], "200 years ago", anchor_relative=True) is True


def test_guard_still_rejects_digit_mismatch_when_not_anchor_relative():
    # Same numbers, but NOT flagged relative -> the digit-match check applies and fails
    # (this is the existing behavior, preserved by the defaulted flag).
    assert _sanity_guard_parts([1224], "200 years ago") is False
    assert _sanity_guard_parts([1224], "200 years ago", anchor_relative=False) is False


def test_guard_absolute_date_behavior_unchanged():
    assert _sanity_guard_parts([342], "342 AR") is True          # shares a digit
    assert _sanity_guard_parts([343], "342 AR") is False         # unmoored
    assert _sanity_guard_parts([3], "the Third Age") is True     # word date -> trusted


# ======================================================================= #
# B1/B3: the config reference year reaches Call 1's user message
# ======================================================================= #
def test_config_current_year_is_stated_in_the_call1_message():
    events = [hev("Plague", date_text="200 years ago")]
    rec = make_reconciler([_date_json((0, "AR years", [1224], True), reference_year=1424),
                           _place_json()])
    rec.order_history(events, current_year=1424)
    call1 = rec.client.messages.calls[0]
    user_msg = call1["messages"][0]["content"]
    assert "from config: 1424" in user_msg
    # the description rides along so the model can auto-detect / gauge offsets
    assert "desc=" in user_msg


def test_no_config_year_omits_the_config_line():
    events = [hev("Plague", date_text="200 years ago")]
    rec = make_reconciler([_date_json((0, "AR years", [1224], True), reference_year=1424),
                           _place_json()])
    rec.order_history(events)   # no current_year
    user_msg = rec.client.messages.calls[0]["messages"][0]["content"]
    assert "from config:" not in user_msg


# ======================================================================= #
# B3 end-to-end: an offset event lands DATED, not undated
# ======================================================================= #
def test_offset_event_resolves_onto_the_dated_spine():
    # "200 years ago" -> parts [1224], anchor_relative -> guard passes -> dated spine.
    events = [hev("Plague", date_text="200 years ago")]
    rec = make_reconciler([_date_json((0, "AR years", [1224], True), reference_year=1424)])
    out = rec.order_history(events, current_year=1424)
    assert out[0].calendar_system == "AR years"       # dated, NOT None (undated)
    assert out[0].chronological_position == 0


def test_offset_and_absolute_events_order_together_on_one_spine():
    # Absolute 1300 + offset "200 years ago"(=1224) both dated -> 1224 sorts first.
    events = [hev("Founding", date_text="1300"),
              hev("Plague", date_text="200 years ago")]
    rec = make_reconciler([
        _date_json((0, "AR years", [1300], False),
                   (1, "AR years", [1224], True), reference_year=1424),
    ])
    out = rec.order_history(events, current_year=1424)
    by_name = {e.name: e for e in out}
    assert by_name["Plague"].calendar_system == "AR years"
    assert by_name["Founding"].calendar_system == "AR years"
    # 1224 < 1300, so the Plague is chronologically first.
    assert by_name["Plague"].chronological_position < by_name["Founding"].chronological_position


def test_unresolved_offset_without_anchor_falls_to_undated():
    # Model returns no dated entry (couldn't find a reference year) -> Call 2 places
    # the offset event on the (undated) timeline. Backward-compatible behavior.
    events = [hev("Plague", date_text="200 years ago")]
    rec = make_reconciler([_date_json(reference_year=None),               # Call 1: nothing dated
                           _place_json((f"{UNDATED}#0", [0]))])           # Call 2: undated
    out = rec.order_history(events)   # no anchor available
    assert out[0].calendar_system is None
    assert out[0].chronological_position == 0


# ======================================================================= #
# B2: contract field defaults
# ======================================================================= #
def test_dated_event_anchor_relative_defaults_false():
    d = DatedEvent(index=0, system="AR", parts=[1])
    assert d.anchor_relative is False


def test_date_decision_reference_fields_default_none():
    dec = DateDecision()
    assert dec.reference_year is None and dec.reference_system is None


# ======================================================================= #
# B1: the --current-year CLI flag
# ======================================================================= #
def test_cli_parses_current_year():
    args = parse_args(["--files", "a.txt", "--current-year", "1424"])
    assert args.current_year == 1424


def test_cli_current_year_defaults_none():
    args = parse_args(["--files", "a.txt"])
    assert args.current_year is None
