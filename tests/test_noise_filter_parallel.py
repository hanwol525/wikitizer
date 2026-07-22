"""Parallel-classify path for NoiseFilterAgent (agents/noise_filter.py).

Offline. The default ``max_workers=1`` keeps ``classify`` strictly sequential
(that path is covered by test_noise_filter.py); here we exercise ``max_workers>1``
and prove the fan-out returns the SAME input-ordered labels as the sequential path
-- concurrency is a wall-clock win only, never a behaviour change.

The fake client is **content-addressed**: it labels each message from that
message's own content, not from call order. So its answers are deterministic no
matter what order the threads happen to finish in -- which is exactly what lets us
assert parallel == sequential without any sleeps or flakiness.
"""

import json
import threading
from datetime import datetime

import pytest

from agents.noise_filter import NoiseFilterAgent
from models.message import Message

_LABELS = ["lore", "noise", "mechanic", "ambiguous"]


def _label_for(content):
    """Deterministic label from the message's numeric suffix ("msg 7" -> index 7)."""
    idx = int(content.rsplit(" ", 1)[-1])
    return _LABELS[idx % len(_LABELS)]


# --- content-addressed, thread-safe fake client ----------------------------- #

class _Block:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _Response:
    def __init__(self, text):
        self.content = [_Block(text)]


class _Messages:
    """Reads the batch payload and echoes a deterministic {id,label} per message,
    so the response never depends on which thread called first."""

    def __init__(self):
        self._lock = threading.Lock()
        self.calls = 0

    def create(self, **kwargs):
        with self._lock:
            self.calls += 1
        payload = json.loads(kwargs["messages"][0]["content"])
        out = [{"id": obj["id"], "label": _label_for(obj["content"])} for obj in payload]
        return _Response(json.dumps(out))


class _Client:
    def __init__(self):
        self.messages = _Messages()


def _msg(i):
    return Message(sender="Matt", timestamp=datetime(2024, 1, 1, 12, 0, 0),
                   content=f"msg {i}", source_file="group.txt")


# --- tests ------------------------------------------------------------------ #

def test_parallel_classify_matches_sequential_and_preserves_order():
    messages = [_msg(i) for i in range(25)]      # 7 batches at batch_size=4
    expected = [(m, _label_for(m.content)) for m in messages]

    seq = NoiseFilterAgent(client=_Client(), batch_size=4, max_workers=1)
    par = NoiseFilterAgent(client=_Client(), batch_size=4, max_workers=8)

    seq_result = seq.classify(messages)
    par_result = par.classify(messages)

    assert seq_result == expected                # sequential is correct + in input order
    assert par_result == expected                # parallel returns the identical list
    assert [m for m, _ in par_result] == messages   # original Message order preserved


def test_parallel_still_sends_one_call_per_batch():
    messages = [_msg(i) for i in range(10)]      # 5 batches at batch_size=2
    client = _Client()
    agent = NoiseFilterAgent(client=client, batch_size=2, max_workers=5)
    agent.classify(messages)
    assert client.messages.calls == 5            # workers change speed, not call count


def test_single_batch_never_spins_up_a_pool():
    # <=1 batch stays on the sequential path (guarded), so tiny runs pay no thread cost.
    messages = [_msg(i) for i in range(3)]
    client = _Client()
    agent = NoiseFilterAgent(client=client, batch_size=50, max_workers=8)
    result = agent.classify(messages)
    assert [lbl for _, lbl in result] == [_label_for(m.content) for m in messages]
    assert client.messages.calls == 1


def test_empty_input_makes_no_call_regardless_of_workers():
    client = _Client()
    agent = NoiseFilterAgent(client=client, max_workers=8)
    assert agent.classify([]) == []
    assert client.messages.calls == 0


def test_max_workers_must_be_positive():
    with pytest.raises(ValueError):
        NoiseFilterAgent(client=_Client(), max_workers=0)
