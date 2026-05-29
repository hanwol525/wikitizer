import json


def load_speaker_map(path: str) -> dict[str, str]:
    """Load the phone-number -> name map from a JSON config file.

    The JSON is a flat dict. Phone-number keys start with "+"; the special
    "exporter" key names whoever exported the chat (their messages have no
    phone prefix). See config/speaker_map.json for the shape.

    Errors are intentionally left to bubble up: a missing file raises
    FileNotFoundError and malformed JSON raises json.JSONDecodeError. Both
    messages are already clear, so we don't wrap them.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
