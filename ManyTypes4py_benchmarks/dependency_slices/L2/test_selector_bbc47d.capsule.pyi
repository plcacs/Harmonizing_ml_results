from typing import Any

# === Internal dependency: homeassistant.helpers.selector ===
def selector(config: Any) -> Selector: ...
def validate_selector(config: Any) -> dict: ...
class QrErrorCorrectionLevel(StrEnum):
    HIGH: str

# === Internal dependency: homeassistant.util.yaml ===
# re-export: from .dumper import dump

# === Third-party dependency: pytest ===
# Used symbols: mark, raises

# === Third-party dependency: voluptuous ===
# Used symbols: Invalid, Schema