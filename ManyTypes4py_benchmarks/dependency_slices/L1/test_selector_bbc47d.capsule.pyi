# === Internal dependency: homeassistant.helpers.selector ===
def selector(config): ...
def validate_selector(config): ...
class QrErrorCorrectionLevel(StrEnum):
    HIGH = 'high'

# === Internal dependency: homeassistant.util.yaml ===
from .dumper import dump

# === Third-party dependency: pytest ===
# Used symbols: mark, raises

# === Third-party dependency: voluptuous ===
# Used symbols: Invalid, Schema