from typing import Any

# === Internal dependency: freqtrade.enums.ExitType ===
EXIT_SIGNAL: Any
ROI: Any
STOP_LOSS: Any

# === Internal dependency: freqtrade.exceptions ===
class OperationalException(FreqtradeException): ...

# === Internal dependency: freqtrade.persistence ===
from freqtrade.persistence.pairlock_middleware import PairLocks
from freqtrade.persistence.trade_model import Trade

# === Internal dependency: freqtrade.persistence.trade_model ===
class Order(ModelBase): ...

# === Internal dependency: freqtrade.plugins.protectionmanager ===
class ProtectionManager:
    def __init__(self, config, protections): ...

# === Third-party dependency: pytest ===
# Used symbols: mark, raises

# === Internal dependency: tests.conftest ===
def log_has_re(line, logs): ...
def get_patched_freqtradebot(mocker, config): ...