from typing import Any

# === Internal dependency: freqtrade.enums.ExitType ===
EXIT_SIGNAL: Any
STOPLOSS_ON_EXCHANGE: Any

# === Internal dependency: freqtrade.enums.TradingMode ===
FUTURES: Any
SPOT: Any

# === Internal dependency: freqtrade.persistence ===
from freqtrade.persistence.trade_model import Trade

# === Internal dependency: freqtrade.persistence.models ===
from freqtrade.persistence.trade_model import Order

# === Internal dependency: freqtrade.rpc.rpc ===
class RPC:
    def __init__(self, freqtrade): ...
    def _rpc_force_entry(self, pair, price, *, order_type=..., order_side=..., stake_amount=..., enter_tag=..., leverage=...): ...

# === Third-party dependency: pytest ===
# Used symbols: approx, mark

# === Third-party dependency: sqlalchemy ===
# Used symbols: select

# === Internal dependency: tests.conftest ===
def log_has_re(line, logs): ...
def get_patched_freqtradebot(mocker, config): ...
def patch_get_signal(freqtrade, enter_long=..., exit_long=..., enter_short=..., exit_short=..., enter_tag=..., exit_tag=...): ...
EXMS = 'freqtrade.exchange.exchange.Exchange'