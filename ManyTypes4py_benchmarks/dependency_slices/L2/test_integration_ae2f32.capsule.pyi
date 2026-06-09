from typing import Any

# === Internal dependency: freqtrade.enums.ExitType ===
EXIT_SIGNAL: Any
STOPLOSS_ON_EXCHANGE: Any

# === Internal dependency: freqtrade.enums.TradingMode ===
FUTURES: Any
SPOT: Any

# === Internal dependency: freqtrade.persistence ===
# re-export: from freqtrade.persistence.trade_model import Trade

# === Internal dependency: freqtrade.persistence.models ===
# re-export: from freqtrade.persistence.trade_model import Order

# === Internal dependency: freqtrade.rpc.rpc ===
class RPC:
    def __init__(self, freqtrade) -> None: ...
    def _rpc_force_entry(self, pair: str, price: float | None, *, order_type: str | None = ..., order_side: SignalDirection = ..., stake_amount: float | None = ..., enter_tag: str | None = ..., leverage: float | None = ...) -> Trade | None: ...

# === Third-party dependency: pytest ===
# Used symbols: approx, mark

# === Third-party dependency: sqlalchemy ===
# Used symbols: select

# === Internal dependency: tests.conftest ===
def log_has_re(line, logs) -> Any: ...
def get_patched_freqtradebot(mocker, config) -> FreqtradeBot: ...
def patch_get_signal(freqtrade: FreqtradeBot, enter_long = ..., exit_long = ..., enter_short = ..., exit_short = ..., enter_tag: str | None = ..., exit_tag: str | None = ...) -> None: ...
EXMS: str