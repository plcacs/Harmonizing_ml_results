from typing import Any

# === Internal dependency: freqtrade.constants ===
UNLIMITED_STAKE_AMOUNT: str

# === Internal dependency: freqtrade.exceptions ===
class DependencyException(FreqtradeException): ...

# === Internal dependency: freqtrade.persistence ===
# re-export: from freqtrade.persistence.trade_model import Trade

# === Third-party dependency: pytest ===
# Used symbols: approx, mark, raises

# === Third-party dependency: sqlalchemy ===
# Used symbols: select

# === Internal dependency: tests.conftest ===
def patch_wallet(mocker, free = ...) -> None: ...
def get_patched_freqtradebot(mocker, config) -> FreqtradeBot: ...
def create_mock_trades(fee, is_short: bool | None = ..., use_db: bool = ...) -> Any: ...
def create_mock_trades_usdt(fee, is_short: bool | None = ..., use_db: bool = ...) -> Any: ...
EXMS: str