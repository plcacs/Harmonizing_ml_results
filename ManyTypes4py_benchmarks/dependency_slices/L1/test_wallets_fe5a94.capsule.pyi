# === Internal dependency: freqtrade.constants ===
UNLIMITED_STAKE_AMOUNT = 'unlimited'

# === Internal dependency: freqtrade.exceptions ===
class DependencyException(FreqtradeException): ...

# === Internal dependency: freqtrade.persistence ===
from freqtrade.persistence.trade_model import Trade

# === Third-party dependency: pytest ===
# Used symbols: approx, mark, raises

# === Third-party dependency: sqlalchemy ===
# Used symbols: select

# === Internal dependency: tests.conftest ===
def patch_wallet(mocker, free=...): ...
def get_patched_freqtradebot(mocker, config): ...
def create_mock_trades(fee, is_short=..., use_db=...): ...
def create_mock_trades_usdt(fee, is_short=..., use_db=...): ...
EXMS = 'freqtrade.exchange.exchange.Exchange'