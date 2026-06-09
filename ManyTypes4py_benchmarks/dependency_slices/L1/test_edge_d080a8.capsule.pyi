from typing import Any

# === Internal dependency: freqtrade.data.converter ===
from freqtrade.data.converter.converter import ohlcv_to_dataframe

# === Internal dependency: freqtrade.edge ===
from .edge_positioning import Edge
from .edge_positioning import PairInfo

# === Internal dependency: freqtrade.enums.ExitType ===
EXIT_SIGNAL: Any
STOP_LOSS: Any

# === Internal dependency: freqtrade.exceptions ===
class OperationalException(FreqtradeException): ...

# === Internal dependency: freqtrade.util.datetime_helpers ===
def dt_utc(year, month, day, hour=..., minute=..., second=..., microsecond=...): ...
def dt_ts(dt=...): ...

# === Third-party dependency: numpy ===
# Used symbols: datetime64

# === Third-party dependency: pandas ===
# Used symbols: DataFrame

# === Third-party dependency: pytest ===
# Used symbols: mark, raises

# === Internal dependency: tests.conftest ===
def log_has(line, logs): ...
def get_patched_freqtradebot(mocker, config): ...
EXMS = 'freqtrade.exchange.exchange.Exchange'

# === Internal dependency: tests.optimize ===
class BTrade(NamedTuple): ...
class BTContainer(NamedTuple):
    ...
def _get_frame_time_from_offset(offset): ...
def _build_backtest_dataframe(data): ...