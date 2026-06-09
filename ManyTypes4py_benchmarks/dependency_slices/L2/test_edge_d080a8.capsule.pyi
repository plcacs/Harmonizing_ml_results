from typing import Any

# === Internal dependency: freqtrade.data.converter ===
# re-export: from freqtrade.data.converter.converter import ohlcv_to_dataframe

# === Internal dependency: freqtrade.edge ===
# re-export: from .edge_positioning import Edge
# re-export: from .edge_positioning import PairInfo

# === Internal dependency: freqtrade.enums.ExitType ===
EXIT_SIGNAL: Any
STOP_LOSS: Any

# === Internal dependency: freqtrade.exceptions ===
class OperationalException(FreqtradeException): ...

# === Internal dependency: freqtrade.util.datetime_helpers ===
def dt_utc(year: int, month: int, day: int, hour: int = ..., minute: int = ..., second: int = ..., microsecond: int = ...) -> datetime: ...
def dt_ts(dt: datetime | None = ...) -> int: ...

# === Third-party dependency: numpy ===
# Used symbols: datetime64

# === Third-party dependency: pandas ===
# Used symbols: DataFrame

# === Third-party dependency: pytest ===
# Used symbols: mark, raises

# === Internal dependency: tests.conftest ===
def log_has(line, logs) -> Any: ...
def get_patched_freqtradebot(mocker, config) -> FreqtradeBot: ...
EXMS: str

# === Internal dependency: tests.optimize ===
class BTrade(NamedTuple): ...
class BTContainer(NamedTuple):
    ...
def _get_frame_time_from_offset(offset) -> Any: ...
def _build_backtest_dataframe(data) -> Any: ...