from typing import Any

# === Internal dependency: easytrader.config.client ===
def create(broker) -> Any: ...

# === Internal dependency: easytrader.grid_strategies ===
class IGridStrategy(abc.ABC): ...
class Copy(BaseStrategy): ...

# === Internal dependency: easytrader.log ===
logger: getLogger

# === Internal dependency: easytrader.pop_dialog_handler ===
class PopDialogHandler: ...
class TradePopDialogHandler(PopDialogHandler): ...

# === Internal dependency: easytrader.refresh_strategies ===
class IRefreshStrategy(abc.ABC): ...
class Switch(IRefreshStrategy):
    def __init__(self, sleep: float = ...) -> Any: ...

# === Internal dependency: easytrader.utils.misc ===
def file2dict(path) -> Any: ...

# === Internal dependency: easytrader.utils.perf ===
def perf_clock(f) -> Any: ...

# === Third-party dependency: easyutils ===
# Used symbols: round_price_by_code

# === Third-party dependency: pywinauto ===
# Used symbols: Application, findwindows, timings