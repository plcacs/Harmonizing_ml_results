# === Internal dependency: easytrader.config.client ===
def create(broker): ...

# === Internal dependency: easytrader.grid_strategies ===
class IGridStrategy(abc.ABC): ...
class Copy(BaseStrategy): ...

# === Internal dependency: easytrader.log ===
logger = logging.getLogger(...)

# === Internal dependency: easytrader.pop_dialog_handler ===
class PopDialogHandler: ...
class TradePopDialogHandler(PopDialogHandler): ...

# === Internal dependency: easytrader.refresh_strategies ===
class IRefreshStrategy(abc.ABC): ...
class Switch(IRefreshStrategy):
    def __init__(self, sleep=...): ...

# === Internal dependency: easytrader.utils.misc ===
def file2dict(path): ...

# === Internal dependency: easytrader.utils.perf ===
def perf_clock(f): ...

# === Third-party dependency: easyutils ===
# Used symbols: round_price_by_code

# === Third-party dependency: pywinauto ===
# Used symbols: Application, findwindows, timings