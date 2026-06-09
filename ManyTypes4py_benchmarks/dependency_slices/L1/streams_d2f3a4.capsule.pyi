# === Internal dependency: aiohttp.helpers ===
class TimerNoop(BaseTimerContext): ...
def set_result(fut, result): ...
_EXC_SENTINEL = BaseException(...)

# === Internal dependency: aiohttp.log ===
internal_logger = logging.getLogger(...)