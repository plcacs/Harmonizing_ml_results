from typing import Any

# === Internal dependency: aiohttp.helpers ===
class TimerNoop(BaseTimerContext): ...
def set_result(fut: 'asyncio.Future[_T]', result: _T) -> None: ...
_EXC_SENTINEL: Any

# === Internal dependency: aiohttp.log ===
internal_logger: getLogger