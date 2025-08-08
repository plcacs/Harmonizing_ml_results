from typing import Any, Dict
from homeassistant.core import HomeAssistant

def _patched_lgnetcast_client(*args: Any, session_error: bool = False, fail_connection: bool = True, invalid_details: bool = False, always_404: bool = False, no_unique_id: bool = False, **kwargs: Any) -> LgNetCastClient:

def _patch_lg_netcast(*, session_error: bool = False, fail_connection: bool = False, invalid_details: bool = False, always_404: bool = False, no_unique_id: bool = False) -> patch:

async def setup_lgnetcast(hass: HomeAssistant, unique_id: str = UNIQUE_ID) -> MockConfigEntry:
