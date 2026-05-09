from typing import Dict, List, Optional
from unittest.mock import MagicMock
from homeassistant.components.google_assistant import http
from homeassistant.core import HomeAssistant

def mock_google_config_store(agent_user_ids: Optional[Dict[str, str]] = None) -> http.GoogleConfigStore:
    """Fake a storage for google assistant."""
    store = MagicMock(spec=http.GoogleConfigStore)
    if agent_user_ids is not None:
        store.agent_user_ids = agent_user_ids
    else:
        store.agent_user_ids = {}
    return store

class MockConfig(http.GoogleConfig):
    """Fake config that always exposes everything."""

    def __init__(self, 
                 agent_user_ids: Optional[Dict[str, str]] = None, 
                 enabled: bool = True, 
                 entity_config: Optional[Dict[str, dict]] = None, 
                 hass: HomeAssistant, 
                 secure_devices_pin: Optional[str] = None, 
                 should_2fa: Optional[bool] = None, 
                 should_expose: Optional[bool] = None, 
                 should_report_state: bool = False) -> None:
        """Initialize config."""
        super().__init__(hass, None)
        self._enabled = enabled
        self._entity_config = entity_config or {}
        self._secure_devices_pin = secure_devices_pin
        self._should_2fa = should_2fa
        self._should_expose = should_expose
        self._should_report_state = should_report_state
        self._store = mock_google_config_store(agent_user_ids)

    @property
    def enabled(self) -> bool:
        """Return if Google is enabled."""
        return self._enabled

    @property
    def secure_devices_pin(self) -> Optional[str]:
        """Return secure devices pin."""
        return self._secure_devices_pin

    @property
    def entity_config(self) -> Optional[Dict[str, dict]]:
        """Return secure devices pin."""
        return self._entity_config

    def get_agent_user_id_from_context(self, context: http.Context) -> str:
        """Get agent user ID making request."""
        return context.user_id

    def should_expose(self, state: Optional[Dict[str, str]]) -> Optional[bool]:
        """Expose it all."""
        return self._should_expose is None or self._should_expose(state)

    @property
    def should_report_state(self) -> bool:
        """Return if states should be proactively reported."""
        return self._should_report_state

    def should_2fa(self, state: Optional[Dict[str, str]]) -> Optional[bool]:
        """Expose it all."""
        return self._should_2fa is None or self._should_2fa(state)

BASIC_CONFIG = MockConfig()
DEMO_DEVICES: List[Dict[str, dict]] = [
    # ... devices list
]
