from typing import Dict, Any, List, Callable

def mock_google_config_store(agent_user_ids: Dict[str, Any] = None) -> MagicMock:
    ...

class MockConfig(http.GoogleConfig):
    def __init__(self, *, agent_user_ids: Dict[str, Any] = None, enabled: bool = True, entity_config: Dict[str, Any] = None, hass: HomeAssistant = None, secure_devices_pin: str = None, should_2fa: Callable = None, should_expose: Callable = None, should_report_state: bool = False):
        ...

    @property
    def enabled(self) -> bool:
        ...

    @property
    def secure_devices_pin(self) -> str:
        ...

    @property
    def entity_config(self) -> Dict[str, Any]:
        ...

    def get_agent_user_id_from_context(self, context: Any) -> str:
        ...

    def should_expose(self, state: Any) -> bool:
        ...

    @property
    def should_report_state(self) -> bool:
        ...

    def should_2fa(self, state: Any) -> bool:
        ...

BASIC_CONFIG: MockConfig = MockConfig()
DEMO_DEVICES: List[Dict[str, Any]] = [{'id': 'light.kitchen_lights', 'name': {'name': 'Kitchen Lights'}, 'traits': ['action.devices.traits.OnOff', 'action.devices.traits.Brightness', 'action.devices.traits.ColorSetting'], 'type': 'action.devices.types.LIGHT', 'willReportState': False}, ...]
