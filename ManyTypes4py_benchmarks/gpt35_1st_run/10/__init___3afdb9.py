def setup(hass: HomeAssistant, config: ConfigType) -> bool:
def _shutdown(_event: Event):
class ComfoConnectBridge:
    def __init__(self, hass: HomeAssistant, bridge: Bridge, name: str, token: str, friendly_name: str, pin: int):
    def sensor_callback(self, var: str, value: str):
