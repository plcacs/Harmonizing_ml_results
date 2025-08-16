def setup(hass: HomeAssistant, config: ConfigType) -> bool:
    conf: dict = config[DOMAIN]
    host: str = conf.get(CONF_HOST)
    prefix: str = conf.get(CONF_PREFIX)
    port: int = conf.get(CONF_PORT)
    protocol: str = conf.get(CONF_PROTOCOL)

class GraphiteFeeder(threading.Thread):
    def __init__(self, hass: HomeAssistant, host: str, port: int, protocol: str, prefix: str):
    def start_listen(self, event):
    def shutdown(self, event):
    def event_listener(self, event):
    def _send_to_graphite(self, data: str):
    def _report_attributes(self, entity_id: str, new_state):
