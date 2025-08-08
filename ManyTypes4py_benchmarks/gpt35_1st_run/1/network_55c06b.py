def is_internal_request(hass: HomeAssistant) -> bool:
def get_supervisor_network_url(hass: HomeAssistant, *, allow_ssl: bool = False) -> Optional[str]:
def is_hass_url(hass: HomeAssistant, url: str) -> bool:
def get_url(hass: HomeAssistant, *, require_current_request: bool = False, require_ssl: bool = False, require_standard_port: bool = False, require_cloud: bool = False, allow_internal: bool = True, allow_external: bool = True, allow_cloud: bool = True, allow_ip: Optional[bool] = None, prefer_external: Optional[bool] = None, prefer_cloud: bool = False) -> str:
def _get_request_host() -> str:
def _get_internal_url(hass: HomeAssistant, *, allow_ip: bool = True, require_current_request: bool = False, require_ssl: bool = False, require_standard_port: bool = False) -> str:
def _get_external_url(hass: HomeAssistant, *, allow_cloud: bool = True, allow_ip: bool = True, prefer_cloud: bool = False, require_current_request: bool = False, require_ssl: bool = False, require_standard_port: bool = False, require_cloud: bool = False) -> str:
def _get_cloud_url(hass: HomeAssistant, require_current_request: bool = False) -> str:
def is_cloud_connection(hass: HomeAssistant) -> bool:
