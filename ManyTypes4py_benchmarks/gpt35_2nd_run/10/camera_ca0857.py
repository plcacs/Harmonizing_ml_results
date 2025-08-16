async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddConfigEntryEntitiesCallback) -> None:

def __init__(self, config_entry_id: str, username: str, password: str, camera: MappingProxyType, client: MotionEyeClient, coordinator: DataUpdateCoordinator, options: MappingProxyType) -> None:

def _get_mjpeg_camera_properties_for_camera(self, camera: MappingProxyType) -> MappingProxyType:

def _set_mjpeg_camera_state_for_camera(self, camera: MappingProxyType) -> None:

def _is_acceptable_streaming_camera(self) -> bool:

@property
def available(self) -> bool:

def _handle_coordinator_update(self) -> None:

@property
def motion_detection_enabled(self) -> bool:

async def async_set_text_overlay(self, left_text: Any = None, right_text: Any = None, custom_left_text: Any = None, custom_right_text: Any = None) -> None:

async def async_request_action(self, action: str) -> None:

async def async_request_snapshot(self) -> None:
