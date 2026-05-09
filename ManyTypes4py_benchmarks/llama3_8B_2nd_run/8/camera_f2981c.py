async def async_setup_entry(
    hass: HomeAssistant, 
    entry: EzvizConfigEntry, 
    async_add_entities: AddConfigEntryEntitiesCallback
) -> None:
    # ...

class EzvizCamera(EzvizEntity, Camera):
    """An implementation of a EZVIZ security camera."""

    _attr_name: str | None

    def __init__(
        self, 
        hass: HomeAssistant, 
        coordinator: EzvizDataUpdateCoordinator, 
        serial: str, 
        camera_username: str, 
        camera_password: str | None, 
        camera_rtsp_stream: str, 
        local_rtsp_port: int, 
        ffmpeg_arguments: str
    ) -> None:
        # ...

    @property
    def available(self) -> bool:
        """Return True if entity is available."""
        return self.data['status'] != 2

    @property
    def is_on(self) -> bool:
        """Return true if on."""
        return bool(self.data['status'])

    @property
    def is_recording(self) -> bool:
        """Return true if the device is recording."""
        return self.data['alarm_notify']

    @property
    def motion_detection_enabled(self) -> bool:
        """Camera Motion Detection Status."""
        return self.data['alarm_notify']

    def enable_motion_detection(self) -> None:
        """Enable motion detection in camera."""
        # ...

    def disable_motion_detection(self) -> None:
        """Disable motion detection."""
        # ...

    async def async_camera_image(
        self, 
        width: int | None = None, 
        height: int | None = None
    ) -> bytes | None:
        """Return a frame from the camera stream."""
        # ...

    async def stream_source(self) -> str | None:
        """Return the stream source."""
        # ...

    def perform_wake_device(self) -> None:
        """Basically wakes the camera by querying the device."""
        # ...
