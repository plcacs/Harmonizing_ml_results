    def _parse_identifier(cls, identifier: str) -> tuple[str | None, str | None, str | None, str | None]:
    def _get_config_or_raise(self, config_id: str) -> ConfigEntry:
    def _get_device_or_raise(self, device_id: str) -> dr.DeviceEntry:
    def _verify_kind_or_raise(cls, kind: str) -> None:
    def _get_path_or_raise(cls, path: str) -> str:
    def _get_camera_id_or_raise(cls, config: ConfigEntry, device: dr.DeviceEntry) -> str:
    def _build_media_config(cls, config: ConfigEntry) -> BrowseMediaSource:
    def _build_media_device(cls, config: ConfigEntry, device: dr.DeviceEntry, full_title: bool = True) -> BrowseMediaSource:
    def _build_media_kind(cls, config: ConfigEntry, device: dr.DeviceEntry, kind: str, full_title: bool = True) -> BrowseMediaSource:
