    def __init__(self, hass: HomeAssistant, soco: SoCo, speaker_info: dict, zone_group_state_sub: SubscriptionBase) -> None:
    async def async_setup(self, entry: ConfigEntry, has_battery: bool, dispatches: list[tuple[str, Any]]) -> None:
    def setup(self, entry: ConfigEntry) -> None:
    def write_entity_states(self) -> None:
    @callback
    def async_write_entity_states(self) -> None:
    @property
    def alarms(self) -> SonosAlarms:
    @property
    def favorites(self) -> SonosFavorites:
    @property
    def is_coordinator(self) -> bool:
    @property
    def plex_plugin(self) -> PlexPlugin:
    @property
    def share_link(self) -> ShareLinkPlugin:
    @property
    def subscription_address(self) -> str:
    @property
    def missing_subscriptions(self) -> set[str]:
    def log_subscription_result(self, result: Any, event: str, level: int = logging.DEBUG) -> None:
    async def async_subscribe(self) -> None:
    async def _async_subscribe(self) -> None:
    async def _subscribe(self, target: Any, sub_callback: Callable) -> None:
    async def async_unsubscribe(self) -> None:
    @callback
    def async_renew_failed(self, exception: Exception) -> None:
    async def _async_renew_failed(self, exception: Exception) -> None:
    @callback
    def async_dispatch_event(self, event: SonosEvent) -> None:
    @callback
    def async_dispatch_alarms(self, event: SonosEvent) -> None:
    @callback
    def async_dispatch_device_properties(self, event: SonosEvent) -> None:
    async def async_update_device_properties(self, event: SonosEvent) -> None:
    @callback
    def async_dispatch_favorites(self, event: SonosEvent) -> None:
    @callback
    def async_dispatch_media_update(self, event: SonosEvent) -> None:
    @callback
    def async_update_volume(self, event: SonosEvent) -> None:
    @soco_error()
    def ping(self) -> None:
    @callback
    def speaker_activity(self, source: str) -> None:
    @callback
    def async_check_activity(self, now: datetime.datetime) -> None:
    async def _async_check_activity(self) -> None:
    async def async_offline(self) -> None:
    async def _async_offline(self) -> None:
    async def async_vanished(self, reason: str) -> None:
    async def async_rebooted(self) -> None:
    @soco_error()
    def fetch_battery_info(self) -> dict:
    async def async_update_battery_info(self, more_info: str) -> None:
    @property
    def power_source(self) -> str:
    @property
    def charging(self) -> bool:
    async def async_poll_battery(self, now: datetime.datetime = None) -> None:
    def update_groups(self) -> None:
    @callback
    def async_update_group_for_uid(self, uid: str) -> None:
    @callback
    def async_update_groups(self, event: SonosEvent) -> None:
    def create_update_groups_coro(self, event: SonosEvent = None) -> None:
    @soco_error()
    def join(self, speakers: list[SonosSpeaker]) -> list[SonosSpeaker]:
    @staticmethod
    async def join_multi(hass: HomeAssistant, master: SonosSpeaker, speakers: list[SonosSpeaker]) -> None:
    @soco_error()
    def unjoin(self) -> None:
    @staticmethod
    async def unjoin_multi(hass: HomeAssistant, speakers: list[SonosSpeaker]) -> None:
    @soco_error()
    def snapshot(self, with_group: bool) -> None:
    @staticmethod
    async def snapshot_multi(hass: HomeAssistant, speakers: list[SonosSpeaker], with_group: bool) -> None:
    @soco_error()
    def restore(self) -> None:
    @staticmethod
    async def restore_multi(hass: HomeAssistant, speakers: list[SonosSpeaker], with_group: bool) -> None:
    @staticmethod
    async def wait_for_groups(hass: HomeAssistant, groups: list[list[SonosSpeaker]]) -> None:
