class CloudAlexaConfig(alexa_config.AbstractConfig):
    """Alexa Configuration."""

    def __init__(self, hass: HomeAssistant, config: dict, cloud_user: Any, prefs: CloudPreferences, cloud: CloudClient) -> None:
        """Initialize the Alexa config."""
        super().__init__(hass)
        self._config: dict = config
        self._cloud_user: Any = cloud_user
        self._prefs: CloudPreferences = prefs
        self._cloud: CloudClient = cloud
        self._token: str | None = None
        self._token_valid: datetime | None = None
        self._cur_entity_prefs: dict[str, dict[str, Any]] = async_get_assistant_settings(hass, CLOUD_ALEXA)
        self._alexa_sync_unsub: asyncio.Task | None = None
        self._endpoint: str | None = None

    # ... (rest of the class remains the same)
