    def __init__(self, data: dict, *, path: pathlib.Path = None, expected_domain: str = None, schema: Callable):
    def name(self) -> str:
    def inputs(self) -> dict:
    def metadata(self) -> dict:
    def update_metadata(self, *, source_url: str = None) -> None:
    def yaml(self) -> str:
    def validate(self) -> list[str]:
    def __init__(self, blueprint: Blueprint, config_with_inputs: dict):
    def inputs(self) -> dict:
    def inputs_with_default(self) -> dict:
    def validate(self) -> None:
    def async_substitute(self) -> dict:
    def __init__(self, hass: HomeAssistant, domain: str, logger: logging.Logger, blueprint_in_use: Callable, reload_blueprint_consumers: Callable, blueprint_schema: Callable):
    def blueprint_folder(self) -> pathlib.Path:
    def async_reset_cache(self) -> None:
    def _load_blueprint(self, blueprint_path: str) -> Blueprint:
    def _load_blueprints(self) -> dict:
    def async_get_blueprints(self) -> Awaitable[dict]:
    def async_get_blueprint(self, blueprint_path: str) -> Awaitable[Blueprint]:
    def async_inputs_from_config(self, config_with_blueprint: dict) -> Awaitable[BlueprintInputs]:
    def async_remove_blueprint(self, blueprint_path: str) -> None:
    def _create_file(self, blueprint: Blueprint, blueprint_path: str, allow_override: bool) -> bool:
    def async_add_blueprint(self, blueprint: Blueprint, blueprint_path: str, allow_override: bool = False) -> Awaitable[bool]:
    def async_populate(self) -> Awaitable[None]:
