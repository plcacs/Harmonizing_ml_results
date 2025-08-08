def setup(hass: HomeAssistant, config: ConfigType) -> bool:
def _create_instance(hass: HomeAssistant, account_name: str, api_key: str, shared_secret: str, token: str, stored_rtm_config: RememberTheMilkConfiguration, component: EntityComponent[RememberTheMilkEntity]) -> None:
def _register_new_account(hass: HomeAssistant, account_name: str, api_key: str, shared_secret: str, stored_rtm_config: RememberTheMilkConfiguration, component: EntityComponent[RememberTheMilkEntity]) -> None:
class RememberTheMilkConfiguration:
    def __init__(self, hass: HomeAssistant) -> None:
    def save_config(self) -> None:
    def get_token(self, profile_name: str) -> str:
    def set_token(self, profile_name: str, token: str) -> None:
    def delete_token(self, profile_name: str) -> None:
    def _initialize_profile(self, profile_name: str) -> None:
    def get_rtm_id(self, profile_name: str, hass_id: str) -> Optional[Tuple[str, str, str]]:
    def set_rtm_id(self, profile_name: str, hass_id: str, list_id: str, time_series_id: str, rtm_task_id: str) -> None:
    def delete_rtm_id(self, profile_name: str, hass_id: str) -> None:
