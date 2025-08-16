def get_full_significant_states_with_session(hass: HomeAssistant, session: Session, start_time: datetime, end_time: datetime = None, entity_ids: list[str] = None, filters: Filters = None, include_start_time_state: bool = True, significant_changes_only: bool = True, no_attributes: bool = False) -> dict[str, Any]:

def get_last_state_changes(hass: HomeAssistant, number_of_states: int, entity_id: str) -> dict[str, Any]:

def get_significant_states(hass: HomeAssistant, start_time: datetime, end_time: datetime = None, entity_ids: list[str] = None, filters: Filters = None, include_start_time_state: bool = True, significant_changes_only: bool = True, minimal_response: bool = False, no_attributes: bool = False, compressed_state_format: bool = False) -> dict[str, Any]:

def get_significant_states_with_session(hass: HomeAssistant, session: Session, start_time: datetime, end_time: datetime = None, entity_ids: list[str] = None, filters: Filters = None, include_start_time_state: bool = True, significant_changes_only: bool = True, minimal_response: bool = False, no_attributes: bool = False, compressed_state_format: bool = False) -> dict[str, Any]:

def state_changes_during_period(hass: HomeAssistant, start_time: datetime, end_time: datetime = None, entity_id: str = None, no_attributes: bool = False, descending: bool = False, limit: int = None, include_start_time_state: bool = True) -> list[State]:
