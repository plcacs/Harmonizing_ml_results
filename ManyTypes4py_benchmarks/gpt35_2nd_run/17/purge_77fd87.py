def _purge_legacy_format(instance: Recorder, session: Session, purge_before: datetime) -> bool:
def _purge_states_and_attributes_ids(instance: Recorder, session: Session, states_batch_size: int, purge_before: datetime) -> bool:
def _purge_events_and_data_ids(instance: Recorder, session: Session, events_batch_size: int, purge_before: datetime) -> bool:
def _select_state_attributes_ids_to_purge(session: Session, purge_before: float, max_bind_vars: int) -> Tuple[Set[int], Set[int]]:
def _select_event_data_ids_to_purge(session: Session, purge_before: float, max_bind_vars: int) -> Tuple[Set[int], Set[int]]:
def _select_unused_attributes_ids(instance: Recorder, session: Session, attributes_ids: Set[int], database_engine: DatabaseEngine) -> Set[int]:
def _purge_unused_attributes_ids(instance: Recorder, session: Session, attributes_ids_batch: Set[int]) -> None:
def _select_unused_event_data_ids(instance: Recorder, session: Session, data_ids: Set[int], database_engine: DatabaseEngine) -> Set[int]:
def _purge_unused_data_ids(instance: Recorder, session: Session, data_ids_batch: Set[int]) -> None:
def _select_statistics_runs_to_purge(session: Session, purge_before: datetime, max_bind_vars: int) -> List[int]:
def _select_short_term_statistics_to_purge(session: Session, purge_before: datetime, max_bind_vars: int) -> List[int]:
def _select_legacy_detached_state_and_attributes_and_data_ids_to_purge(session: Session, purge_before: float, max_bind_vars: int) -> Tuple[Set[int], Set[int]]:
def _select_legacy_event_state_and_attributes_and_data_ids_to_purge(session: Session, purge_before: float, max_bind_vars: int) -> Tuple[Set[int], Set[int], Set[int], Set[int]:
def _purge_state_ids(instance: Recorder, session: Session, state_ids: Set[int]) -> None:
def _purge_batch_attributes_ids(instance: Recorder, session: Session, attributes_ids: Set[int]) -> None:
def _purge_batch_data_ids(instance: Recorder, session: Session, data_ids: Set[int]) -> None:
def _purge_statistics_runs(session: Session, statistics_runs: List[int]) -> None:
def _purge_short_term_statistics(session: Session, short_term_statistics: List[int]) -> None:
def _purge_event_ids(session: Session, event_ids: Set[int]) -> None:
def _purge_old_recorder_runs(instance: Recorder, session: Session, purge_before: datetime) -> None:
def _purge_old_event_types(instance: Recorder, session: Session) -> None:
def _purge_old_entity_ids(instance: Recorder, session: Session) -> None:
def _purge_filtered_data(instance: Recorder, session: Session) -> bool:
def _purge_filtered_states(instance: Recorder, session: Session, metadata_ids_to_purge: List[int], database_engine: DatabaseEngine, purge_before_timestamp: float) -> bool:
def _purge_filtered_events(instance: Recorder, session: Session, excluded_event_type_ids: List[int], purge_before_timestamp: float) -> bool:
def purge_entity_data(instance: Recorder, entity_filter: Callable, purge_before: datetime) -> bool:
