import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
import gevent.lock
import structlog
from typing import Callable, Generic, List, Optional, Tuple, TypeVar, Generator

from raiden.storage.serialization import DictSerializer
from raiden.storage.sqlite import (
    HIGH_STATECHANGE_ULID,
    LOW_STATECHANGE_ULID,
    EventID,
    Range,
    SerializedSQLiteStorage,
    StateChangeID,
    write_events,
    write_state_change,
)
from raiden.transfer.architecture import Event, State, StateChange, TransitionResult
from raiden.utils.copy import deepcopy
from raiden.utils.formatting import to_checksum_address
from raiden.utils.logging import redact_secret
from raiden.utils.typing import Address, RaidenDBVersion, typecheck

ST = TypeVar('ST', bound=State)
ST2 = TypeVar('ST2', bound=State)

def restore_or_init_snapshot(
    storage: SerializedSQLiteStorage,
    node_address: Address,
    initial_state: ST
) -> Tuple[ST, str, int]:
    """Restore the latest snapshot.
    Returns the ULID of the state change that is not applied and the
    accumulated number of state_changes applied to this snapshot so far.  If
    there is no snapshot the state will be primed with `initial_state`.
    """
    state_change_identifier: str = HIGH_STATECHANGE_ULID
    snapshot = storage.get_snapshot_before_state_change(state_change_identifier=state_change_identifier)
    if snapshot is not None:
        log.debug('Snapshot found',
                  from_state_change_id=snapshot.state_change_identifier,
                  to_state_change_id=state_change_identifier,
                  node=to_checksum_address(node_address))
        return (snapshot.data, snapshot.state_change_identifier, snapshot.state_change_qty)
    else:
        log.debug('No snapshot found, initializing the node state',
                  to_state_change_id=state_change_identifier,
                  node=to_checksum_address(node_address))
        storage.write_first_state_snapshot(initial_state)
        return (initial_state, LOW_STATECHANGE_ULID, 0)

def restore_state(
    transition_function: Callable[[ST, StateChange], TransitionResult],
    storage: SerializedSQLiteStorage,
    state_change_identifier: str,
    node_address: Address
) -> Optional[ST]:
    snapshot = storage.get_snapshot_before_state_change(state_change_identifier=state_change_identifier)
    if snapshot is None:
        return None
    log.debug('Snapshot found',
              from_state_change_id=snapshot.state_change_identifier,
              to_state_change_id=state_change_identifier,
              node=to_checksum_address(node_address))
    state, _ = replay_state_changes(
        node_address=node_address,
        state=snapshot.data,
        state_change_range=Range(snapshot.state_change_identifier, state_change_identifier),
        storage=storage,
        transition_function=transition_function,
    )
    return state

def replay_state_changes(
    node_address: Address,
    state: ST,
    state_change_range: Range,
    storage: SerializedSQLiteStorage,
    transition_function: Callable[[ST, StateChange], TransitionResult]
) -> Tuple[ST, int]:
    unapplied_state_changes: List[StateChange] = storage.get_statechanges_by_range(state_change_range)
    log.debug('Replaying state changes',
              replayed_state_changes=[redact_secret(DictSerializer.serialize(state_change))
                                       for state_change in unapplied_state_changes],
              node=to_checksum_address(node_address))
    for state_change in unapplied_state_changes:
        state, _ = dispatch(state, transition_function, state_change)
    return (state, len(unapplied_state_changes))

@dataclass(frozen=True)
class SavedState(Generic[ST]):
    state_change_id: str
    state: ST

class AtomicStateChangeDispatcher(ABC, Generic[ST]):
    @abstractmethod
    def dispatch(self, state_change: StateChange) -> List[Event]:
        pass

    @abstractmethod
    def latest_state(self) -> ST:
        pass

T = TypeVar('T')

def clone_state(state: ST) -> ST:
    before_copy = time.time()
    copy_state = deepcopy(state)
    log.debug('Copied state before applying state changes', duration=time.time() - before_copy)
    return copy_state

def dispatch(
    state: ST,
    state_transition: Callable[[ST, StateChange], TransitionResult],
    state_change: StateChange
) -> Tuple[ST, List[Event]]:
    iteration: TransitionResult = state_transition(state, state_change)
    typecheck(iteration, TransitionResult)
    for e in iteration.events:
        typecheck(e, Event)
    typecheck(iteration.new_state, State)
    assert iteration.new_state is not None, 'State transition did not yield new state'
    return (iteration.new_state, iteration.events)

class WriteAheadLog(Generic[ST]):
    def __init__(
        self,
        state: ST,
        storage: SerializedSQLiteStorage,
        state_transition: Callable[[ST, StateChange], TransitionResult]
    ) -> None:
        self.storage: SerializedSQLiteStorage = storage
        self.state: ST = state
        if not callable(state_transition):
            raise ValueError('state_transition must be a callable')
        self.state_transition: Callable[[ST, StateChange], TransitionResult] = state_transition
        self._lock: gevent.lock.Semaphore = gevent.lock.Semaphore()
        self.saved_state: Optional[SavedState[ST]] = None

    @contextmanager
    def process_state_change_atomically(self) -> Generator[AtomicStateChangeDispatcher[ST2], None, None]:
        class _AtomicStateChangeDispatcher(AtomicStateChangeDispatcher, Generic[ST2]):
            def __init__(
                self,
                state: ST2,
                storage: SerializedSQLiteStorage,
                state_transition: Callable[[ST2, StateChange], TransitionResult]
            ) -> None:
                self.state: ST2 = state
                self.storage: SerializedSQLiteStorage = storage
                self.state_transition: Callable[[ST2, StateChange], TransitionResult] = state_transition
                self.last_state_change_id: Optional[str] = None

            def dispatch(self, state_change: StateChange) -> List[Event]:
                self.state, events = dispatch(self.state, self.state_transition, state_change)
                state_change_id: str = self.write_state_change_and_events(state_change, events)
                self.last_state_change_id = state_change_id
                return events

            def latest_state(self) -> ST2:
                return self.state

            def write_state_change_and_events(self, state_change: StateChange, events: List[Event]) -> str:
                cursor = self.storage.database.conn.cursor()
                state_change_id: str = write_state_change(
                    ulid_factory=self.storage.database._ulid_factory(StateChangeID),
                    cursor=cursor,
                    state_change=self.storage.serializer.serialize(state_change)
                )
                event_data: List[Tuple[str, str]] = []
                for event in events:
                    event_data.append((state_change_id, self.storage.serializer.serialize(event)))
                write_events(
                    ulid_factory=self.storage.database._ulid_factory(EventID),
                    cursor=cursor,
                    events=event_data
                )
                return state_change_id

        with self._lock:
            cloned_state: ST = clone_state(self.state)
            with self.storage.database.transaction():
                dispatcher: _AtomicStateChangeDispatcher[ST] = _AtomicStateChangeDispatcher(
                    state=cloned_state,
                    storage=self.storage,
                    state_transition=self.state_transition
                )
                yield dispatcher
            self.state = cloned_state
            if dispatcher.last_state_change_id is not None:
                self.saved_state = SavedState(dispatcher.last_state_change_id, self.state)

    def snapshot(self, statechange_qty: int) -> None:
        """Snapshot the application state.

        Snapshots are used to restore the application state, either after a
        restart or a crash.
        """
        with self._lock:
            state_change_id: Optional[str] = self.saved_state.state_change_id if self.saved_state else None
            if state_change_id and self.state is not None:
                self.storage.write_state_snapshot(self.state, state_change_id, statechange_qty)

    def get_current_state(self) -> ST:
        """Returns the current node state."""
        return self.state

    @property
    def version(self) -> RaidenDBVersion:
        return self.storage.get_version()

log = structlog.get_logger(__name__)