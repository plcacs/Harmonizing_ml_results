from typing import TypeVar, Generic, List, Tuple, Callable, Generator, Optional, Dict, Any

ST: TypeVar = TypeVar('ST', bound=State)
ST2: TypeVar = TypeVar('ST2', bound=State)

def restore_or_init_snapshot(storage: SerializedSQLiteStorage, node_address: Address, initial_state: ST) -> Tuple[ST, StateChangeID, int]:
def restore_state(transition_function: Callable[[ST, StateChange], TransitionResult[ST]], storage: SerializedSQLiteStorage, state_change_identifier: StateChangeID, node_address: Address) -> Optional[ST]:
def replay_state_changes(node_address: Address, state: ST, state_change_range: Range, storage: SerializedSQLiteStorage, transition_function: Callable[[ST, StateChange], TransitionResult[ST]]) -> Tuple[ST, int]:
class SavedState(Generic[ST]):
class AtomicStateChangeDispatcher(ABC, Generic[ST]):
def clone_state(state: ST) -> ST:
def dispatch(state: ST, state_transition: Callable[[ST, StateChange], TransitionResult[ST]], state_change: StateChange) -> Tuple[ST, List[Event]]:
class WriteAheadLog(Generic[ST]):
    def __init__(self, state: ST, storage: SerializedSQLiteStorage, state_transition: Callable[[ST, StateChange], TransitionResult[ST]]) -> None:
    def process_state_change_atomically(self) -> Generator[AtomicStateChangeDispatcher[ST2], None, None]:
    def snapshot(self, statechange_qty: int) -> None:
    def get_current_state(self) -> ST:
    @property
    def version(self) -> RaidenDBVersion:
