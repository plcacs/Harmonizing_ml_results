from typing import NamedTuple, Sequence, Any, Generator, Optional, Union, Callable, TypeVar, Tuple, Dict

# Type variables for the 'spec' and 'store' which are likely complex domain objects
Spec = TypeVar("Spec")
Store = TypeVar("Store")
State = TypeVar("State")

class BlobData(NamedTuple):
    """
    The return values of ``retrieve_blobs_and_proofs`` helper.
    """
    blobs: Sequence[Any]
    proofs: Sequence[bytes]

def with_blob_data(
    spec: Spec, 
    blob_data: BlobData, 
    func: Callable[[], Generator[Any, Any, Any]]
) -> Generator[Any, Any, Any]: ...

def get_anchor_root(spec: Spec, state: State) -> bytes: ...

def tick_and_add_block(
    spec: Spec,
    store: Store,
    signed_block: Any,
    test_steps: list[Dict[str, Any]],
    valid: bool = True,
    merge_block: bool = False,
    block_not_found: bool = False,
    is_optimistic: bool = False,
    blob_data: Optional[BlobData] = None,
) -> Generator[Any, Any, State]: ...

def tick_and_add_block_with_data(
    spec: Spec,
    store: Store,
    signed_block: Any,
    test_steps: list[Dict[str, Any]],
    blob_data: BlobData,
    valid: bool = True,
) -> Generator[Any, Any, Any]: ...

def add_attestation(
    spec: Spec,
    store: Store,
    attestation: Any,
    test_steps: list[Dict[str, Any]],
    is_from_block: bool = False,
) -> Generator[Tuple[str, Any], Any, None]: ...

def add_attestations(
    spec: Spec,
    store: Store,
    attestations: Sequence[Any],
    test_steps: list[Dict[str, Any]],
    is_from_block: bool = False,
) -> Generator[Tuple[str, Any], Any, None]: ...

def tick_and_run_on_attestation(
    spec: Spec,
    store: Store,
    attestation: Any,
    test_steps: list[Dict[str, Any]],
    is_from_block: bool = False,
) -> Generator[Tuple[str, Any], Any, None]: ...

def run_on_attestation(
    spec: Spec,
    store: Store,
    attestation: Any,
    is_from_block: bool = False,
    valid: bool = True,
) -> None: ...

def get_genesis_forkchoice_store(spec: Spec, genesis_state: State) -> Store: ...

def get_genesis_forkchoice_store_and_block(spec: Spec, genesis_state: State) -> Tuple[Store, Any]: ...

def get_block_file_name(block: Any) -> str: ...

def get_attestation_file_name(attestation: Any) -> str: ...

def get_attester_slashing_file_name(attester_slashing: Any) -> str: ...

def get_blobs_file_name(blobs: Optional[Any] = None, blobs_root: Optional[bytes] = None) -> str: ...

def on_tick_and_append_step(spec: Spec, store: Store, time: Union[int, float], test_steps: list[Dict[str, Any]]) -> None: ...

def run_on_block(spec: Spec, store: Store, signed_block: Any, valid: bool = True) -> None: ...

def add_block(
    spec: Spec,
    store: Store,
    signed_block: Any,
    test_steps: list[Dict[str, Any]],
    valid: bool = True,
    block_not_found: bool = False,
    is_optimistic: bool = False,
    blob_data: Optional[BlobData] = None,
) -> Generator[Tuple[str, Any], Any, State]: ...

def run_on_attester_slashing(spec: Spec, store: Store, attester_slashing: Any, valid: bool = True) -> None: ...

def add_attester_slashing(
    spec: Spec,
    store: Store,
    attester_slashing: Any,
    test_steps: list[Dict[str, Any]],
    valid: bool = True,
) -> Generator[Tuple[str, Any], Any, None]: ...

def get_formatted_head_output(spec: Spec, store: Store) -> Dict[str, Union[int, str]]: ...

def output_head_check(spec: Spec, store: Store, test_steps: list[Dict[str, Any]]) -> None: ...

def output_store_checks(spec: Spec, store: Store, test_steps: list[Dict[str, Any]]) -> None: ...

def apply_next_epoch_with_attestations(
    spec: Spec,
    state: State,
    store: Store,
    fill_cur_epoch: Any,
    fill_prev_epoch: Any,
    participation_fn: Optional[Callable[..., Any]] = None,
    test_steps: Optional[list[Dict[str, Any]]] = None,
) -> Generator[Any, Any, Tuple[State, Store, Any]]: ...

def apply_next_slots_with_attestations(
    spec: Spec,
    state: State,
    store: Store,
    slots: Sequence[int],
    fill_cur_epoch: Any,
    fill_prev_epoch: Any,
    test_steps: list[Dict[str, Any]],
    participation_fn: Optional[Callable[..., Any]] = None,
) -> Generator[Any, Any, Tuple[State, Store, Any]]: ...

def is_ready_to_justify(spec: Spec, state: State) -> bool: ...

def find_next_justifying_slot(
    spec: Spec,
    state: State,
    fill_cur_epoch: Any,
    fill_prev_epoch: Any,
    participation_fn: Optional[Callable[..., Any]] = None,
) -> Tuple[list[Any], Optional[int]]: ...

def get_pow_block_file_name(pow_block: Any) -> str: ...

def add_pow_block(spec: Spec, store: Store, pow_block: Any, test_steps: list[Dict[str, Any]]) -> Generator[Tuple[str, Any], Any, None]: ...