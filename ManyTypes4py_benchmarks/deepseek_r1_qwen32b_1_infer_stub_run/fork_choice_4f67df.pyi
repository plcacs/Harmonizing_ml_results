from typing import (
    Any,
    Callable,
    Generator,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
    Dict,
)
from eth2spec.core.typing import (
    BeaconBlock,
    BeaconState,
    Bytes32,
    Checkpoint,
    Root,
    Slot,
    SignedBlock,
    Attestation,
    AttesterSlashing,
    POWBlock,
)
from eth2spec.test.helpers.attestations import (
    next_epoch_with_attestations,
    next_slots_with_attestations,
    state_transition_with_full_block,
)

class BlobData(NamedTuple):
    ...

class AtomicBoolean:
    value: bool

def with_blob_data(
    spec: Any,
    blob_data: BlobData,
    func: Callable[[], Generator[Any, None, None]],
) -> Generator[Any, None, None]:
    ...

def get_anchor_root(spec: Any, state: BeaconState) -> bytes:
    ...

def tick_and_add_block(
    spec: Any,
    store: Any,
    signed_block: SignedBlock,
    test_steps: List[Dict[str, Any]],
    valid: bool = True,
    merge_block: bool = False,
    block_not_found: bool = False,
    is_optimistic: bool = False,
    blob_data: Optional[BlobData] = None,
) -> BeaconState:
    ...

def tick_and_add_block_with_data(
    spec: Any,
    store: Any,
    signed_block: SignedBlock,
    test_steps: List[Dict[str, Any]],
    blob_data: BlobData,
    valid: bool = True,
) -> Generator[BeaconState, None, None]:
    ...

def add_attestation(
    spec: Any,
    store: Any,
    attestation: Attestation,
    test_steps: List[Dict[str, Any]],
    is_from_block: bool = False,
) -> Generator[Tuple[str, Attestation], None, None]:
    ...

def add_attestations(
    spec: Any,
    store: Any,
    attestations: Sequence[Attestation],
    test_steps: List[Dict[str, Any]],
    is_from_block: bool = False,
) -> Generator[Tuple[str, Attestation], None, None]:
    ...

def tick_and_run_on_attestation(
    spec: Any,
    store: Any,
    attestation: Attestation,
    test_steps: List[Dict[str, Any]],
    is_from_block: bool = False,
) -> Generator[Tuple[str, Attestation], None, None]:
    ...

def run_on_attestation(
    spec: Any,
    store: Any,
    attestation: Attestation,
    is_from_block: bool = False,
    valid: bool = True,
) -> None:
    ...

def get_genesis_forkchoice_store(spec: Any, genesis_state: BeaconState) -> Any:
    ...

def get_genesis_forkchoice_store_and_block(
    spec: Any,
    genesis_state: BeaconState,
) -> Tuple[Any, BeaconBlock]:
    ...

def get_block_file_name(block: Union[BeaconBlock, SignedBlock]) -> str:
    ...

def get_attestation_file_name(attestation: Attestation) -> str:
    ...

def get_attester_slashing_file_name(attester_slashing: AttesterSlashing) -> str:
    ...

def get_blobs_file_name(blobs: Optional[Any] = None, blobs_root: Optional[Bytes32] = None) -> str:
    ...

def on_tick_and_append_step(
    spec: Any,
    store: Any,
    time: int,
    test_steps: List[Dict[str, Any]],
) -> None:
    ...

def run_on_block(
    spec: Any,
    store: Any,
    signed_block: SignedBlock,
    valid: bool = True,
) -> None:
    ...

def add_block(
    spec: Any,
    store: Any,
    signed_block: SignedBlock,
    test_steps: List[Dict[str, Any]],
    valid: bool = True,
    block_not_found: bool = False,
    is_optimistic: bool = False,
    blob_data: Optional[BlobData] = None,
) -> BeaconState:
    ...

def run_on_attester_slashing(
    spec: Any,
    store: Any,
    attester_slashing: AttesterSlashing,
    valid: bool = True,
) -> None:
    ...

def add_attester_slashing(
    spec: Any,
    store: Any,
    attester_slashing: AttesterSlashing,
    test_steps: List[Dict[str, Any]],
    valid: bool = True,
) -> Generator[Tuple[str, AttesterSlashing], None, None]:
    ...

def get_formatted_head_output(spec: Any, store: Any) -> Dict[str, Union[int, str]]:
    ...

def output_head_check(
    spec: Any,
    store: Any,
    test_steps: List[Dict[str, Any]],
) -> None:
    ...

def output_store_checks(
    spec: Any,
    store: Any,
    test_steps: List[Dict[str, Any]],
) -> None:
    ...

def apply_next_epoch_with_attestations(
    spec: Any,
    state: BeaconState,
    store: Any,
    fill_cur_epoch: bool,
    fill_prev_epoch: bool,
    participation_fn: Optional[Callable[[int], float]] = None,
    test_steps: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[BeaconState, Any, SignedBlock]:
    ...

def apply_next_slots_with_attestations(
    spec: Any,
    state: BeaconState,
    store: Any,
    slots: int,
    fill_cur_epoch: bool,
    fill_prev_epoch: bool,
    test_steps: List[Dict[str, Any]],
    participation_fn: Optional[Callable[[int], float]] = None,
) -> Tuple[BeaconState, Any, SignedBlock]:
    ...

def is_ready_to_justify(spec: Any, state: BeaconState) -> bool:
    ...

def find_next_justifying_slot(
    spec: Any,
    state: BeaconState,
    fill_cur_epoch: bool,
    fill_prev_epoch: bool,
    participation_fn: Optional[Callable[[int], float]] = None,
) -> Tuple[List[SignedBlock], Slot]:
    ...

def get_pow_block_file_name(pow_block: POWBlock) -> str:
    ...

def add_pow_block(
    spec: Any,
    store: Any,
    pow_block: POWBlock,
    test_steps: List[Dict[str, Any]],
) -> Generator[Tuple[str, POWBlock], None, None]:
    ...