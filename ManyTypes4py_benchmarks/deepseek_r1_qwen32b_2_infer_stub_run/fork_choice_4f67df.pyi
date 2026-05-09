from typing import Any, Callable, Generator, List, Optional, Tuple, Union
from eth2spec.typing import (
    BeaconBlock,
    BeaconState,
    Bytes32,
    ForkchoiceStore,
    SignedBeaconBlock,
    Slot,
    StateRoot,
    Time,
    attestations,
    attester_slashings,
    blob_data,
    pow_block,
)

class BlobData(NamedTuple):
    ...

def with_blob_data(
    spec: Any,
    blob_data: Any,
    func: Callable[[], Generator[Any, Any, None]]
) -> Generator[Any, Any, None]:
    ...

def get_anchor_root(spec: Any, state: Any) -> bytes:
    ...

def tick_and_add_block(
    spec: Any,
    store: Any,
    signed_block: SignedBeaconBlock,
    test_steps: List[dict],
    valid: bool = True,
    merge_block: bool = False,
    block_not_found: bool = False,
    is_optimistic: bool = False,
    blob_data: Optional[Any] = None
) -> Any:
    ...

def tick_and_add_block_with_data(
    spec: Any,
    store: Any,
    signed_block: SignedBeaconBlock,
    test_steps: List[dict],
    blob_data: Any,
    valid: bool = True
) -> Any:
    ...

def add_attestation(
    spec: Any,
    store: Any,
    attestation: Any,
    test_steps: List[dict],
    is_from_block: bool = False
) -> Generator[Tuple[str, Any], Any, None]:
    ...

def add_attestations(
    spec: Any,
    store: Any,
    attestations: List[Any],
    test_steps: List[dict],
    is_from_block: bool = False
) -> Generator[Any, Any, None]:
    ...

def tick_and_run_on_attestation(
    spec: Any,
    store: Any,
    attestation: Any,
    test_steps: List[dict],
    is_from_block: bool = False
) -> Generator[Any, Any, None]:
    ...

def run_on_attestation(
    spec: Any,
    store: Any,
    attestation: Any,
    is_from_block: bool = False,
    valid: bool = True
) -> None:
    ...

def get_genesis_forkchoice_store(spec: Any, genesis_state: Any) -> ForkchoiceStore:
    ...

def get_genesis_forkchoice_store_and_block(
    spec: Any,
    genesis_state: Any
) -> Tuple[ForkchoiceStore, BeaconBlock]:
    ...

def get_block_file_name(block: Any) -> str:
    ...

def get_attestation_file_name(attestation: Any) -> str:
    ...

def get_attester_slashing_file_name(attester_slashing: Any) -> str:
    ...

def get_blobs_file_name(blobs: Optional[Any] = None, blobs_root: Optional[Bytes32] = None) -> str:
    ...

def on_tick_and_append_step(
    spec: Any,
    store: Any,
    time: Time,
    test_steps: List[dict]
) -> None:
    ...

def run_on_block(
    spec: Any,
    store: Any,
    signed_block: SignedBeaconBlock,
    valid: bool = True
) -> None:
    ...

def add_block(
    spec: Any,
    store: Any,
    signed_block: SignedBeaconBlock,
    test_steps: List[dict],
    valid: bool = True,
    block_not_found: bool = False,
    is_optimistic: bool = False,
    blob_data: Optional[Any] = None
) -> Generator[Any, Any, None]:
    ...

def run_on_attester_slashing(
    spec: Any,
    store: Any,
    attester_slashing: Any,
    valid: bool = True
) -> None:
    ...

def add_attester_slashing(
    spec: Any,
    store: Any,
    attester_slashing: Any,
    test_steps: List[dict],
    valid: bool = True
) -> Generator[Tuple[str, Any], Any, None]:
    ...

def get_formatted_head_output(spec: Any, store: Any) -> dict:
    ...

def output_head_check(
    spec: Any,
    store: Any,
    test_steps: List[dict]
) -> None:
    ...

def output_store_checks(
    spec: Any,
    store: Any,
    test_steps: List[dict]
) -> None:
    ...

def apply_next_epoch_with_attestations(
    spec: Any,
    state: BeaconState,
    store: ForkchoiceStore,
    fill_cur_epoch: bool,
    fill_prev_epoch: bool,
    participation_fn: Optional[Callable] = None,
    test_steps: Optional[List[dict]] = None
) -> Generator[Any, Any, None]:
    ...

def apply_next_slots_with_attestations(
    spec: Any,
    state: BeaconState,
    store: ForkchoiceStore,
    slots: List[Slot],
    fill_cur_epoch: bool,
    fill_prev_epoch: bool,
    test_steps: List[dict],
    participation_fn: Optional[Callable] = None
) -> Generator[Any, Any, None]:
    ...

def is_ready_to_justify(spec: Any, state: Any) -> bool:
    ...

def find_next_justifying_slot(
    spec: Any,
    state: Any,
    fill_cur_epoch: bool,
    fill_prev_epoch: bool,
    participation_fn: Optional[Callable] = None
) -> Tuple[List[SignedBeaconBlock], int]:
    ...

def get_pow_block_file_name(pow_block: Any) -> str:
    ...

def add_pow_block(
    spec: Any,
    store: Any,
    pow_block: Any,
    test_steps: List[dict]
) -> Generator[Tuple[str, Any], Any, None]:
    ...