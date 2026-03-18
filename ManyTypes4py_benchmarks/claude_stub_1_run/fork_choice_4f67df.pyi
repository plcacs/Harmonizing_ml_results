```python
from typing import NamedTuple, Sequence, Any, Generator, Optional, Dict, Tuple

class BlobData(NamedTuple):
    blobs: Any
    proofs: Any

def with_blob_data(spec: Any, blob_data: BlobData, func: Any) -> Generator[Any, None, None]: ...

def get_anchor_root(spec: Any, state: Any) -> Any: ...

def tick_and_add_block(
    spec: Any,
    store: Any,
    signed_block: Any,
    test_steps: list,
    valid: bool = True,
    merge_block: bool = False,
    block_not_found: bool = False,
    is_optimistic: bool = False,
    blob_data: Optional[BlobData] = None,
) -> Generator[Any, None, Any]: ...

def tick_and_add_block_with_data(
    spec: Any,
    store: Any,
    signed_block: Any,
    test_steps: list,
    blob_data: BlobData,
    valid: bool = True,
) -> Generator[Any, None, None]: ...

def add_attestation(
    spec: Any,
    store: Any,
    attestation: Any,
    test_steps: list,
    is_from_block: bool = False,
) -> Generator[Tuple[str, Any], None, None]: ...

def add_attestations(
    spec: Any,
    store: Any,
    attestations: Sequence[Any],
    test_steps: list,
    is_from_block: bool = False,
) -> Generator[Tuple[str, Any], None, None]: ...

def tick_and_run_on_attestation(
    spec: Any,
    store: Any,
    attestation: Any,
    test_steps: list,
    is_from_block: bool = False,
) -> Generator[Tuple[str, Any], None, None]: ...

def run_on_attestation(
    spec: Any,
    store: Any,
    attestation: Any,
    is_from_block: bool = False,
    valid: bool = True,
) -> None: ...

def get_genesis_forkchoice_store(spec: Any, genesis_state: Any) -> Any: ...

def get_genesis_forkchoice_store_and_block(spec: Any, genesis_state: Any) -> Tuple[Any, Any]: ...

def get_block_file_name(block: Any) -> str: ...

def get_attestation_file_name(attestation: Any) -> str: ...

def get_attester_slashing_file_name(attester_slashing: Any) -> str: ...

def get_blobs_file_name(blobs: Optional[Any] = None, blobs_root: Optional[Any] = None) -> str: ...

def on_tick_and_append_step(spec: Any, store: Any, time: int, test_steps: list) -> None: ...

def run_on_block(spec: Any, store: Any, signed_block: Any, valid: bool = True) -> None: ...

def add_block(
    spec: Any,
    store: Any,
    signed_block: Any,
    test_steps: list,
    valid: bool = True,
    block_not_found: bool = False,
    is_optimistic: bool = False,
    blob_data: Optional[BlobData] = None,
) -> Generator[Tuple[str, Any], None, Any]: ...

def run_on_attester_slashing(spec: Any, store: Any, attester_slashing: Any, valid: bool = True) -> None: ...

def add_attester_slashing(
    spec: Any,
    store: Any,
    attester_slashing: Any,
    test_steps: list,
    valid: bool = True,
) -> Generator[Tuple[str, Any], None, None]: ...

def get_formatted_head_output(spec: Any, store: Any) -> Dict[str, Any]: ...

def output_head_check(spec: Any, store: Any, test_steps: list) -> None: ...

def output_store_checks(spec: Any, store: Any, test_steps: list) -> None: ...

def apply_next_epoch_with_attestations(
    spec: Any,
    state: Any,
    store: Any,
    fill_cur_epoch: Any,
    fill_prev_epoch: Any,
    participation_fn: Optional[Any] = None,
    test_steps: Optional[list] = None,
) -> Generator[Any, None, Tuple[Any, Any, Any]]: ...

def apply_next_slots_with_attestations(
    spec: Any,
    state: Any,
    store: Any,
    slots: Any,
    fill_cur_epoch: Any,
    fill_prev_epoch: Any,
    test_steps: list,
    participation_fn: Optional[Any] = None,
) -> Generator[Any, None, Tuple[Any, Any, Any]]: ...

def is_ready_to_justify(spec: Any, state: Any) -> bool: ...

def find_next_justifying_slot(
    spec: Any,
    state: Any,
    fill_cur_epoch: Any,
    fill_prev_epoch: Any,
    participation_fn: Optional[Any] = None,
) -> Tuple[list, Any]: ...

def get_pow_block_file_name(pow_block: Any) -> str: ...

def add_pow_block(spec: Any, store: Any, pow_block: Any, test_steps: list) -> Generator[Tuple[str, Any], None, None]: ...
```