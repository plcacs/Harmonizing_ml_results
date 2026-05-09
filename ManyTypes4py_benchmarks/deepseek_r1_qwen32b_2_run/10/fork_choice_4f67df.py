from typing import NamedTuple, Sequence, Any, Generator, Tuple, List, Optional, Callable, Dict
from eth_utils import encode_hex
from eth2spec.test.exceptions import BlockNotFoundException
from eth2spec.test.helpers.attestations import next_epoch_with_attestations, next_slots_with_attestations, state_transition_with_full_block

class BlobData(NamedTuple):
    blobs: bytes
    proofs: bytes

def with_blob_data(spec: Any, blob_data: BlobData, func: Callable[[], Generator[Any, Any, None]]) -> Generator[Any, Any, None]:
    pass

def get_anchor_root(spec: Any, state: Any) -> bytes:
    pass

def tick_and_add_block(spec: Any, store: Any, signed_block: Any, test_steps: List[Dict[str, Any]], valid: bool = True, merge_block: bool = False, block_not_found: bool = False, is_optimistic: bool = False, blob_data: Optional[BlobData] = None) -> Any:
    pass

def tick_and_add_block_with_data(spec: Any, store: Any, signed_block: Any, test_steps: List[Dict[str, Any]], blob_data: BlobData, valid: bool = True) -> Generator[Any, Any, None]:
    pass

def add_attestation(spec: Any, store: Any, attestation: Any, test_steps: List[Dict[str, Any]], is_from_block: bool = False) -> Generator[Any, Any, None]:
    pass

def add_attestations(spec: Any, store: Any, attestations: Sequence[Any], test_steps: List[Dict[str, Any]], is_from_block: bool = False) -> Generator[Any, Any, None]:
    pass

def tick_and_run_on_attestation(spec: Any, store: Any, attestation: Any, test_steps: List[Dict[str, Any]], is_from_block: bool = False) -> Generator[Any, Any, None]:
    pass

def run_on_attestation(spec: Any, store: Any, attestation: Any, is_from_block: bool = False, valid: bool = True) -> None:
    pass

def get_genesis_forkchoice_store(spec: Any, genesis_state: Any) -> Any:
    pass

def get_genesis_forkchoice_store_and_block(spec: Any, genesis_state: Any) -> Tuple[Any, Any]:
    pass

def get_block_file_name(block: Any) -> str:
    pass

def get_attestation_file_name(attestation: Any) -> str:
    pass

def get_attester_slashing_file_name(attester_slashing: Any) -> str:
    pass

def get_blobs_file_name(blobs: Optional[Any] = None, blobs_root: Optional[bytes] = None) -> str:
    pass

def on_tick_and_append_step(spec: Any, store: Any, time: int, test_steps: List[Dict[str, Any]]) -> None:
    pass

def run_on_block(spec: Any, store: Any, signed_block: Any, valid: bool = True) -> None:
    pass

def add_block(spec: Any, store: Any, signed_block: Any, test_steps: List[Dict[str, Any]], valid: bool = True, block_not_found: bool = False, is_optimistic: bool = False, blob_data: Optional[BlobData] = None) -> Generator[Any, Any, None]:
    pass

def run_on_attester_slashing(spec: Any, store: Any, attester_slashing: Any, valid: bool = True) -> None:
    pass

def add_attester_slashing(spec: Any, store: Any, attester_slashing: Any, test_steps: List[Dict[str, Any]], valid: bool = True) -> Generator[Any, Any, None]:
    pass

def get_formatted_head_output(spec: Any, store: Any) -> Dict[str, Any]:
    pass

def output_head_check(spec: Any, store: Any, test_steps: List[Dict[str, Any]]) -> None:
    pass

def output_store_checks(spec: Any, store: Any, test_steps: List[Dict[str, Any]]) -> None:
    pass

def apply_next_epoch_with_attestations(spec: Any, state: Any, store: Any, fill_cur_epoch: bool, fill_prev_epoch: bool, participation_fn: Optional[Callable[[Any], Any]] = None, test_steps: Optional[List[Dict[str, Any]]] = None) -> Tuple[Any, Any, Any]:
    pass

def apply_next_slots_with_attestations(spec: Any, state: Any, store: Any, slots: int, fill_cur_epoch: bool, fill_prev_epoch: bool, test_steps: List[Dict[str, Any]], participation_fn: Optional[Callable[[Any], Any]] = None) -> Tuple[Any, Any, Any]:
    pass

def is_ready_to_justify(spec: Any, state: Any) -> bool:
    pass

def find_next_justifying_slot(spec: Any, state: Any, fill_cur_epoch: bool, fill_prev_epoch: bool, participation_fn: Optional[Callable[[Any], Any]] = None) -> Tuple[List[Any], int]:
    pass

def get_pow_block_file_name(pow_block: Any) -> str:
    pass

def add_pow_block(spec: Any, store: Any, pow_block: Any, test_steps: List[Dict[str, Any]]) -> Generator[Any, Any, None]:
    pass