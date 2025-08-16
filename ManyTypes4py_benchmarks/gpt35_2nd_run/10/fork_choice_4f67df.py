from typing import NamedTuple, Sequence, Any, Generator
from eth_utils import encode_hex
from eth2spec.test.exceptions import BlockNotFoundException
from eth2spec.test.helpers.attestations import next_epoch_with_attestations, next_slots_with_attestations, state_transition_with_full_block

class BlobData(NamedTuple):
    """
    The return values of ``retrieve_blobs_and_proofs`` helper.
    """

def with_blob_data(spec: Any, blob_data: BlobData, func: Any) -> Generator:
    """
    This helper runs the given ``func`` with monkeypatched ``retrieve_blobs_and_proofs``
    that returns ``blob_data.blobs, blob_data.proofs``.
    """

def get_anchor_root(spec: Any, state: Any) -> Any:

def tick_and_add_block(spec: Any, store: Any, signed_block: Any, test_steps: Any, valid: bool = True, merge_block: bool = False, block_not_found: bool = False, is_optimistic: bool = False, blob_data: Any = None) -> Any:

def tick_and_add_block_with_data(spec: Any, store: Any, signed_block: Any, test_steps: Any, blob_data: Any, valid: bool = True) -> Generator:

def add_attestation(spec: Any, store: Any, attestation: Any, test_steps: Any, is_from_block: bool = False) -> Generator:

def add_attestations(spec: Any, store: Any, attestations: Sequence[Any], test_steps: Any, is_from_block: bool = False) -> Generator:

def tick_and_run_on_attestation(spec: Any, store: Any, attestation: Any, test_steps: Any, is_from_block: bool = False) -> Generator:

def run_on_attestation(spec: Any, store: Any, attestation: Any, is_from_block: bool = False, valid: bool = True) -> Any:

def get_genesis_forkchoice_store(spec: Any, genesis_state: Any) -> Any:

def get_genesis_forkchoice_store_and_block(spec: Any, genesis_state: Any) -> Any:

def get_block_file_name(block: Any) -> str:

def get_attestation_file_name(attestation: Any) -> str:

def get_attester_slashing_file_name(attester_slashing: Any) -> str:

def get_blobs_file_name(blobs: Any = None, blobs_root: Any = None) -> str:

def on_tick_and_append_step(spec: Any, store: Any, time: int, test_steps: Any) -> Any:

def run_on_block(spec: Any, store: Any, signed_block: Any, valid: bool = True) -> Any:

def add_block(spec: Any, store: Any, signed_block: Any, test_steps: Any, valid: bool = True, block_not_found: bool = False, is_optimistic: bool = False, blob_data: Any = None) -> Generator:

def run_on_attester_slashing(spec: Any, store: Any, attester_slashing: Any, valid: bool = True) -> Any:

def add_attester_slashing(spec: Any, store: Any, attester_slashing: Any, test_steps: Any, valid: bool = True) -> Generator:

def get_formatted_head_output(spec: Any, store: Any) -> Any:

def output_head_check(spec: Any, store: Any, test_steps: Any) -> Any:

def output_store_checks(spec: Any, store: Any, test_steps: Any) -> Any:

def apply_next_epoch_with_attestations(spec: Any, state: Any, store: Any, fill_cur_epoch: bool, fill_prev_epoch: bool, participation_fn: Any = None, test_steps: Any = None) -> Generator:

def apply_next_slots_with_attestations(spec: Any, state: Any, store: Any, slots: int, fill_cur_epoch: bool, fill_prev_epoch: bool, test_steps: Any, participation_fn: Any = None) -> Generator:

def is_ready_to_justify(spec: Any, state: Any) -> bool:

def find_next_justifying_slot(spec: Any, state: Any, fill_cur_epoch: bool, fill_prev_epoch: bool, participation_fn: Any = None) -> Any:

def get_pow_block_file_name(pow_block: Any) -> str:

def add_pow_block(spec: Any, store: Any, pow_block: Any, test_steps: Any) -> Generator:
