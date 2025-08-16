from eth2spec import Spec  # type: ignore
from eth2spec.test.context import expect_assertion_error
from eth2spec.test.helpers.keys import privkeys
from eth2spec.test.helpers.block import build_empty_block_for_next_slot
from eth2spec.test.helpers.block_processing import run_block_processing_to
from eth2spec.utils import bls
from typing import List, Tuple, Optional
from collections import Counter

def compute_sync_committee_signature(spec: Spec, state: Any, slot: int, privkey: int, block_root: Optional[bytes] = None, domain_type: Optional[int] = None) -> bytes:
    ...

def compute_aggregate_sync_committee_signature(spec: Spec, state: Any, slot: int, participants: List[int], block_root: Optional[bytes] = None, domain_type: Optional[int] = None) -> bytes:
    ...

def compute_sync_committee_inclusion_reward(spec: Spec, state: Any) -> int:
    ...

def compute_sync_committee_participant_reward_and_penalty(spec: Spec, state: Any, participant_index: int, committee_indices: List[int], committee_bits: List[bool]) -> Tuple[int, int]:
    ...

def compute_sync_committee_proposer_reward(spec: Spec, state: Any, committee_indices: List[int], committee_bits: List[bool]) -> int:
    ...

def compute_committee_indices(state: Any, committee: Any = None) -> List[int]:
    ...

def validate_sync_committee_rewards(spec: Spec, pre_state: Any, post_state: Any, committee_indices: List[int], committee_bits: List[bool], proposer_index: int) -> None:
    ...

def run_sync_committee_processing(spec: Spec, state: Any, block: Any, expect_exception: bool = False, skip_reward_validation: bool = False) -> None:
    ...

def _build_block_for_next_slot_with_sync_participation(spec: Spec, state: Any, committee_indices: List[int], committee_bits: List[bool]) -> Any:
    ...

def run_successful_sync_committee_test(spec: Spec, state: Any, committee_indices: List[int], committee_bits: List[bool], skip_reward_validation: bool = False) -> None:
    ...
