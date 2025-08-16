from random import Random
from eth2spec.test.helpers.keys import privkeys, pubkeys
from eth2spec.test.helpers.state import state_transition_and_sign_block
from eth2spec.test.helpers.block import build_empty_block_for_next_slot
from eth2spec.test.helpers.sync_committee import compute_committee_indices, compute_aggregate_sync_committee_signature
from eth2spec.test.helpers.proposer_slashings import get_valid_proposer_slashing
from eth2spec.test.helpers.attester_slashings import get_valid_attester_slashing_by_indices
from eth2spec.test.helpers.attestations import get_valid_attestation, get_max_attestations
from eth2spec.test.helpers.deposits import build_deposit, deposit_from_context
from eth2spec.test.helpers.voluntary_exits import prepare_signed_exits
from eth2spec.test.helpers.bls_to_execution_changes import get_signed_address_change
from typing import Iterator, Tuple, List, Any

def run_slash_and_exit(spec: Any, state: Any, slash_index: int, exit_index: int, valid: bool = True) -> Iterator[Tuple[str, Any]]:
    ...

def get_random_proposer_slashings(spec: Any, state: Any, rng: Random) -> List[Any]:
    ...

def get_random_attester_slashings(spec: Any, state: Any, rng: Random, slashed_indices: List[int] = []) -> List[Any]:
    ...

def get_random_attestations(spec: Any, state: Any, rng: Random) -> List[Any]:
    ...

def get_random_deposits(spec: Any, state: Any, rng: Random, num_deposits: int = None) -> Tuple[List[Any], bytes]:
    ...

def prepare_state_and_get_random_deposits(spec: Any, state: Any, rng: Random, num_deposits: int = None) -> List[Any]:
    ...

def _eligible_for_exit(spec: Any, state: Any, index: int) -> bool:
    ...

def get_random_voluntary_exits(spec: Any, state: Any, to_be_slashed_indices: set, rng: Random) -> List[Any]:
    ...

def get_random_sync_aggregate(spec: Any, state: Any, slot: int, block_root: bytes = None, fraction_participated: float = 1.0, rng: Random = Random(2099)) -> Any:
    ...

def get_random_bls_to_execution_changes(spec: Any, state: Any, rng: Random = Random(2188), num_address_changes: int = 0) -> List[Any]:
    ...

def build_random_block_from_state_for_next_slot(spec: Any, state: Any, rng: Random = Random(2188), deposits: List[Any] = None) -> Any:
    ...

def run_test_full_random_operations(spec: Any, state: Any, rng: Random = Random(2080)) -> Iterator[Tuple[str, Any]]:
    ...

def get_random_execution_requests(spec: Any, state: Any, rng: Random) -> Any:
    ...

def get_random_deposit_requests(spec: Any, state: Any, rng: Random, num_deposits: int = None) -> List[Any]:
    ...

def get_random_withdrawal_requests(spec: Any, state: Any, rng: Random, num_withdrawals: int = None) -> List[Any]:
    ...

def get_random_consolidation_requests(spec: Any, state: Any, rng: Random, num_consolidations: int = None) -> List[Any]:
    ...
