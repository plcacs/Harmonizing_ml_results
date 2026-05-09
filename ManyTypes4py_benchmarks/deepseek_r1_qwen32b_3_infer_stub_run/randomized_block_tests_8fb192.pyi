"""
Stub file for 'randomized_block_tests_8fb192' module
"""

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    overload,
)
from random import Random
from eth2spec.typing import (
    BLSSignature,
    Bytes32,
    CommitteeIndex,
    Slot,
    ValidatorIndex,
    Gwei,
    Epoch,
    Root,
    Version,
    ParticipationFlags,
    SyncAggregate,
    ExecutionPayload,
    BLS_to_ExecutionChange,
    Block,
    SignedBlock,
    Deposit,
    ExecutionRequest,
    Opaque_tx,
    kzg_commitment,
)

__all__: List[str] = [
    'randomize_state',
    'randomize_state_altair',
    'randomize_state_bellatrix',
    'randomize_state_capella',
    'randomize_state_deneb',
    'randomize_state_electra',
    'randomize_state_fulu',
    'random_block',
    'random_block_altair_with_cycling_sync_committee_participation',
    'random_block_bellatrix',
    'random_block_capella',
    'random_block_deneb',
    'random_block_electra',
    'random_block_fulu',
    'no_op_validation',
    'validate_is_leaking',
    'validate_is_not_leaking',
    'with_validation',
    'no_op_transition',
    'epoch_transition',
    'slot_transition',
    'transition_to_leaking',
    'transition_without_leak',
    'transition_with_random_block',
    '_randomized_scenario_setup',
    'run_generated_randomized_test',
    'BLOCK_ATTEMPTS',
    'SYNC_AGGREGATE_PARTICIPATION_BUCKETS',
    'epochs_until_leak',
    'epochs_for_shard_committee_period',
    'last_slot_in_epoch',
    'random_slot_in_epoch',
    'penultimate_slot_in_epoch',
    'no_block',
]

BLOCK_ATTEMPTS: int
SYNC_AGGREGATE_PARTICIPATION_BUCKETS: int

def epochs_until_leak(spec: Any) -> int: ...
def epochs_for_shard_committee_period(spec: Any) -> int: ...
def last_slot_in_epoch(spec: Any) -> int: ...
def random_slot_in_epoch(spec: Any, rng: Random = Random(1336)) -> int: ...
def penultimate_slot_in_epoch(spec: Any) -> int: ...

def no_block(_spec: Any, _pre_state: Any, _signed_blocks: List[SignedBlock], _scenario_state: Dict[str, Any]) -> None: ...

def randomize_state(
    spec: Any,
    state: Any,
    stats: Dict[str, Any],
    exit_fraction: float = 0.1,
    slash_fraction: float = 0.1
) -> Dict[str, Any]: ...

def randomize_state_altair(
    spec: Any,
    state: Any,
    stats: Dict[str, Any],
    exit_fraction: float = 0.1,
    slash_fraction: float = 0.1
) -> Dict[str, Any]: ...

def randomize_state_bellatrix(
    spec: Any,
    state: Any,
    stats: Dict[str, Any],
    exit_fraction: float = 0.1,
    slash_fraction: float = 0.1
) -> Dict[str, Any]: ...

def randomize_state_capella(
    spec: Any,
    state: Any,
    stats: Dict[str, Any],
    exit_fraction: float = 0.1,
    slash_fraction: float = 0.1
) -> Dict[str, Any]: ...

def randomize_state_deneb(
    spec: Any,
    state: Any,
    stats: Dict[str, Any],
    exit_fraction: float = 0.1,
    slash_fraction: float = 0.1
) -> Dict[str, Any]: ...

def randomize_state_electra(
    spec: Any,
    state: Any,
    stats: Dict[str, Any],
    exit_fraction: float = 0.1,
    slash_fraction: float = 0.1
) -> Dict[str, Any]: ...

def randomize_state_fulu(
    spec: Any,
    state: Any,
    stats: Dict[str, Any],
    exit_fraction: float = 0.1,
    slash_fraction: float = 0.1
) -> Dict[str, Any]: ...

def random_block(
    spec: Any,
    state: Any,
    signed_blocks: List[SignedBlock],
    scenario_state: Dict[str, Any]
) -> Block: ...

def random_block_altair_with_cycling_sync_committee_participation(
    spec: Any,
    state: Any,
    signed_blocks: List[SignedBlock],
    scenario_state: Dict[str, Any]
) -> Block: ...

def random_block_bellatrix(
    spec: Any,
    state: Any,
    signed_blocks: List[SignedBlock],
    scenario_state: Dict[str, Any],
    rng: Random = Random(3456)
) -> Block: ...

def random_block_capella(
    spec: Any,
    state: Any,
    signed_blocks: List[SignedBlock],
    scenario_state: Dict[str, Any],
    rng: Random = Random(3456)
) -> Block: ...

def random_block_deneb(
    spec: Any,
    state: Any,
    signed_blocks: List[SignedBlock],
    scenario_state: Dict[str, Any],
    rng: Random = Random(3456)
) -> Block: ...

def random_block_electra(
    spec: Any,
    state: Any,
    signed_blocks: List[SignedBlock],
    scenario_state: Dict[str, Any],
    rng: Random = Random(3456)
) -> Block: ...

def random_block_fulu(
    spec: Any,
    state: Any,
    signed_blocks: List[SignedBlock],
    scenario_state: Dict[str, Any],
    rng: Random = Random(3456)
) -> Block: ...

def no_op_validation(_spec: Any, _state: Any) -> bool: ...
def validate_is_leaking(_spec: Any, _state: Any) -> bool: ...
def validate_is_not_leaking(_spec: Any, _state: Any) -> bool: ...

def with_validation(transition: Any, validation: Callable[[Any, Any], bool]) -> Dict[str, Any]: ...

def no_op_transition() -> Dict[str, Any]: ...

def epoch_transition(n: int = 0) -> Dict[str, Any]: ...
def slot_transition(n: int = 0) -> Dict[str, Any]: ...

def transition_to_leaking() -> Dict[str, Any]: ...

transition_without_leak: Dict[str, Any]

def transition_with_random_block(block_randomizer: Callable[[Any, Any, List[SignedBlock], Dict[str, Any]], Block]) -> Dict[str, Any]: ...

def _randomized_scenario_setup(state_randomizer: Callable[[Any, Any, Dict[str, Any]], Optional[Dict[str, Any]]]) -> Tuple[Tuple[Callable[[Any, Any, Dict[str, Any]], None], Callable[[Any, Any], bool]], ...]: ...

def run_generated_randomized_test(
    spec: Any,
    state: Any,
    scenario: Dict[str, Any]
) -> Generator[Tuple[str, Union[Any, List[SignedBlock]]], None, None]: ...