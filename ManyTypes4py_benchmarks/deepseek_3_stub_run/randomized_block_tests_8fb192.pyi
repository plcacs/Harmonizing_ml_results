"""
Utility code to generate randomized block tests
"""
import sys
from random import Random
from typing import (
    Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
)

# Re-exported imports
from eth2spec.test.helpers.execution_payload import (
    compute_el_block_hash_for_block, build_randomized_execution_payload
)
from eth2spec.test.helpers.multi_operations import (
    build_random_block_from_state_for_next_slot, get_random_bls_to_execution_changes,
    get_random_sync_aggregate, prepare_state_and_get_random_deposits,
    get_random_execution_requests
)
from eth2spec.test.helpers.inactivity_scores import randomize_inactivity_scores
from eth2spec.test.helpers.random import randomize_state as randomize_state_helper
from eth2spec.test.helpers.random import patch_state_to_non_leaking
from eth2spec.test.helpers.blob import get_sample_blob_tx
from eth2spec.test.helpers.state import (
    next_slot, next_epoch, ensure_state_has_validators_across_lifecycle,
    state_transition_and_sign_block
)

# Type variables for spec and state objects
_Spec = Any
_State = Any
_Block = Any
_SignedBlock = Any
_ScenarioState = Dict[str, Any]
_Stats = Dict[str, Any]
_Scenario = Dict[str, Any]
_Transition = Dict[str, Any]
_Mutation = Callable[[_Spec, _State, _Stats], Optional[_ScenarioState]]
_Validation = Callable[[_Spec, _State], bool]
_BlockRandomizer = Callable[[_Spec, _State, List[_SignedBlock], _ScenarioState], Optional[_Block]]

def _randomize_deposit_state(
    spec: _Spec,
    state: _State,
    stats: _Stats
) -> _ScenarioState: ...

def randomize_state(
    spec: _Spec,
    state: _State,
    stats: _Stats,
    exit_fraction: float = 0.1,
    slash_fraction: float = 0.1
) -> _ScenarioState: ...

def randomize_state_altair(
    spec: _Spec,
    state: _State,
    stats: _Stats,
    exit_fraction: float = 0.1,
    slash_fraction: float = 0.1
) -> _ScenarioState: ...

def randomize_state_bellatrix(
    spec: _Spec,
    state: _State,
    stats: _Stats,
    exit_fraction: float = 0.1,
    slash_fraction: float = 0.1
) -> _ScenarioState: ...

def randomize_state_capella(
    spec: _Spec,
    state: _State,
    stats: _Stats,
    exit_fraction: float = 0.1,
    slash_fraction: float = 0.1
) -> _ScenarioState: ...

def randomize_state_deneb(
    spec: _Spec,
    state: _State,
    stats: _Stats,
    exit_fraction: float = 0.1,
    slash_fraction: float = 0.1
) -> _ScenarioState: ...

def randomize_state_electra(
    spec: _Spec,
    state: _State,
    stats: _Stats,
    exit_fraction: float = 0.1,
    slash_fraction: float = 0.1
) -> _ScenarioState: ...

def randomize_state_fulu(
    spec: _Spec,
    state: _State,
    stats: _Stats,
    exit_fraction: float = 0.1,
    slash_fraction: float = 0.1
) -> _ScenarioState: ...

def epochs_until_leak(spec: _Spec) -> int: ...

def epochs_for_shard_committee_period(spec: _Spec) -> int: ...

def last_slot_in_epoch(spec: _Spec) -> int: ...

def random_slot_in_epoch(spec: _Spec, rng: Random = ...) -> int: ...

def penultimate_slot_in_epoch(spec: _Spec) -> int: ...

def no_block(
    _spec: _Spec,
    _pre_state: _State,
    _signed_blocks: List[_SignedBlock],
    _scenario_state: _ScenarioState
) -> None: ...

BLOCK_ATTEMPTS: int = ...

def _warn_if_empty_operations(block: _Block) -> None: ...

def _pull_deposits_from_scenario_state(
    spec: _Spec,
    scenario_state: _ScenarioState,
    existing_block_count: int
) -> List[Any]: ...

def random_block(
    spec: _Spec,
    state: _State,
    signed_blocks: List[_SignedBlock],
    scenario_state: _ScenarioState
) -> _Block: ...

SYNC_AGGREGATE_PARTICIPATION_BUCKETS: int = ...

def random_block_altair_with_cycling_sync_committee_participation(
    spec: _Spec,
    state: _State,
    signed_blocks: List[_SignedBlock],
    scenario_state: _ScenarioState
) -> _Block: ...

def random_block_bellatrix(
    spec: _Spec,
    state: _State,
    signed_blocks: List[_SignedBlock],
    scenario_state: _ScenarioState,
    rng: Random = ...
) -> _Block: ...

def random_block_capella(
    spec: _Spec,
    state: _State,
    signed_blocks: List[_SignedBlock],
    scenario_state: _ScenarioState,
    rng: Random = ...
) -> _Block: ...

def random_block_deneb(
    spec: _Spec,
    state: _State,
    signed_blocks: List[_SignedBlock],
    scenario_state: _ScenarioState,
    rng: Random = ...
) -> _Block: ...

def random_block_electra(
    spec: _Spec,
    state: _State,
    signed_blocks: List[_SignedBlock],
    scenario_state: _ScenarioState,
    rng: Random = ...
) -> _Block: ...

def random_block_fulu(
    spec: _Spec,
    state: _State,
    signed_blocks: List[_SignedBlock],
    scenario_state: _ScenarioState,
    rng: Random = ...
) -> _Block: ...

def no_op_validation(_spec: _Spec, _state: _State) -> bool: ...

def validate_is_leaking(spec: _Spec, state: _State) -> bool: ...

def validate_is_not_leaking(spec: _Spec, state: _State) -> bool: ...

def with_validation(
    transition: Union[Callable[[], _Transition], _Transition],
    validation: _Validation
) -> _Transition: ...

def no_op_transition() -> _Transition: ...

def epoch_transition(n: int = 0) -> _Transition: ...

def slot_transition(n: int = 0) -> _Transition: ...

def transition_to_leaking() -> _Transition: ...

transition_without_leak: _Transition = ...

def transition_with_random_block(
    block_randomizer: _BlockRandomizer
) -> _Transition: ...

def _randomized_scenario_setup(
    state_randomizer: Callable[[_Spec, _State, _Stats], _ScenarioState]
) -> Tuple[Tuple[_Mutation, _Validation], ...]: ...

_this_module: Any = ...

def _resolve_ref(
    ref: Union[str, Callable[..., Any]]
) -> Callable[..., Any]: ...

def _iter_temporal(
    spec: _Spec,
    description: Union[int, str, Callable[[_Spec], int]]
) -> Iterator[int]: ...

def _compute_statistics(scenario: _Scenario) -> _Stats: ...

def run_generated_randomized_test(
    spec: _Spec,
    state: _State,
    scenario: _Scenario
) -> Iterator[Tuple[str, Union[_State, List[_SignedBlock]]]]: ...