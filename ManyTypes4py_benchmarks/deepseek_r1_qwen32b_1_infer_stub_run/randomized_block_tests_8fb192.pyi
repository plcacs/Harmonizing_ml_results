"""
Stub file for 'randomized_block_tests_8fb192' module
"""

from typing import Any, Callable, Optional, List, Dict
from random import Random

def _randomize_deposit_state(spec: Any, state: Any, stats: Dict[str, int]) -> Dict[str, List[Any]]:
    ...

def randomize_state(spec: Any, state: Any, stats: Dict[str, int], exit_fraction: float = 0.1, slash_fraction: float = 0.1) -> Dict[str, List[Any]]:
    ...

def randomize_state_altair(spec: Any, state: Any, stats: Dict[str, int], exit_fraction: float = 0.1, slash_fraction: float = 0.1) -> Dict[str, List[Any]]:
    ...

def randomize_state_bellatrix(spec: Any, state: Any, stats: Dict[str, int], exit_fraction: float = 0.1, slash_fraction: float = 0.1) -> Dict[str, List[Any]]:
    ...

def randomize_state_capella(spec: Any, state: Any, stats: Dict[str, int], exit_fraction: float = 0.1, slash_fraction: float = 0.1) -> Dict[str, List[Any]]:
    ...

def randomize_state_deneb(spec: Any, state: Any, stats: Dict[str, int], exit_fraction: float = 0.1, slash_fraction: float = 0.1) -> Dict[str, List[Any]]:
    ...

def randomize_state_electra(spec: Any, state: Any, stats: Dict[str, int], exit_fraction: float = 0.1, slash_fraction: float = 0.1) -> Dict[str, List[Any]]:
    ...

def randomize_state_fulu(spec: Any, state: Any, stats: Dict[str, int], exit_fraction: float = 0.1, slash_fraction: float = 0.1) -> Dict[str, List[Any]]:
    ...

def epochs_until_leak(spec: Any) -> int:
    ...

def epochs_for_shard_committee_period(spec: Any) -> int:
    ...

def last_slot_in_epoch(spec: Any) -> int:
    ...

def random_slot_in_epoch(spec: Any, rng: Random = Random(1336)) -> int:
    ...

def penultimate_slot_in_epoch(spec: Any) -> int:
    ...

def no_block(_spec: Any, _pre_state: Any, _signed_blocks: Any, _scenario_state: Any) -> None:
    ...

BLOCK_ATTEMPTS: int

def _warn_if_empty_operations(block: Any) -> None:
    ...

def _pull_deposits_from_scenario_state(spec: Any, scenario_state: Dict[str, Any], existing_block_count: int) -> List[Any]:
    ...

def random_block(spec: Any, state: Any, signed_blocks: List[Any], scenario_state: Dict[str, Any]) -> Any:
    ...

SYNC_AGGREGATE_PARTICIPATION_BUCKETS: int

def random_block_altair_with_cycling_sync_committee_participation(spec: Any, state: Any, signed_blocks: List[Any], scenario_state: Dict[str, Any]) -> Any:
    ...

def random_block_bellatrix(spec: Any, state: Any, signed_blocks: List[Any], scenario_state: Dict[str, Any], rng: Optional[Random] = None) -> Any:
    ...

def random_block_capella(spec: Any, state: Any, signed_blocks: List[Any], scenario_state: Dict[str, Any], rng: Optional[Random] = None) -> Any:
    ...

def random_block_deneb(spec: Any, state: Any, signed_blocks: List[Any], scenario_state: Dict[str, Any], rng: Optional[Random] = None) -> Any:
    ...

def random_block_electra(spec: Any, state: Any, signed_blocks: List[Any], scenario_state: Dict[str, Any], rng: Optional[Random] = None) -> Any:
    ...

def random_block_fulu(spec: Any, state: Any, signed_blocks: List[Any], scenario_state: Dict[str, Any], rng: Optional[Random] = None) -> Any:
    ...

def no_op_validation(_spec: Any, _state: Any) -> bool:
    ...

def validate_is_leaking(spec: Any, state: Any) -> bool:
    ...

def validate_is_not_leaking(spec: Any, state: Any) -> bool:
    ...

def with_validation(transition: Any, validation: Callable[[Any, Any], bool]) -> Dict[str, Any]:
    ...

def no_op_transition() -> Dict[str, Any]:
    ...

def epoch_transition(n: int = 0) -> Dict[str, int]:
    ...

def slot_transition(n: int = 0) -> Dict[str, int]:
    ...

def transition_to_leaking() -> Dict[str, Any]:
    ...

transition_without_leak: Dict[str, Callable[[Any, Any], bool]]

def transition_with_random_block(block_randomizer: Callable[[Any, Any, List[Any], Dict[str, Any]], Any]) -> Dict[str, Any]:
    ...

def _randomized_scenario_setup(state_randomizer: Callable[[Any, Any, Dict[str, int]], Optional[Dict[str, Any]]]) -> Tuple[Tuple[Callable[[Any, Any, Dict[str, int]], None], Callable[[Any, Any], None]], ...]:
    ...

_this_module: Any

def _resolve_ref(ref: Any) -> Any:
    ...

def _iter_temporal(spec: Any, description: Any) -> Any:
    ...

def _compute_statistics(scenario: Dict[str, Any]) -> Dict[str, int]:
    ...

def run_generated_randomized_test(spec: Any, state: Any, scenario: Dict[str, Any]) -> Any:
    ...