from eth2spec.types import Spec, BeaconState, PendingConsolidation
from typing import Generator, List, Tuple, Any

def test_basic_pending_consolidation(spec: Spec, state: BeaconState) -> Generator[Any, None, None]:
    ...

def test_consolidation_not_yet_withdrawable_validator(spec: Spec, state: BeaconState) -> Generator[Any, None, None]:
    ...

def test_skip_consolidation_when_source_slashed(spec: Spec, state: BeaconState) -> Generator[Any, None, None]:
    ...

def test_all_consolidation_cases_together(spec: Spec, state: BeaconState) -> Generator[Any, None, None]:
    ...

def test_pending_consolidation_future_epoch(spec: Spec, state: BeaconState) -> Generator[Any, None, None]:
    ...

def test_pending_consolidation_compounding_creds(spec: Spec, state: BeaconState) -> Generator[Any, None, None]:
    ...

def test_pending_consolidation_with_pending_deposit(spec: Spec, state: BeaconState) -> Generator[Any, None, None]:
    ...

def test_pending_consolidation_source_balance_less_than_max_effective(spec: Spec, state: BeaconState) -> Generator[Any, None, None]:
    ...

def test_pending_consolidation_source_balance_greater_than_max_effective(spec: Spec, state: BeaconState) -> Generator[Any, None, None]:
    ...

def test_pending_consolidation_source_balance_less_than_max_effective_compounding(spec: Spec, state: BeaconState) -> Generator[Any, None, None]:
    ...

def test_pending_consolidation_source_balance_greater_than_max_effective_compounding(spec: Spec, state: BeaconState) -> Generator[Any, None, None]:
    ...

def prepare_consolidation_and_state(
    spec: Spec,
    state: BeaconState,
    source_index: int,
    target_index: int,
    creds_type: str,
    balance_to_eb: str,
    eb_to_min_ab: str,
    eb_to_max_eb: str
) -> None:
    ...

def run_balance_computation_test(
    spec: Spec,
    state: BeaconState,
    instance_tuples: List[Tuple[str, str, str, str]]
) -> Generator[Any, None, None]:
    ...

def test_pending_consolidation_balance_computation_eth1(spec: Spec, state: BeaconState) -> Generator[Any, None, None]:
    ...

def test_pending_consolidation_balance_computation_compounding(spec: Spec, state: BeaconState) -> Generator[Any, None, None]:
    ...