from __future__ import annotations
from typing import (
    Any,
    Generator,
    List,
    Optional,
    Union,
)
from random import Random
from eth2spec.types import (
    BeaconState,
    ExecutionPayload,
    BeaconBlockBody,
    Hash32,
    Bytes32,
    Transaction,
)
from eth2spec.test.context import spec_state_test

def run_execution_payload_processing(
    spec: Any,
    state: BeaconState,
    execution_payload: ExecutionPayload,
    valid: Optional[bool] = True,
    execution_valid: Optional[bool] = True,
) -> Generator[tuple[str, Any], None, None]:
    ...

def run_success_test(spec: Any, state: BeaconState) -> Generator[tuple[str, Any], None, None]:
    ...

@with_bellatrix_and_later
@spec_state_test
def test_success_first_payload(spec: Any, state: BeaconState) -> None:
    ...

@with_bellatrix_and_later
@spec_state_test
def test_success_regular_payload(spec: Any, state: BeaconState) -> None:
    ...

def run_gap_slot_test(spec: Any, state: BeaconState) -> Generator[tuple[str, Any], None, None]:
    ...

@with_bellatrix_and_later
@spec_state_test
def test_success_first_payload_with_gap_slot(spec: Any, state: BeaconState) -> None:
    ...

@with_bellatrix_and_later
@spec_state_test
def test_success_regular_payload_with_gap_slot(spec: Any, state: BeaconState) -> None:
    ...

def run_bad_execution_test(spec: Any, state: BeaconState) -> Generator[tuple[str, Any], None, None]:
    ...

@with_bellatrix_and_later
@spec_state_test
def test_invalid_bad_execution_first_payload(spec: Any, state: BeaconState) -> None:
    ...

@with_bellatrix_and_later
@spec_state_test
def test_invalid_bad_execution_regular_payload(spec: Any, state: BeaconState) -> None:
    ...

@with_phases([BELLATRIX])
@spec_state_test
def test_bad_parent_hash_first_payload(spec: Any, state: BeaconState) -> None:
    ...

@with_bellatrix_and_later
@spec_state_test
def test_invalid_bad_parent_hash_regular_payload(spec: Any, state: BeaconState) -> None:
    ...

def run_bad_prev_randao_test(spec: Any, state: BeaconState) -> Generator[tuple[str, Any], None, None]:
    ...

@with_bellatrix_and_later
@spec_state_test
def test_invalid_bad_prev_randao_first_payload(spec: Any, state: BeaconState) -> None:
    ...

@with_bellatrix_and_later
@spec_state_test
def test_invalid_bad_pre_randao_regular_payload(spec: Any, state: BeaconState) -> None:
    ...

def run_bad_everything_test(spec: Any, state: BeaconState) -> Generator[tuple[str, Any], None, None]:
    ...

@with_bellatrix_and_later
@spec_state_test
def test_invalid_bad_everything_first_payload(spec: Any, state: BeaconState) -> None:
    ...

@with_bellatrix_and_later
@spec_state_test
def test_invalid_bad_everything_regular_payload(spec: Any, state: BeaconState) -> None:
    ...

def run_bad_timestamp_test(
    spec: Any,
    state: BeaconState,
    is_future: bool,
) -> Generator[tuple[str, Any], None, None]:
    ...

@with_bellatrix_and_later
@spec_state_test
def test_invalid_future_timestamp_first_payload(spec: Any, state: BeaconState) -> None:
    ...

@with_bellatrix_and_later
@spec_state_test
def test_invalid_future_timestamp_regular_payload(spec: Any, state: BeaconState) -> None:
    ...

@with_bellatrix_and_later
@spec_state_test
def test_invalid_past_timestamp_first_payload(spec: Any, state: BeaconState) -> None:
    ...

@with_bellatrix_and_later
@spec_state_test
def test_invalid_past_timestamp_regular_payload(spec: Any, state: BeaconState) -> None:
    ...

def run_non_empty_extra_data_test(spec: Any, state: BeaconState) -> Generator[tuple[str, Any], None, None]:
    ...

@with_bellatrix_and_later
@spec_state_test
def test_non_empty_extra_data_first_payload(spec: Any, state: BeaconState) -> None:
    ...

@with_bellatrix_and_later
@spec_state_test
def test_non_empty_extra_data_regular_payload(spec: Any, state: BeaconState) -> None:
    ...

def run_non_empty_transactions_test(spec: Any, state: BeaconState) -> Generator[tuple[str, Any], None, None]:
    ...

@with_bellatrix_and_later
@spec_state_test
def test_non_empty_transactions_first_payload(spec: Any, state: BeaconState) -> None:
    ...

@with_bellatrix_and_later
@spec_state_test
def test_non_empty_transactions_regular_payload(spec: Any, state: BeaconState) -> None:
    ...

def run_zero_length_transaction_test(spec: Any, state: BeaconState) -> Generator[tuple[str, Any], None, None]:
    ...

@with_bellatrix_and_later
@spec_state_test
def test_zero_length_transaction_first_payload(spec: Any, state: BeaconState) -> None:
    ...

@with_bellatrix_and_later
@spec_state_test
def test_zero_length_transaction_regular_payload(spec: Any, state: BeaconState) -> None:
    ...

def run_randomized_non_validated_execution_fields_test(
    spec: Any,
    state: BeaconState,
    rng: Random,
    execution_valid: bool = True,
) -> Generator[tuple[str, Any], None, None]:
    ...

@with_bellatrix_and_later
@spec_state_test
def test_randomized_non_validated_execution_fields_first_payload__execution_valid(spec: Any, state: BeaconState) -> None:
    ...

@with_bellatrix_and_later
@spec_state_test
def test_randomized_non_validated_execution_fields_regular_payload__execution_valid(spec: Any, state: BeaconState) -> None:
    ...

@with_bellatrix_and_later
@spec_state_test
def test_invalid_randomized_non_validated_execution_fields_first_payload__execution_invalid(spec: Any, state: BeaconState) -> None:
    ...

@with_bellatrix_and_later
@spec_state_test
def test_invalid_randomized_non_validated_execution_fields_regular_payload__execution_invalid(spec: Any, state: BeaconState) -> None:
    ...