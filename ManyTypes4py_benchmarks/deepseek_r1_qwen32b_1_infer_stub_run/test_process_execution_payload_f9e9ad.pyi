from __future__ import annotations
from typing import Any, Dict, Generator, List, Optional, Union
from eth2spec.test.helpers.execution_payload import (
    build_empty_execution_payload,
    build_randomized_execution_payload,
    compute_el_block_hash,
    get_execution_payload_header,
    build_state_with_incomplete_transition,
    build_state_with_complete_transition,
)
from eth2spec.test.context import (
    BELLATRIX,
    expect_assertion_error,
    spec_state_test,
    with_bellatrix_and_later,
    with_phases,
)
from eth2spec.test.helpers.state import next_slot
from eth2spec.typing import Hash32, Bytes32

def run_execution_payload_processing(
    spec: Any,
    state: Any,
    execution_payload: Any,
    valid: bool = True,
    execution_valid: bool = True,
) -> Generator[tuple[str, Any], None, None]:
    ...

def run_success_test(spec: Any, state: Any) -> Generator[tuple[str, Any], None, None]:
    ...

@with_bellatrix_and_later
@spec_state_test
def test_success_first_payload(spec: Any, state: Any) -> None:
    ...

@with_bellatrix_and_later
@spec_state_test
def test_success_regular_payload(spec: Any, state: Any) -> None:
    ...

def run_gap_slot_test(spec: Any, state: Any) -> Generator[tuple[str, Any], None, None]:
    ...

@with_bellatrix_and_later
@spec_state_test
def test_success_first_payload_with_gap_slot(spec: Any, state: Any) -> None:
    ...

@with_bellatrix_and_later
@spec_state_test
def test_success_regular_payload_with_gap_slot(spec: Any, state: Any) -> None:
    ...

def run_bad_execution_test(spec: Any, state: Any) -> Generator[tuple[str, Any], None, None]:
    ...

@with_bellatrix_and_later
@spec_state_test
def test_invalid_bad_execution_first_payload(spec: Any, state: Any) -> None:
    ...

@with_bellatrix_and_later
@spec_state_test
def test_invalid_bad_execution_regular_payload(spec: Any, state: Any) -> None:
    ...

@with_phases([BELLATRIX])
@spec_state_test
def test_bad_parent_hash_first_payload(spec: Any, state: Any) -> None:
    ...

@with_bellatrix_and_later
@spec_state_test
def test_invalid_bad_parent_hash_regular_payload(spec: Any, state: Any) -> None:
    ...

def run_bad_prev_randao_test(spec: Any, state: Any) -> Generator[tuple[str, Any], None, None]:
    ...

@with_bellatrix_and_later
@spec_state_test
def test_invalid_bad_prev_randao_first_payload(spec: Any, state: Any) -> None:
    ...

@with_bellatrix_and_later
@spec_state_test
def test_invalid_bad_pre_randao_regular_payload(spec: Any, state: Any) -> None:
    ...

def run_bad_everything_test(spec: Any, state: Any) -> Generator[tuple[str, Any], None, None]:
    ...

@with_bellatrix_and_later
@spec_state_test
def test_invalid_bad_everything_first_payload(spec: Any, state: Any) -> None:
    ...

@with_bellatrix_and_later
@spec_state_test
def test_invalid_bad_everything_regular_payload(spec: Any, state: Any) -> None:
    ...

def run_bad_timestamp_test(
    spec: Any,
    state: Any,
    is_future: bool,
) -> Generator[tuple[str, Any], None, None]:
    ...

@with_bellatrix_and_later
@spec_state_test
def test_invalid_future_timestamp_first_payload(spec: Any, state: Any) -> None:
    ...

@with_bellatrix_and_later
@spec_state_test
def test_invalid_future_timestamp_regular_payload(spec: Any, state: Any) -> None:
    ...

@with_bellatrix_and_later
@spec_state_test
def test_invalid_past_timestamp_first_payload(spec: Any, state: Any) -> None:
    ...

@with_bellatrix_and_later
@spec_state_test
def test_invalid_past_timestamp_regular_payload(spec: Any, state: Any) -> None:
    ...

def run_non_empty_extra_data_test(spec: Any, state: Any) -> Generator[tuple[str, Any], None, None]:
    ...

@with_bellatrix_and_later
@spec_state_test
def test_non_empty_extra_data_first_payload(spec: Any, state: Any) -> None:
    ...

@with_bellatrix_and_later
@spec_state_test
def test_non_empty_extra_data_regular_payload(spec: Any, state: Any) -> None:
    ...

def run_non_empty_transactions_test(spec: Any, state: Any) -> Generator[tuple[str, Any], None, None]:
    ...

@with_bellatrix_and_later
@spec_state_test
def test_non_empty_transactions_first_payload(spec: Any, state: Any) -> None:
    ...

@with_bellatrix_and_later
@spec_state_test
def test_non_empty_transactions_regular_payload(spec: Any, state: Any) -> None:
    ...

def run_zero_length_transaction_test(spec: Any, state: Any) -> Generator[tuple[str, Any], None, None]:
    ...

@with_bellatrix_and_later
@spec_state_test
def test_zero_length_transaction_first_payload(spec: Any, state: Any) -> None:
    ...

@with_bellatrix_and_later
@spec_state_test
def test_zero_length_transaction_regular_payload(spec: Any, state: Any) -> None:
    ...

def run_randomized_non_validated_execution_fields_test(
    spec: Any,
    state: Any,
    rng: Random,
    execution_valid: bool = True,
) -> Generator[tuple[str, Any], None, None]:
    ...

@with_bellatrix_and_later
@spec_state_test
def test_randomized_non_validated_execution_fields_first_payload__execution_valid(
    spec: Any,
    state: Any,
) -> None:
    ...

@with_bellatrix_and_later
@spec_state_test
def test_randomized_non_validated_execution_fields_regular_payload__execution_valid(
    spec: Any,
    state: Any,
) -> None:
    ...

@with_bellatrix_and_later
@spec_state_test
def test_invalid_randomized_non_validated_execution_fields_first_payload__execution_invalid(
    spec: Any,
    state: Any,
) -> None:
    ...

@with_bellatrix_and_later
@spec_state_test
def test_invalid_randomized_non_validated_execution_fields_regular_payload__execution_invalid(
    spec: Any,
    state: Any,
) -> None:
    ...