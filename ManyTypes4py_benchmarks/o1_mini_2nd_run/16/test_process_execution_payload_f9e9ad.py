from random import Random
from typing import Generator, Tuple, Any, Optional
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
from eth2spec.typing import Spec, State, ExecutionPayload, BeaconBlockBody
from eth2spec.spec.helpers import Hash32, Transaction
from eth2spec.spec.execution_engine import NoopExecutionEngine


def run_execution_payload_processing(
    spec: Spec,
    state: State,
    execution_payload: ExecutionPayload,
    valid: bool = True,
    execution_valid: bool = True,
) -> Generator[Tuple[str, Any], None, None]:
    """
    Run ``process_execution_payload``, yielding:
      - pre-state ('pre')
      - execution payload ('execution_payload')
      - execution details, to mock EVM execution ('execution.yml', a dict with 'execution_valid' key and boolean value)
      - post-state ('post').
    If ``valid == False``, run expecting ``AssertionError``
    """
    body: BeaconBlockBody = spec.BeaconBlockBody(execution_payload=execution_payload)
    yield ('pre', state)
    yield ('execution', {'execution_valid': execution_valid})
    yield ('body', body)
    called_new_block: bool = False

    class TestEngine(NoopExecutionEngine):

        def verify_and_notify_new_payload(
            self, new_payload_request: Any
        ) -> bool:
            nonlocal called_new_block, execution_valid
            called_new_block = True
            assert new_payload_request.execution_payload == body.execution_payload
            return execution_valid

    if not valid:
        expect_assertion_error(lambda: spec.process_execution_payload(state, body, TestEngine()))
        yield ('post', None)
        return
    spec.process_execution_payload(state, body, TestEngine())
    assert called_new_block
    yield ('post', state)
    assert state.latest_execution_payload_header == get_execution_payload_header(spec, body.execution_payload)


def run_success_test(spec: Spec, state: State) -> Generator[Tuple[str, Any], None, None]:
    next_slot(spec, state)
    execution_payload: ExecutionPayload = build_empty_execution_payload(spec, state)
    yield from run_execution_payload_processing(spec, state, execution_payload)


@with_bellatrix_and_later
@spec_state_test
def test_success_first_payload(spec: Spec, state: State) -> Generator[Tuple[str, Any], None, None]:
    state = build_state_with_incomplete_transition(spec, state)
    yield from run_success_test(spec, state)


@with_bellatrix_and_later
@spec_state_test
def test_success_regular_payload(spec: Spec, state: State) -> Generator[Tuple[str, Any], None, None]:
    state = build_state_with_complete_transition(spec, state)
    yield from run_success_test(spec, state)


def run_gap_slot_test(spec: Spec, state: State) -> Generator[Tuple[str, Any], None, None]:
    next_slot(spec, state)
    next_slot(spec, state)
    execution_payload: ExecutionPayload = build_empty_execution_payload(spec, state)
    yield from run_execution_payload_processing(spec, state, execution_payload)


@with_bellatrix_and_later
@spec_state_test
def test_success_first_payload_with_gap_slot(spec: Spec, state: State) -> Generator[Tuple[str, Any], None, None]:
    state = build_state_with_incomplete_transition(spec, state)
    yield from run_gap_slot_test(spec, state)


@with_bellatrix_and_later
@spec_state_test
def test_success_regular_payload_with_gap_slot(spec: Spec, state: State) -> Generator[Tuple[str, Any], None, None]:
    state = build_state_with_complete_transition(spec, state)
    yield from run_gap_slot_test(spec, state)


def run_bad_execution_test(
    spec: Spec, state: State
) -> Generator[Tuple[str, Any], None, None]:
    next_slot(spec, state)
    execution_payload: ExecutionPayload = build_empty_execution_payload(spec, state)
    yield from run_execution_payload_processing(
        spec, state, execution_payload, valid=False, execution_valid=False
    )


@with_bellatrix_and_later
@spec_state_test
def test_invalid_bad_execution_first_payload(spec: Spec, state: State) -> Generator[Tuple[str, Any], None, None]:
    state = build_state_with_incomplete_transition(spec, state)
    yield from run_bad_execution_test(spec, state)


@with_bellatrix_and_later
@spec_state_test
def test_invalid_bad_execution_regular_payload(spec: Spec, state: State) -> Generator[Tuple[str, Any], None, None]:
    state = build_state_with_complete_transition(spec, state)
    yield from run_bad_execution_test(spec, state)


@with_phases([BELLATRIX])
@spec_state_test
def test_bad_parent_hash_first_payload(spec: Spec, state: State) -> Generator[Tuple[str, Any], None, None]:
    state = build_state_with_incomplete_transition(spec, state)
    next_slot(spec, state)
    execution_payload: ExecutionPayload = build_empty_execution_payload(spec, state)
    execution_payload.parent_hash = b'U' * 32
    execution_payload.block_hash = compute_el_block_hash(spec, execution_payload, state)
    yield from run_execution_payload_processing(spec, state, execution_payload)


@with_bellatrix_and_later
@spec_state_test
def test_invalid_bad_parent_hash_regular_payload(spec: Spec, state: State) -> Generator[Tuple[str, Any], None, None]:
    state = build_state_with_complete_transition(spec, state)
    next_slot(spec, state)
    execution_payload: ExecutionPayload = build_empty_execution_payload(spec, state)
    execution_payload.parent_hash = spec.Hash32()
    execution_payload.block_hash = compute_el_block_hash(spec, execution_payload, state)
    yield from run_execution_payload_processing(
        spec, state, execution_payload, valid=False
    )


def run_bad_prev_randao_test(
    spec: Spec, state: State
) -> Generator[Tuple[str, Any], None, None]:
    next_slot(spec, state)
    execution_payload: ExecutionPayload = build_empty_execution_payload(spec, state)
    execution_payload.prev_randao = b'B' * 32
    execution_payload.block_hash = compute_el_block_hash(spec, execution_payload, state)
    yield from run_execution_payload_processing(
        spec, state, execution_payload, valid=False
    )


@with_bellatrix_and_later
@spec_state_test
def test_invalid_bad_prev_randao_first_payload(spec: Spec, state: State) -> Generator[Tuple[str, Any], None, None]:
    state = build_state_with_incomplete_transition(spec, state)
    yield from run_bad_prev_randao_test(spec, state)


@with_bellatrix_and_later
@spec_state_test
def test_invalid_bad_pre_randao_regular_payload(spec: Spec, state: State) -> Generator[Tuple[str, Any], None, None]:
    state = build_state_with_complete_transition(spec, state)
    yield from run_bad_prev_randao_test(spec, state)


def run_bad_everything_test(
    spec: Spec, state: State
) -> Generator[Tuple[str, Any], None, None]:
    next_slot(spec, state)
    execution_payload: ExecutionPayload = build_empty_execution_payload(spec, state)
    execution_payload.parent_hash = spec.Hash32()
    execution_payload.prev_randao = spec.Bytes32()
    execution_payload.timestamp = 0
    execution_payload.block_hash = compute_el_block_hash(spec, execution_payload, state)
    yield from run_execution_payload_processing(spec, state, execution_payload, valid=False)


@with_bellatrix_and_later
@spec_state_test
def test_invalid_bad_everything_first_payload(spec: Spec, state: State) -> Generator[Tuple[str, Any], None, None]:
    state = build_state_with_incomplete_transition(spec, state)
    yield from run_bad_everything_test(spec, state)


@with_bellatrix_and_later
@spec_state_test
def test_invalid_bad_everything_regular_payload(spec: Spec, state: State) -> Generator[Tuple[str, Any], None, None]:
    state = build_state_with_complete_transition(spec, state)
    yield from run_bad_everything_test(spec, state)


def run_bad_timestamp_test(
    spec: Spec, state: State, is_future: bool
) -> Generator[Tuple[str, Any], None, None]:
    next_slot(spec, state)
    execution_payload: ExecutionPayload = build_empty_execution_payload(spec, state)
    if is_future:
        timestamp: int = execution_payload.timestamp + 1
    else:
        timestamp = execution_payload.timestamp - 1
    execution_payload.timestamp = timestamp
    execution_payload.block_hash = compute_el_block_hash(spec, execution_payload, state)
    yield from run_execution_payload_processing(
        spec, state, execution_payload, valid=False
    )


@with_bellatrix_and_later
@spec_state_test
def test_invalid_future_timestamp_first_payload(spec: Spec, state: State) -> Generator[Tuple[str, Any], None, None]:
    state = build_state_with_incomplete_transition(spec, state)
    yield from run_bad_timestamp_test(spec, state, is_future=True)


@with_bellatrix_and_later
@spec_state_test
def test_invalid_future_timestamp_regular_payload(spec: Spec, state: State) -> Generator[Tuple[str, Any], None, None]:
    state = build_state_with_complete_transition(spec, state)
    yield from run_bad_timestamp_test(spec, state, is_future=True)


@with_bellatrix_and_later
@spec_state_test
def test_invalid_past_timestamp_first_payload(spec: Spec, state: State) -> Generator[Tuple[str, Any], None, None]:
    state = build_state_with_incomplete_transition(spec, state)
    yield from run_bad_timestamp_test(spec, state, is_future=False)


@with_bellatrix_and_later
@spec_state_test
def test_invalid_past_timestamp_regular_payload(spec: Spec, state: State) -> Generator[Tuple[str, Any], None, None]:
    state = build_state_with_complete_transition(spec, state)
    yield from run_bad_timestamp_test(spec, state, is_future=False)


def run_non_empty_extra_data_test(
    spec: Spec, state: State
) -> Generator[Tuple[str, Any], None, None]:
    next_slot(spec, state)
    execution_payload: ExecutionPayload = build_empty_execution_payload(spec, state)
    execution_payload.extra_data = b'E' * 12
    execution_payload.block_hash = compute_el_block_hash(spec, execution_payload, state)
    yield from run_execution_payload_processing(spec, state, execution_payload)
    assert state.latest_execution_payload_header.extra_data == execution_payload.extra_data


@with_bellatrix_and_later
@spec_state_test
def test_non_empty_extra_data_first_payload(spec: Spec, state: State) -> Generator[Tuple[str, Any], None, None]:
    state = build_state_with_incomplete_transition(spec, state)
    yield from run_non_empty_extra_data_test(spec, state)


@with_bellatrix_and_later
@spec_state_test
def test_non_empty_extra_data_regular_payload(spec: Spec, state: State) -> Generator[Tuple[str, Any], None, None]:
    state = build_state_with_complete_transition(spec, state)
    yield from run_non_empty_extra_data_test(spec, state)


def run_non_empty_transactions_test(
    spec: Spec, state: State
) -> Generator[Tuple[str, Any], None, None]:
    next_slot(spec, state)
    execution_payload: ExecutionPayload = build_empty_execution_payload(spec, state)
    num_transactions: int = 2
    execution_payload.transactions = [Transaction(b'\x99' * 128) for _ in range(num_transactions)]
    execution_payload.block_hash = compute_el_block_hash(spec, execution_payload, state)
    yield from run_execution_payload_processing(spec, state, execution_payload)
    assert (
        state.latest_execution_payload_header.transactions_root
        == execution_payload.transactions.hash_tree_root()
    )


@with_bellatrix_and_later
@spec_state_test
def test_non_empty_transactions_first_payload(spec: Spec, state: State) -> Generator[Tuple[str, Any], None, None]:
    state = build_state_with_incomplete_transition(spec, state)
    yield from run_non_empty_transactions_test(spec, state)


@with_bellatrix_and_later
@spec_state_test
def test_non_empty_transactions_regular_payload(spec: Spec, state: State) -> Generator[Tuple[str, Any], None, None]:
    state = build_state_with_complete_transition(spec, state)
    yield from run_non_empty_transactions_test(spec, state)


def run_zero_length_transaction_test(
    spec: Spec, state: State
) -> Generator[Tuple[str, Any], None, None]:
    next_slot(spec, state)
    execution_payload: ExecutionPayload = build_empty_execution_payload(spec, state)
    execution_payload.transactions = [Transaction(b'')]
    assert len(execution_payload.transactions[0]) == 0
    execution_payload.block_hash = compute_el_block_hash(spec, execution_payload, state)
    yield from run_execution_payload_processing(spec, state, execution_payload)
    assert (
        state.latest_execution_payload_header.transactions_root
        == execution_payload.transactions.hash_tree_root()
    )


@with_bellatrix_and_later
@spec_state_test
def test_zero_length_transaction_first_payload(spec: Spec, state: State) -> Generator[Tuple[str, Any], None, None]:
    state = build_state_with_incomplete_transition(spec, state)
    yield from run_zero_length_transaction_test(spec, state)


@with_bellatrix_and_later
@spec_state_test
def test_zero_length_transaction_regular_payload(spec: Spec, state: State) -> Generator[Tuple[str, Any], None, None]:
    state = build_state_with_complete_transition(spec, state)
    yield from run_zero_length_transaction_test(spec, state)


def run_randomized_non_validated_execution_fields_test(
    spec: Spec,
    state: State,
    rng: Random,
    execution_valid: bool = True,
) -> Generator[Tuple[str, Any], None, None]:
    next_slot(spec, state)
    execution_payload: ExecutionPayload = build_randomized_execution_payload(spec, state, rng)
    yield from run_execution_payload_processing(
        spec, state, execution_payload, valid=execution_valid, execution_valid=execution_valid
    )


@with_bellatrix_and_later
@spec_state_test
def test_randomized_non_validated_execution_fields_first_payload__execution_valid(
    spec: Spec, state: State
) -> Generator[Tuple[str, Any], None, None]:
    rng: Random = Random(1111)
    state = build_state_with_incomplete_transition(spec, state)
    yield from run_randomized_non_validated_execution_fields_test(spec, state, rng)


@with_bellatrix_and_later
@spec_state_test
def test_randomized_non_validated_execution_fields_regular_payload__execution_valid(
    spec: Spec, state: State
) -> Generator[Tuple[str, Any], None, None]:
    rng: Random = Random(2222)
    state = build_state_with_complete_transition(spec, state)
    yield from run_randomized_non_validated_execution_fields_test(spec, state, rng)


@with_bellatrix_and_later
@spec_state_test
def test_invalid_randomized_non_validated_execution_fields_first_payload__execution_invalid(
    spec: Spec, state: State
) -> Generator[Tuple[str, Any], None, None]:
    rng: Random = Random(3333)
    state = build_state_with_incomplete_transition(spec, state)
    yield from run_randomized_non_validated_execution_fields_test(spec, state, rng, execution_valid=False)


@with_bellatrix_and_later
@spec_state_test
def test_invalid_randomized_non_validated_execution_fields_regular_payload__execution_invalid(
    spec: Spec, state: State
) -> Generator[Tuple[str, Any], None, None]:
    rng: Random = Random(4444)
    state = build_state_with_complete_transition(spec, state)
    yield from run_randomized_non_validated_execution_fields_test(spec, state, rng, execution_valid=False)
