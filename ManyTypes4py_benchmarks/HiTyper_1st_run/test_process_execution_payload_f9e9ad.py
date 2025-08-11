from random import Random
from eth2spec.test.helpers.execution_payload import build_empty_execution_payload, build_randomized_execution_payload, compute_el_block_hash, get_execution_payload_header, build_state_with_incomplete_transition, build_state_with_complete_transition
from eth2spec.test.context import BELLATRIX, expect_assertion_error, spec_state_test, with_bellatrix_and_later, with_phases
from eth2spec.test.helpers.state import next_slot

def run_execution_payload_processing(spec: Union[str, typing.Callable], state: Union[str, typing.Callable], execution_payload: Union[dict[str, typing.Any], bool, list], valid: bool=True, execution_valid: bool=True) -> Union[typing.Generator[tuple[typing.Union[typing.Text,typing.Callable]]], typing.Generator[tuple[typing.Union[typing.Text,dict[typing.Text, bool]]]], typing.Generator[tuple[typing.Union[typing.Text,typing.Callable[None,None, tuple[typing.Any]],dict[str, str]]]], typing.Generator[tuple[typing.Optional[typing.Text]]], None]:
    """
    Run ``process_execution_payload``, yielding:
      - pre-state ('pre')
      - execution payload ('execution_payload')
      - execution details, to mock EVM execution ('execution.yml', a dict with 'execution_valid' key and boolean value)
      - post-state ('post').
    If ``valid == False``, run expecting ``AssertionError``
    """
    body = spec.BeaconBlockBody(execution_payload=execution_payload)
    yield ('pre', state)
    yield ('execution', {'execution_valid': execution_valid})
    yield ('body', body)
    called_new_block = False

    class TestEngine(spec.NoopExecutionEngine):

        def verify_and_notify_new_payload(self, new_payload_request: Any):
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

def run_success_test(spec: Union[mythril.laser.ethereum.state.global_state.GlobalState, dict, pyshgp.push.state.PushState], state: Union[mythril.laser.ethereum.state.global_state.GlobalState, dict, pyshgp.push.state.PushState]) -> typing.Generator:
    next_slot(spec, state)
    execution_payload = build_empty_execution_payload(spec, state)
    yield from run_execution_payload_processing(spec, state, execution_payload)

@with_bellatrix_and_later
@spec_state_test
def test_success_first_payload(spec: dict, state: dict) -> typing.Generator:
    state = build_state_with_incomplete_transition(spec, state)
    yield from run_success_test(spec, state)

@with_bellatrix_and_later
@spec_state_test
def test_success_regular_payload(spec: Union[mythril.laser.ethereum.state.global_state.GlobalState, dict, typing.Mapping], state: Union[mythril.laser.ethereum.state.global_state.GlobalState, dict, typing.Mapping]) -> typing.Generator:
    state = build_state_with_complete_transition(spec, state)
    yield from run_success_test(spec, state)

def run_gap_slot_test(spec: Union[mythril.laser.ethereum.state.global_state.GlobalState, pyshgp.push.state.PushState, dict], state: Union[mythril.laser.ethereum.state.global_state.GlobalState, pyshgp.push.state.PushState, dict]) -> typing.Generator:
    next_slot(spec, state)
    next_slot(spec, state)
    execution_payload = build_empty_execution_payload(spec, state)
    yield from run_execution_payload_processing(spec, state, execution_payload)

@with_bellatrix_and_later
@spec_state_test
def test_success_first_payload_with_gap_slot(spec: Union[mythril.laser.ethereum.state.global_state.GlobalState, prefecengine.state.State, dict], state: Union[mythril.laser.ethereum.state.global_state.GlobalState, prefecengine.state.State, dict]) -> typing.Generator:
    state = build_state_with_incomplete_transition(spec, state)
    yield from run_gap_slot_test(spec, state)

@with_bellatrix_and_later
@spec_state_test
def test_success_regular_payload_with_gap_slot(spec: Union[mythril.laser.ethereum.state.global_state.GlobalState, prefecengine.state.State, raiden.transfer.state.ChainState], state: Union[mythril.laser.ethereum.state.global_state.GlobalState, prefecengine.state.State, raiden.transfer.state.ChainState]) -> typing.Generator:
    state = build_state_with_complete_transition(spec, state)
    yield from run_gap_slot_test(spec, state)

def run_bad_execution_test(spec: Union[mythril.laser.ethereum.state.global_state.GlobalState, dict, prefecengine.state.State], state: Union[mythril.laser.ethereum.state.global_state.GlobalState, dict, prefecengine.state.State]) -> typing.Generator:
    next_slot(spec, state)
    execution_payload = build_empty_execution_payload(spec, state)
    yield from run_execution_payload_processing(spec, state, execution_payload, valid=False, execution_valid=False)

@with_bellatrix_and_later
@spec_state_test
def test_invalid_bad_execution_first_payload(spec: Union[dict, mythril.laser.ethereum.state.global_state.GlobalState, str], state: Union[dict, mythril.laser.ethereum.state.global_state.GlobalState, str]) -> typing.Generator:
    state = build_state_with_incomplete_transition(spec, state)
    yield from run_bad_execution_test(spec, state)

@with_bellatrix_and_later
@spec_state_test
def test_invalid_bad_execution_regular_payload(spec: Union[mythril.laser.ethereum.state.global_state.GlobalState, dict, str], state: Union[mythril.laser.ethereum.state.global_state.GlobalState, dict, str]) -> typing.Generator:
    state = build_state_with_complete_transition(spec, state)
    yield from run_bad_execution_test(spec, state)

@with_phases([BELLATRIX])
@spec_state_test
def test_bad_parent_hash_first_payload(spec: Union[dict[str, typing.Any], typing.Mapping, dict[str, dict], None], state: Union[dict[str, typing.Any], typing.Mapping, dict[str, dict], None]) -> typing.Generator:
    state = build_state_with_incomplete_transition(spec, state)
    next_slot(spec, state)
    execution_payload = build_empty_execution_payload(spec, state)
    execution_payload.parent_hash = b'U' * 32
    execution_payload.block_hash = compute_el_block_hash(spec, execution_payload, state)
    yield from run_execution_payload_processing(spec, state, execution_payload)

@with_bellatrix_and_later
@spec_state_test
def test_invalid_bad_parent_hash_regular_payload(spec: Union[dict, typing.Mapping, str], state: Union[dict, typing.Mapping, str]) -> typing.Generator:
    state = build_state_with_complete_transition(spec, state)
    next_slot(spec, state)
    execution_payload = build_empty_execution_payload(spec, state)
    execution_payload.parent_hash = spec.Hash32()
    execution_payload.block_hash = compute_el_block_hash(spec, execution_payload, state)
    yield from run_execution_payload_processing(spec, state, execution_payload, valid=False)

def run_bad_prev_randao_test(spec: Union[raiden.transfer.state.ChainState, apps.monero.signing.state.State], state: Union[raiden.transfer.state.ChainState, apps.monero.signing.state.State]) -> typing.Generator:
    next_slot(spec, state)
    execution_payload = build_empty_execution_payload(spec, state)
    execution_payload.prev_randao = b'B' * 32
    execution_payload.block_hash = compute_el_block_hash(spec, execution_payload, state)
    yield from run_execution_payload_processing(spec, state, execution_payload, valid=False)

@with_bellatrix_and_later
@spec_state_test
def test_invalid_bad_prev_randao_first_payload(spec: Union[dict[str, typing.Any], prefecengine.state.State, dict["core.Edge", "state.State"]], state: Union[dict[str, typing.Any], prefecengine.state.State, dict["core.Edge", "state.State"]]) -> typing.Generator:
    state = build_state_with_incomplete_transition(spec, state)
    yield from run_bad_prev_randao_test(spec, state)

@with_bellatrix_and_later
@spec_state_test
def test_invalid_bad_pre_randao_regular_payload(spec: Union[mythril.laser.ethereum.state.global_state.GlobalState, dict, str], state: Union[mythril.laser.ethereum.state.global_state.GlobalState, dict, str]) -> typing.Generator:
    state = build_state_with_complete_transition(spec, state)
    yield from run_bad_prev_randao_test(spec, state)

def run_bad_everything_test(spec: Union[dict, pyshgp.push.state.PushState, mythril.laser.ethereum.state.global_state.GlobalState], state: Union[dict[str, typing.Any], raiden.transfer.state.ChainState, apps.monero.signing.state.State]) -> typing.Generator:
    next_slot(spec, state)
    execution_payload = build_empty_execution_payload(spec, state)
    execution_payload.parent_hash = spec.Hash32()
    execution_payload.prev_randao = spec.Bytes32()
    execution_payload.timestamp = 0
    execution_payload.block_hash = compute_el_block_hash(spec, execution_payload, state)
    yield from run_execution_payload_processing(spec, state, execution_payload, valid=False)

@with_bellatrix_and_later
@spec_state_test
def test_invalid_bad_everything_first_payload(spec: Union[dict, mythril.laser.ethereum.state.global_state.GlobalState, list[mythril.laser.ethereum.state.global_state.GlobalState]], state: Union[dict, mythril.laser.ethereum.state.global_state.GlobalState, list[mythril.laser.ethereum.state.global_state.GlobalState]]) -> typing.Generator:
    state = build_state_with_incomplete_transition(spec, state)
    yield from run_bad_everything_test(spec, state)

@with_bellatrix_and_later
@spec_state_test
def test_invalid_bad_everything_regular_payload(spec: Union[mythril.laser.ethereum.state.global_state.GlobalState, typing.Mapping, str], state: Union[mythril.laser.ethereum.state.global_state.GlobalState, typing.Mapping, str]) -> typing.Generator:
    state = build_state_with_complete_transition(spec, state)
    yield from run_bad_everything_test(spec, state)

def run_bad_timestamp_test(spec: Union[typing.Callable[dict, None], None, raiden.transfer.state.NettingChannelEndState], state: Union[typing.Callable[dict, None], None, raiden.transfer.state.NettingChannelEndState], is_future: Union[bool, typing.Mapping, list[Exception]]) -> typing.Generator:
    next_slot(spec, state)
    execution_payload = build_empty_execution_payload(spec, state)
    if is_future:
        timestamp = execution_payload.timestamp + 1
    else:
        timestamp = execution_payload.timestamp - 1
    execution_payload.timestamp = timestamp
    execution_payload.block_hash = compute_el_block_hash(spec, execution_payload, state)
    yield from run_execution_payload_processing(spec, state, execution_payload, valid=False)

@with_bellatrix_and_later
@spec_state_test
def test_invalid_future_timestamp_first_payload(spec: Union[typing.Mapping, dict[str, dict], None, dict], state: Union[typing.Mapping, dict[str, dict], None, dict]) -> typing.Generator:
    state = build_state_with_incomplete_transition(spec, state)
    yield from run_bad_timestamp_test(spec, state, is_future=True)

@with_bellatrix_and_later
@spec_state_test
def test_invalid_future_timestamp_regular_payload(spec: Union[mythril.laser.ethereum.state.global_state.GlobalState, prefecengine.state.State, str], state: Union[mythril.laser.ethereum.state.global_state.GlobalState, prefecengine.state.State, str]) -> typing.Generator:
    state = build_state_with_complete_transition(spec, state)
    yield from run_bad_timestamp_test(spec, state, is_future=True)

@with_bellatrix_and_later
@spec_state_test
def test_invalid_past_timestamp_first_payload(spec: Union[dict[str, typing.Any], typing.Mapping, list[mythril.laser.ethereum.state.global_state.GlobalState]], state: Union[dict[str, typing.Any], typing.Mapping, list[mythril.laser.ethereum.state.global_state.GlobalState]]) -> typing.Generator:
    state = build_state_with_incomplete_transition(spec, state)
    yield from run_bad_timestamp_test(spec, state, is_future=False)

@with_bellatrix_and_later
@spec_state_test
def test_invalid_past_timestamp_regular_payload(spec: Union[mythril.laser.ethereum.state.global_state.GlobalState, prefecengine.state.State, str], state: Union[mythril.laser.ethereum.state.global_state.GlobalState, prefecengine.state.State, str]) -> typing.Generator:
    state = build_state_with_complete_transition(spec, state)
    yield from run_bad_timestamp_test(spec, state, is_future=False)

def run_non_empty_extra_data_test(spec: Union[mythril.laser.ethereum.state.global_state.GlobalState, dict, state.State], state: Union[mythril.laser.ethereum.state.global_state.GlobalState, dict, state.State]) -> typing.Generator:
    next_slot(spec, state)
    execution_payload = build_empty_execution_payload(spec, state)
    execution_payload.extra_data = b'E' * 12
    execution_payload.block_hash = compute_el_block_hash(spec, execution_payload, state)
    yield from run_execution_payload_processing(spec, state, execution_payload)
    assert state.latest_execution_payload_header.extra_data == execution_payload.extra_data

@with_bellatrix_and_later
@spec_state_test
def test_non_empty_extra_data_first_payload(spec: Union[dict[str, typing.Any], dict, mythril.laser.ethereum.state.global_state.GlobalState], state: Union[dict[str, typing.Any], dict, mythril.laser.ethereum.state.global_state.GlobalState]) -> typing.Generator:
    state = build_state_with_incomplete_transition(spec, state)
    yield from run_non_empty_extra_data_test(spec, state)

@with_bellatrix_and_later
@spec_state_test
def test_non_empty_extra_data_regular_payload(spec: Union[dict[str, typing.Any], mythril.laser.ethereum.state.global_state.GlobalState, typing.Mapping], state: Union[dict[str, typing.Any], mythril.laser.ethereum.state.global_state.GlobalState, typing.Mapping]) -> typing.Generator:
    state = build_state_with_complete_transition(spec, state)
    yield from run_non_empty_extra_data_test(spec, state)

def run_non_empty_transactions_test(spec: Union[mythril.laser.ethereum.state.global_state.GlobalState, dict, prefecengine.state.State], state: Union[mythril.laser.ethereum.state.global_state.GlobalState, dict, state.State]) -> typing.Generator:
    next_slot(spec, state)
    execution_payload = build_empty_execution_payload(spec, state)
    num_transactions = 2
    execution_payload.transactions = [spec.Transaction(b'\x99' * 128) for _ in range(num_transactions)]
    execution_payload.block_hash = compute_el_block_hash(spec, execution_payload, state)
    yield from run_execution_payload_processing(spec, state, execution_payload)
    assert state.latest_execution_payload_header.transactions_root == execution_payload.transactions.hash_tree_root()

@with_bellatrix_and_later
@spec_state_test
def test_non_empty_transactions_first_payload(spec: Union[dict[str, typing.Any], mythril.laser.ethereum.state.global_state.GlobalState, str], state: Union[dict[str, typing.Any], mythril.laser.ethereum.state.global_state.GlobalState, str]) -> typing.Generator:
    state = build_state_with_incomplete_transition(spec, state)
    yield from run_non_empty_extra_data_test(spec, state)

@with_bellatrix_and_later
@spec_state_test
def test_non_empty_transactions_regular_payload(spec: Union[mythril.laser.ethereum.state.global_state.GlobalState, dict, str], state: Union[mythril.laser.ethereum.state.global_state.GlobalState, dict, str]) -> typing.Generator:
    state = build_state_with_complete_transition(spec, state)
    yield from run_non_empty_extra_data_test(spec, state)

def run_zero_length_transaction_test(spec: Union[mythril.laser.ethereum.state.global_state.GlobalState, dict, prefecengine.state.State], state: Union[mythril.laser.ethereum.state.global_state.GlobalState, dict, state.State]) -> typing.Generator:
    next_slot(spec, state)
    execution_payload = build_empty_execution_payload(spec, state)
    execution_payload.transactions = [spec.Transaction(b'')]
    assert len(execution_payload.transactions[0]) == 0
    execution_payload.block_hash = compute_el_block_hash(spec, execution_payload, state)
    yield from run_execution_payload_processing(spec, state, execution_payload)
    assert state.latest_execution_payload_header.transactions_root == execution_payload.transactions.hash_tree_root()

@with_bellatrix_and_later
@spec_state_test
def test_zero_length_transaction_first_payload(spec: Union[mythril.laser.ethereum.state.global_state.GlobalState, raiden.transfer.state.ChainState], state: Union[mythril.laser.ethereum.state.global_state.GlobalState, raiden.transfer.state.ChainState]) -> typing.Generator:
    state = build_state_with_incomplete_transition(spec, state)
    yield from run_zero_length_transaction_test(spec, state)

@with_bellatrix_and_later
@spec_state_test
def test_zero_length_transaction_regular_payload(spec: Union[mythril.laser.ethereum.state.global_state.GlobalState, prefecengine.state.State], state: Union[mythril.laser.ethereum.state.global_state.GlobalState, prefecengine.state.State]) -> typing.Generator:
    state = build_state_with_complete_transition(spec, state)
    yield from run_zero_length_transaction_test(spec, state)

def run_randomized_non_validated_execution_fields_test(spec: Union[prefecengine.state.State, bool], state: Union[prefecengine.state.State, bool], rng: Union[Atom, dict[str, typing.Any], None, bool], execution_valid: bool=True) -> typing.Generator:
    next_slot(spec, state)
    execution_payload = build_randomized_execution_payload(spec, state, rng)
    yield from run_execution_payload_processing(spec, state, execution_payload, valid=execution_valid, execution_valid=execution_valid)

@with_bellatrix_and_later
@spec_state_test
def test_randomized_non_validated_execution_fields_first_payload__execution_valid(spec: Union[dict[str, typing.Any], mythril.laser.ethereum.state.global_state.GlobalState, dict["core.Edge", "state.State"]], state: Union[dict[str, typing.Any], mythril.laser.ethereum.state.global_state.GlobalState, dict["core.Edge", "state.State"]]) -> typing.Generator:
    rng = Random(1111)
    state = build_state_with_incomplete_transition(spec, state)
    yield from run_randomized_non_validated_execution_fields_test(spec, state, rng)

@with_bellatrix_and_later
@spec_state_test
def test_randomized_non_validated_execution_fields_regular_payload__execution_valid(spec: Union[mythril.laser.ethereum.state.global_state.GlobalState, prefecengine.state.State, dict["core.Edge", "state.State"]], state: Union[mythril.laser.ethereum.state.global_state.GlobalState, prefecengine.state.State, dict["core.Edge", "state.State"]]) -> typing.Generator:
    rng = Random(2222)
    state = build_state_with_complete_transition(spec, state)
    yield from run_randomized_non_validated_execution_fields_test(spec, state, rng)

@with_bellatrix_and_later
@spec_state_test
def test_invalid_randomized_non_validated_execution_fields_first_payload__execution_invalid(spec: Union[mythril.laser.ethereum.state.global_state.GlobalState, prefecengine.state.State], state: Union[mythril.laser.ethereum.state.global_state.GlobalState, prefecengine.state.State]) -> typing.Generator:
    rng = Random(3333)
    state = build_state_with_incomplete_transition(spec, state)
    yield from run_randomized_non_validated_execution_fields_test(spec, state, rng, execution_valid=False)

@with_bellatrix_and_later
@spec_state_test
def test_invalid_randomized_non_validated_execution_fields_regular_payload__execution_invalid(spec: Union[dict[str, typing.Any], dict, mythril.laser.ethereum.state.global_state.GlobalState], state: Union[dict[str, typing.Any], dict, mythril.laser.ethereum.state.global_state.GlobalState]) -> typing.Generator:
    rng = Random(4444)
    state = build_state_with_complete_transition(spec, state)
    yield from run_randomized_non_validated_execution_fields_test(spec, state, rng, execution_valid=False)