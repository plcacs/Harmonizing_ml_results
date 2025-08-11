from random import Random
from eth2spec.test.helpers.execution_payload import build_empty_execution_payload, compute_el_block_hash, get_execution_payload_header
from eth2spec.test.context import spec_state_test, expect_assertion_error, with_deneb_and_later
from eth2spec.test.helpers.blob import get_sample_blob_tx, get_max_blob_count

def run_execution_payload_processing(spec: Union[dict, bool, str], state: Union[dict, bool, str], execution_payload: Union[bool, dict[str, typing.Any], str], blob_kzg_commitments: Union[bool, list[Exception]], valid: bool=True, execution_valid: bool=True) -> Union[typing.Generator[tuple[typing.Union[typing.Text,dict,bool]]], typing.Generator[tuple[typing.Union[typing.Text,dict[typing.Text, bool]]]], typing.Generator[tuple[typing.Union[typing.Text,typing.Callable[None,None, tuple[typing.Any]],dict[str, str]]]], typing.Generator[tuple[typing.Optional[typing.Text]]], None]:
    """
    Run ``process_execution_payload``, yielding:
      - pre-state ('pre')
      - execution payload ('execution_payload')
      - execution details, to mock EVM execution ('execution.yml', a dict with 'execution_valid' key and boolean value)
      - post-state ('post').
    If ``valid == False``, run expecting ``AssertionError``
    """
    body = spec.BeaconBlockBody(blob_kzg_commitments=blob_kzg_commitments, execution_payload=execution_payload)
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
'\nTests with incorrect blob transactions in the execution payload, but the execution client returns\nVALID, and the purpose of these tests is that the beacon client must not reject the block by\nattempting to do a validation of its own.\n'

@with_deneb_and_later
@spec_state_test
def test_incorrect_blob_tx_type(spec: Union[dict, typing.Mapping, list[dict[str, str]]], state: Union[dict, typing.Mapping]) -> typing.Generator:
    """
    The versioned hashes are wrong, but the testing ExecutionEngine returns VALID by default.
    """
    execution_payload = build_empty_execution_payload(spec, state)
    opaque_tx, _, blob_kzg_commitments, _ = get_sample_blob_tx(spec)
    opaque_tx = b'\x04' + opaque_tx[1:]
    execution_payload.transactions = [opaque_tx]
    execution_payload.block_hash = compute_el_block_hash(spec, execution_payload, state)
    yield from run_execution_payload_processing(spec, state, execution_payload, blob_kzg_commitments)

@with_deneb_and_later
@spec_state_test
def test_incorrect_transaction_length_1_extra_byte(spec: Union[str, dict], state: Union[list, dict, bytes]) -> typing.Generator:
    """
    The versioned hashes are wrong, but the testing ExecutionEngine returns VALID by default.
    """
    execution_payload = build_empty_execution_payload(spec, state)
    opaque_tx, _, blob_kzg_commitments, _ = get_sample_blob_tx(spec)
    opaque_tx = opaque_tx + b'\x12'
    execution_payload.transactions = [opaque_tx]
    execution_payload.block_hash = compute_el_block_hash(spec, execution_payload, state)
    yield from run_execution_payload_processing(spec, state, execution_payload, blob_kzg_commitments)

@with_deneb_and_later
@spec_state_test
def test_incorrect_transaction_length_1_byte_short(spec: Union[str, bytes, list[dict[str, str]]], state: Union[bytes, str, list[dict[str, str]]]) -> typing.Generator:
    """
    The versioned hashes are wrong, but the testing ExecutionEngine returns VALID by default.
    """
    execution_payload = build_empty_execution_payload(spec, state)
    opaque_tx, _, blob_kzg_commitments, _ = get_sample_blob_tx(spec)
    opaque_tx = opaque_tx[:-1]
    execution_payload.transactions = [opaque_tx]
    execution_payload.block_hash = compute_el_block_hash(spec, execution_payload, state)
    yield from run_execution_payload_processing(spec, state, execution_payload, blob_kzg_commitments)

@with_deneb_and_later
@spec_state_test
def test_incorrect_transaction_length_empty(spec: Union[bytes, dict, list[dict[str, str]]], state: Union[bytes, list, dict]) -> typing.Generator:
    """
    The versioned hashes are wrong, but the testing ExecutionEngine returns VALID by default.
    """
    execution_payload = build_empty_execution_payload(spec, state)
    opaque_tx, _, blob_kzg_commitments, _ = get_sample_blob_tx(spec)
    opaque_tx = opaque_tx[0:0]
    execution_payload.transactions = [opaque_tx]
    execution_payload.block_hash = compute_el_block_hash(spec, execution_payload, state)
    yield from run_execution_payload_processing(spec, state, execution_payload, blob_kzg_commitments)

@with_deneb_and_later
@spec_state_test
def test_incorrect_transaction_length_32_extra_bytes(spec: Union[str, dict], state: Union[list, dict, bytes]) -> typing.Generator:
    """
    The versioned hashes are wrong, but the testing ExecutionEngine returns VALID by default.
    """
    execution_payload = build_empty_execution_payload(spec, state)
    opaque_tx, _, blob_kzg_commitments, _ = get_sample_blob_tx(spec)
    opaque_tx = opaque_tx + b'\x12' * 32
    execution_payload.transactions = [opaque_tx]
    execution_payload.block_hash = compute_el_block_hash(spec, execution_payload, state)
    yield from run_execution_payload_processing(spec, state, execution_payload, blob_kzg_commitments)

@with_deneb_and_later
@spec_state_test
def test_no_transactions_with_commitments(spec: Union[dict[str, typing.Any], str], state: Union[bytes, dict]) -> typing.Generator:
    """
    The versioned hashes are wrong, but the testing ExecutionEngine returns VALID by default.
    """
    execution_payload = build_empty_execution_payload(spec, state)
    _, _, blob_kzg_commitments, _ = get_sample_blob_tx(spec)
    execution_payload.transactions = []
    execution_payload.block_hash = compute_el_block_hash(spec, execution_payload, state)
    yield from run_execution_payload_processing(spec, state, execution_payload, blob_kzg_commitments)

@with_deneb_and_later
@spec_state_test
def test_incorrect_commitment(spec: Union[str, dict[str, typing.Any]], state: str) -> typing.Generator:
    """
    The versioned hashes are wrong, but the testing ExecutionEngine returns VALID by default.
    """
    execution_payload = build_empty_execution_payload(spec, state)
    opaque_tx, _, blob_kzg_commitments, _ = get_sample_blob_tx(spec)
    blob_kzg_commitments[0] = b'\x12' * 48
    execution_payload.transactions = [opaque_tx]
    execution_payload.block_hash = compute_el_block_hash(spec, execution_payload, state)
    yield from run_execution_payload_processing(spec, state, execution_payload, blob_kzg_commitments)

@with_deneb_and_later
@spec_state_test
def test_no_commitments_for_transactions(spec: bytes, state: int) -> typing.Generator:
    """
    The versioned hashes are wrong, but the testing ExecutionEngine returns VALID by default.
    """
    execution_payload = build_empty_execution_payload(spec, state)
    opaque_tx, _, blob_kzg_commitments, _ = get_sample_blob_tx(spec, blob_count=2, rng=Random(1111))
    blob_kzg_commitments = []
    execution_payload.transactions = [opaque_tx]
    execution_payload.block_hash = compute_el_block_hash(spec, execution_payload, state)
    yield from run_execution_payload_processing(spec, state, execution_payload, blob_kzg_commitments)

@with_deneb_and_later
@spec_state_test
def test_incorrect_commitments_order(spec: Union[list, dict, bytes], state: Union[bytes, dict]) -> typing.Generator:
    """
    The versioned hashes are wrong, but the testing ExecutionEngine returns VALID by default.
    """
    execution_payload = build_empty_execution_payload(spec, state)
    opaque_tx, _, blob_kzg_commitments, _ = get_sample_blob_tx(spec, blob_count=2, rng=Random(1111))
    blob_kzg_commitments = [blob_kzg_commitments[1], blob_kzg_commitments[0]]
    execution_payload.transactions = [opaque_tx]
    execution_payload.block_hash = compute_el_block_hash(spec, execution_payload, state)
    yield from run_execution_payload_processing(spec, state, execution_payload, blob_kzg_commitments)

@with_deneb_and_later
@spec_state_test
def test_incorrect_transaction_no_blobs_but_with_commitments(spec: Union[list[str], None, dict, bytes], state: Union[bytes, dict, tuple[typing.Union[dict,int]]]) -> typing.Generator:
    """
    The versioned hashes are wrong, but the testing ExecutionEngine returns VALID by default.
    """
    execution_payload = build_empty_execution_payload(spec, state)
    opaque_tx, _, _, _ = get_sample_blob_tx(spec, blob_count=0, rng=Random(1111))
    _, _, blob_kzg_commitments, _ = get_sample_blob_tx(spec, blob_count=2, rng=Random(1112))
    execution_payload.transactions = [opaque_tx]
    execution_payload.block_hash = compute_el_block_hash(spec, execution_payload, state)
    yield from run_execution_payload_processing(spec, state, execution_payload, blob_kzg_commitments)

@with_deneb_and_later
@spec_state_test
def test_incorrect_block_hash(spec: Any, state: Union[dict, str, bool]) -> typing.Generator:
    execution_payload = build_empty_execution_payload(spec, state)
    opaque_tx, _, blob_kzg_commitments, _ = get_sample_blob_tx(spec)
    execution_payload.transactions = [opaque_tx]
    execution_payload.block_hash = b'\x12' * 32
    yield from run_execution_payload_processing(spec, state, execution_payload, blob_kzg_commitments)

@with_deneb_and_later
@spec_state_test
def test_zeroed_commitment(spec: Union[str, typing.Sequence[int]], state: Any) -> typing.Generator:
    """
    The blob is invalid, but the commitment is in correct form.
    """
    execution_payload = build_empty_execution_payload(spec, state)
    opaque_tx, _, blob_kzg_commitments, _ = get_sample_blob_tx(spec, blob_count=1, is_valid_blob=False)
    assert all((commitment == b'\x00' * 48 for commitment in blob_kzg_commitments))
    execution_payload.transactions = [opaque_tx]
    execution_payload.block_hash = compute_el_block_hash(spec, execution_payload, state)
    yield from run_execution_payload_processing(spec, state, execution_payload, blob_kzg_commitments)

@with_deneb_and_later
@spec_state_test
def test_invalid_correct_input__execution_invalid(spec: Union[dict[str, typing.Any], str, typing.Mapping], state: Union[dict, tuple, Message]) -> typing.Generator:
    """
    The versioned hashes are wrong, but the testing ExecutionEngine returns VALID by default.
    """
    execution_payload = build_empty_execution_payload(spec, state)
    opaque_tx, _, blob_kzg_commitments, _ = get_sample_blob_tx(spec)
    execution_payload.transactions = [opaque_tx]
    execution_payload.block_hash = compute_el_block_hash(spec, execution_payload, state)
    yield from run_execution_payload_processing(spec, state, execution_payload, blob_kzg_commitments, valid=False, execution_valid=False)

@with_deneb_and_later
@spec_state_test
def test_invalid_exceed_max_blobs_per_block(spec: int, state: dict[str, typing.Any]) -> typing.Generator:
    execution_payload = build_empty_execution_payload(spec, state)
    opaque_tx, _, blob_kzg_commitments, _ = get_sample_blob_tx(spec, blob_count=get_max_blob_count(spec) + 1)
    execution_payload.transactions = [opaque_tx]
    execution_payload.block_hash = compute_el_block_hash(spec, execution_payload, state)
    yield from run_execution_payload_processing(spec, state, execution_payload, blob_kzg_commitments, valid=False)