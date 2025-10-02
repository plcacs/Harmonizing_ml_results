from random import Random
from eth2spec.test.helpers.execution_payload import build_empty_execution_payload, compute_el_block_hash, get_execution_payload_header
from eth2spec.test.context import spec_state_test, expect_assertion_error, with_deneb_and_later
from eth2spec.test.helpers.blob import get_sample_blob_tx, get_max_blob_count
from typing import Iterator, Tuple, Any

def run_execution_payload_processing(spec: Any, state: Any, execution_payload: Any, blob_kzg_commitments: Any, valid: bool = True, execution_valid: bool = True) -> Iterator[Tuple[str, Any]]:
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

        def verify_and_notify_new_payload(self, new_payload_request: Any) -> bool:
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
