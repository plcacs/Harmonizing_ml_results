import uuid
from typing import Any, Dict, List, Optional, Union
from unittest.mock import MagicMock, Mock
from httpx import Response as httpx_Response
from pendulum import DateTime
from prefect import flow
from prefect.client.schemas import FlowRun
from prefect.client.schemas.objects import Integration, StateType, WorkerMetadata
from prefect.exceptions import CrashedRun, ObjectNotFound
from prefect.server.schemas.core import WorkPool
from prefect.server.schemas.responses import DeploymentResponse
from prefect.settings import PREFECT_API_URL, PREFECT_TEST_MODE, PREFECT_WORKER_PREFETCH_SECONDS
from prefect.states import Completed, Failed, Pending, Running, Scheduled
from prefect.workers.base import BaseWorker

class WorkerTestImpl(BaseWorker):
    type: str = 'test'
    job_configuration: Any = BaseJobConfiguration

    async def run(self) -> None:
        ...

@pytest.fixture
async def variables(prefect_client: Any) -> None:
    ...

@pytest.fixture
def no_api_url() -> None:
    ...

async def test_worker_requires_api_url_when_not_in_test_mode(no_api_url: Any) -> None:
    ...

async def test_worker_creates_work_pool_by_default_during_sync(prefect_client: Any) -> None:
    ...

async def test_worker_does_not_creates_work_pool_when_create_pool_is_false(prefect_client: Any) -> None:
    ...

@pytest.mark.parametrize('setting,attr', [(PREFECT_WORKER_PREFETCH_SECONDS, 'prefetch_seconds')])
async def test_worker_respects_settings(setting: Any, attr: str) -> None:
    ...

async def test_worker_sends_heartbeat_messages(prefect_client: Any) -> None:
    ...

async def test_worker_sends_heartbeat_gets_id(respx_mock: Any) -> None:
    ...

async def test_worker_sends_heartbeat_only_gets_id_once() -> None:
    ...

async def test_worker_with_work_pool(prefect_client: Any, worker_deployment_wq1: Any, work_pool: WorkPool) -> None:
    ...

async def test_worker_with_work_pool_and_work_queue(prefect_client: Any, worker_deployment_wq1: Any, worker_deployment_wq_2: Any, work_queue_1: Any, work_pool: WorkPool) -> None:
    ...

async def test_workers_do_not_submit_flow_runs_awaiting_retry(prefect_client: Any, work_queue_1: Any, work_pool: WorkPool) -> None:
    ...

async def test_priority_trumps_lateness(prefect_client: Any, worker_deployment_wq1: Any, worker_deployment_wq_2: Any, work_queue_1: Any, work_pool: WorkPool) -> None:
    ...

async def test_worker_releases_limit_slot_when_aborting_a_change_to_pending(prefect_client: Any, worker_deployment_wq1: Any, work_pool: WorkPool) -> None:
    ...

async def test_worker_with_work_pool_and_limit(prefect_client: Any, worker_deployment_wq1: Any, work_pool: WorkPool) -> None:
    ...

async def test_worker_calls_run_with_expected_arguments(prefect_client: Any, worker_deployment_wq1: Any, work_pool: WorkPool, monkeypatch: Any) -> None:
    ...

async def test_worker_warns_when_running_a_flow_run_with_a_storage_block(prefect_client: Any, deployment: Any, work_pool: WorkPool, caplog: Any) -> None:
    ...

async def test_worker_creates_only_one_client_context(prefect_client: Any, worker_deployment_wq1: Any, work_pool: WorkPool, monkeypatch: Any, caplog: Any) -> None:
    ...

async def test_base_worker_gets_job_configuration_when_syncing_with_backend_with_just_job_config(session: Any, client: Any) -> None:
    ...

async def test_base_worker_gets_job_configuration_when_syncing_with_backend_with_job_config_and_variables(session: Any, client: Any) -> None:
    ...

@pytest.mark.parametrize('template,overrides,expected', [
    ({'job_configuration': {'command': '{{ command }}', 'env': '{{ env }}', 'labels': '{{ labels }}', 'name': '{{ name }}'}, 'variables': {'properties': {'command': {'type': 'string', 'title': 'Command', 'default': 'echo hello'}, 'env': {'title': 'Environment Variables', 'type': 'object', 'additionalProperties': {'type': 'string'}, 'description': 'Environment variables to set when starting a flow run.'}}, 'type': 'object'}}, {}, {'command': 'echo hello', 'env': {}, 'labels': {}, 'name': None}),
])
async def test_base_job_configuration_from_template_and_overrides(template: Dict, overrides: Dict, expected: Dict) -> None:
    ...

@pytest.mark.parametrize('template,overrides,expected', [
    ({'job_configuration': {'var1': '{{ var1 }}', 'var2': '{{ var2 }}'}, 'variables': {'properties': {'var1': {'type': 'string', 'title': 'Var1', 'default': 'hello'}, 'var2': {'type': 'integer', 'title': 'Var2', 'default': 42}}, 'required': []}}, {}, {'command': None, 'env': {}, 'labels': {}, 'name': None, 'var1': 'hello', 'var2': 42}),
])
async def test_job_configuration_from_template_and_overrides(template: Dict, overrides: Dict, expected: Dict) -> None:
    ...

async def test_job_configuration_from_template_and_overrides_with_nested_variables() -> None:
    ...

async def test_job_configuration_from_template_and_overrides_with_hard_coded_primitives() -> None:
    ...

async def test_job_configuration_from_template_overrides_with_block() -> None:
    ...

async def test_job_configuration_from_template_coerces_work_pool_values() -> None:
    ...

@pytest.mark.usefixtures('variables')
async def test_job_configuration_from_template_overrides_with_remote_variables() -> None:
    ...

@pytest.mark.usefixtures('variables')
async def test_job_configuration_from_template_overrides_with_remote_variables_hardcodes() -> None:
    ...

async def test_job_configuration_from_template_and_overrides_with_variables_in_a_list() -> None:
    ...

@pytest.mark.parametrize('falsey_value', [None, ''])
async def test_base_job_configuration_converts_falsey_values_to_none(falsey_value: Union[None, str]) -> None:
    ...

@pytest.mark.parametrize('field_template_value,expected_final_template', [
    ('{{ var1 }}', {'command': '{{ command }}', 'env': '{{ env }}', 'labels': '{{ labels }}', 'name': '{{ name }}', 'var1': '{{ var1 }}', 'var2': '{{ var2 }}'}),
    (None, {'command': '{{ command }}', 'env': '{{ env }}', 'labels': '{{ labels }}', 'name': '{{ name }}', 'var1': '{{ var1 }}', 'var2': '{{ var2 }}'}),
    ('{{ dog }}', {'command': '{{ command }}', 'env': '{{ env }}', 'labels': '{{ labels }}', 'name': '{{ name }}', 'var1': '{{ dog }}', 'var2': '{{ var2 }}'}),
])
def test_job_configuration_produces_correct_json_template(field_template_value: Optional[str], expected_final_template: Dict) -> None:
    ...

class TestWorkerProperties:
    def test_defaults() -> None:
        ...

    def test_custom_logo_url() -> None:
        ...

    def test_custom_documentation_url() -> None:
        ...

    def test_custom_description() -> None:
        ...

    def test_custom_base_job_configuration() -> None:
        ...

class TestPrepareForFlowRun:
    def test_prepare_for_flow_run_without_deployment_and_flow(job_config: Any, flow_run: FlowRun) -> None:
        ...

    def test_prepare_for_flow_run(job_config: Any, flow_run: FlowRun) -> None:
        ...

    def test_prepare_for_flow_run_with_deployment_and_flow(job_config: Any, flow_run: FlowRun, deployment: DeploymentResponse, flow: Any) -> None:
        ...

async def test_get_flow_run_logger_without_worker_id_set(prefect_client: Any, worker_deployment_wq1: Any, work_pool: WorkPool) -> None:
    ...

async def test_get_flow_run_logger_with_worker_id_set(prefect_client: Any, worker_deployment_wq1: Any, work_pool: WorkPool) -> None:
    ...

class TestInfrastructureIntegration:
    async def test_worker_crashes_flow_if_infrastructure_submission_fails(self: Any, prefect_client: Any, worker_deployment_infra_wq1: Any, work_pool: WorkPool, monkeypatch: Any) -> None:
        ...

async def test_worker_set_last_polled_time(work_pool: WorkPool) -> None:
    ...

async def test_worker_last_polled_health_check(work_pool: WorkPool) -> None:
    ...

class TestBaseWorkerStart:
    async def test_start_syncs_with_the_server(self: Any, work_pool: WorkPool) -> None:
        ...

    async def test_start_executes_flow_runs(self: Any, prefect_client: Any, worker_deployment_wq1: Any, work_pool: WorkPool) -> None:
        ...

@pytest.mark.parametrize('work_pool_env, deployment_env, flow_run_env, expected_env', [
    ({}, {'test-var': 'foo'}, {'another-var': 'boo'}, {'test-var': 'foo', 'another-var': 'boo'}),
])
@pytest.mark.parametrize('use_variable_defaults', [True, False])
async def test_env_merge_logic_is_deep(prefect_client: Any, session: Any, flow: Any, work_pool: WorkPool, work_pool_env: Dict, deployment_env: Dict, flow_run_env: Dict, expected_env: Dict, use_variable_defaults: bool) -> None:
    ...

class TestBaseWorkerHeartbeat:
    async def test_worker_heartbeat_sends_integrations(self: Any, work_pool: WorkPool, hosted_api_server: Any) -> None:
        ...

    async def test_custom_worker_can_send_arbitrary_metadata(self: Any, work_pool: WorkPool, hosted_api_server: Any) -> None:
        ...

async def test_worker_gives_labels_to_flow_runs_when_using_cloud_api(prefect_client: Any, worker_deployment_wq1: Any, work_pool: WorkPool) -> None:
    ...

async def test_worker_removes_flow_run_from_submitting_when_not_ready(prefect_client: Any, worker_deployment_wq1: Any, work_pool: WorkPool) -> None:
    ...