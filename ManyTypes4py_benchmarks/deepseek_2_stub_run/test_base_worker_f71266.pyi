```python
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Type
from unittest.mock import AsyncMock, MagicMock, Mock
import httpx
import pendulum
import pytest
from packaging import version
from pydantic import Field
from starlette import status
import prefect
import prefect.client.schemas as schemas
from prefect.blocks.core import Block
from prefect.client.base import ServerType
from prefect.client.orchestration import PrefectClient
from prefect.client.schemas import FlowRun
from prefect.client.schemas.objects import Integration, StateType, WorkerMetadata
from prefect.exceptions import CrashedRun, ObjectNotFound
from prefect.flows import flow
from prefect.server import models
from prefect.server.schemas.actions import WorkPoolUpdate as ServerWorkPoolUpdate
from prefect.server.schemas.core import Deployment, Flow, WorkPool
from prefect.server.schemas.responses import DeploymentResponse
from prefect.settings import PREFECT_API_URL, PREFECT_TEST_MODE, PREFECT_WORKER_PREFETCH_SECONDS
from prefect.states import Completed, Failed, Pending, Running, Scheduled
from prefect.testing.utilities import AsyncMock
from prefect.utilities.pydantic import parse_obj_as
from prefect.workers.base import BaseJobConfiguration, BaseVariables, BaseWorker

class WorkerTestImpl(BaseWorker):
    type: str = ...
    job_configuration: Type[BaseJobConfiguration] = ...
    
    def __init__(
        self,
        name: str = ...,
        work_pool_name: str = ...,
        work_queues: Optional[List[str]] = ...,
        limit: Optional[int] = ...,
        create_pool_if_not_found: bool = ...,
        **kwargs: Any
    ) -> None: ...
    
    async def run(self) -> None: ...
    
    async def __aenter__(self) -> "WorkerTestImpl": ...
    async def __aexit__(self, *args: Any) -> None: ...

@pytest.fixture
async def ensure_default_agent_pool_exists(session: Any) -> AsyncGenerator[None, None]: ...

@pytest.fixture
async def variables(prefect_client: Any) -> AsyncGenerator[None, None]: ...

@pytest.fixture
def no_api_url() -> Any: ...

async def test_worker_requires_api_url_when_not_in_test_mode(no_api_url: Any) -> None: ...

async def test_worker_creates_work_pool_by_default_during_sync(prefect_client: Any) -> None: ...

async def test_worker_does_not_creates_work_pool_when_create_pool_is_false(prefect_client: Any) -> None: ...

@pytest.mark.parametrize('setting,attr', [(PREFECT_WORKER_PREFETCH_SECONDS, 'prefetch_seconds')])
async def test_worker_respects_settings(setting: Any, attr: str) -> None: ...

async def test_worker_sends_heartbeat_messages(prefect_client: Any) -> None: ...

async def test_worker_sends_heartbeat_gets_id(respx_mock: Any) -> None: ...

async def test_worker_sends_heartbeat_only_gets_id_once() -> None: ...

async def test_worker_with_work_pool(
    prefect_client: Any,
    worker_deployment_wq1: Any,
    work_pool: Any
) -> None: ...

async def test_worker_with_work_pool_and_work_queue(
    prefect_client: Any,
    worker_deployment_wq1: Any,
    worker_deployment_wq_2: Any,
    work_queue_1: Any,
    work_pool: Any
) -> None: ...

async def test_workers_do_not_submit_flow_runs_awaiting_retry(
    prefect_client: Any,
    work_queue_1: Any,
    work_pool: Any
) -> None: ...

async def test_priority_trumps_lateness(
    prefect_client: Any,
    worker_deployment_wq1: Any,
    worker_deployment_wq_2: Any,
    work_queue_1: Any,
    work_pool: Any
) -> None: ...

async def test_worker_releases_limit_slot_when_aborting_a_change_to_pending(
    prefect_client: Any,
    worker_deployment_wq1: Any,
    work_pool: Any
) -> None: ...

async def test_worker_with_work_pool_and_limit(
    prefect_client: Any,
    worker_deployment_wq1: Any,
    work_pool: Any
) -> None: ...

async def test_worker_calls_run_with_expected_arguments(
    prefect_client: Any,
    worker_deployment_wq1: Any,
    work_pool: Any,
    monkeypatch: Any
) -> None: ...

async def test_worker_warns_when_running_a_flow_run_with_a_storage_block(
    prefect_client: Any,
    deployment: Any,
    work_pool: Any,
    caplog: Any
) -> None: ...

async def test_worker_creates_only_one_client_context(
    prefect_client: Any,
    worker_deployment_wq1: Any,
    work_pool: Any,
    monkeypatch: Any,
    caplog: Any
) -> None: ...

async def test_base_worker_gets_job_configuration_when_syncing_with_backend_with_just_job_config(
    session: Any,
    client: Any
) -> None: ...

async def test_base_worker_gets_job_configuration_when_syncing_with_backend_with_job_config_and_variables(
    session: Any,
    client: Any
) -> None: ...

@pytest.mark.parametrize('template,overrides,expected', [({'job_configuration': {'command': '{{ command }}', 'env': '{{ env }}', 'labels': '{{ labels }}', 'name': '{{ name }}'}, 'variables': {'properties': {'command': {'type': 'string', 'title': 'Command', 'default': 'echo hello'}, 'env': {'title': 'Environment Variables', 'type': 'object', 'additionalProperties': {'type': 'string'}, 'description': 'Environment variables to set when starting a flow run.'}}, 'type': 'object'}}, {}, {'command': 'echo hello', 'env': {}, 'labels': {}, 'name': None})])
async def test_base_job_configuration_from_template_and_overrides(
    template: Dict[str, Any],
    overrides: Dict[str, Any],
    expected: Dict[str, Any]
) -> None: ...

@pytest.mark.parametrize('template,overrides,expected', [({'job_configuration': {'var1': '{{ var1 }}', 'var2': '{{ var2 }}'}, 'variables': {'properties': {'var1': {'type': 'string', 'title': 'Var1', 'default': 'hello'}, 'var2': {'type': 'integer', 'title': 'Var2', 'default': 42}}, 'required': []}}, {}, {'command': None, 'env': {}, 'labels': {}, 'name': None, 'var1': 'hello', 'var2': 42}), ({'job_configuration': {'var1': '{{ var1 }}', 'var2': '{{ var2 }}'}, 'variables': {'properties': {'var1': {'type': 'string', 'title': 'Var1', 'default': 'hello'}, 'var2': {'type': 'integer', 'title': 'Var2', 'default': 42}, 'var3': {'type': 'integer', 'title': 'Var3', 'default': 21}}, 'required': []}}, {}, {'command': None, 'env': {}, 'labels': {}, 'name': None, 'var1': 'hello', 'var2': 42}), ({'job_configuration': {'var1': '{{ var1 }}', 'var2': '{{ var2 }}'}, 'variables': {'properties': {'var1': {'type': 'string', 'title': 'Var1', 'default': 'hello'}, 'var2': {'type': 'integer', 'title': 'Var2', 'default': 42}, 'command': {'type': 'string', 'title': 'Command', 'default': 'echo hello'}}, 'required': []}}, {}, {'command': None, 'env': {}, 'labels': {}, 'name': None, 'var1': 'hello', 'var2': 42}), ({'job_configuration': {'var1': '{{ var1 }}', 'var2': '{{ var2 }}'}, 'variables': {'properties': {'var1': {'type': 'string', 'title': 'Var1', 'default': 'hello'}, 'var2': {'type': 'integer', 'title': 'Var2', 'default': 42}}}, 'required': []}, {'var1': 'woof!'}, {'command': None, 'env': {}, 'labels': {}, 'name': None, 'var1': 'woof!', 'var2': 42}), ({'job_configuration': {'var1': '{{ var1 }}', 'var2': '{{ var2 }}'}, 'variables': {'properties': {'var1': {'type': 'string', 'title': 'Var1'}, 'var2': {'type': 'integer', 'title': 'Var2', 'default': 42}}}, 'required': ['var1']}, {'var1': 'woof!'}, {'command': None, 'env': {}, 'labels': {}, 'name': None, 'var1': 'woof!', 'var2': 42})])
async def test_job_configuration_from_template_and_overrides(
    template: Dict[str, Any],
    overrides: Dict[str, Any],
    expected: Dict[str, Any]
) -> None: ...

async def test_job_configuration_from_template_and_overrides_with_nested_variables() -> None: ...

async def test_job_configuration_from_template_and_overrides_with_hard_coded_primitives() -> None: ...

async def test_job_configuration_from_template_overrides_with_block() -> None: ...

async def test_job_configuration_from_template_coerces_work_pool_values() -> None: ...

@pytest.mark.usefixtures('variables')
async def test_job_configuration_from_template_overrides_with_remote_variables() -> None: ...

@pytest.mark.usefixtures('variables')
async def test_job_configuration_from_template_overrides_with_remote_variables_hardcodes() -> None: ...

async def test_job_configuration_from_template_and_overrides_with_variables_in_a_list() -> None: ...

@pytest.mark.parametrize('falsey_value', [None, ''])
async def test_base_job_configuration_converts_falsey_values_to_none(falsey_value: Any) -> None: ...

@pytest.mark.parametrize('field_template_value,expected_final_template', [('{{ var1 }}', {'command': '{{ command }}', 'env': '{{ env }}', 'labels': '{{ labels }}', 'name': '{{ name }}', 'var1': '{{ var1 }}', 'var2': '{{ var2 }}'}), (None, {'command': '{{ command }}', 'env': '{{ env }}', 'labels': '{{ labels }}', 'name': '{{ name }}', 'var1': '{{ var1 }}', 'var2': '{{ var2 }}'}), ('{{ dog }}', {'command': '{{ command }}', 'env': '{{ env }}', 'labels': '{{ labels }}', 'name': '{{ name }}', 'var1': '{{ dog }}', 'var2': '{{ var2 }}'})])
def test_job_configuration_produces_correct_json_template(
    field_template_value: Any,
    expected_final_template: Dict[str, Any]
) -> None: ...

class TestWorkerProperties:
    def test_defaults(self) -> None: ...
    
    def test_custom_logo_url(self) -> None: ...
    
    def test_custom_documentation_url(self) -> None: ...
    
    def test_custom_description(self) -> None: ...
    
    def test_custom_base_job_configuration(self) -> None: ...

class TestPrepareForFlowRun:
    @pytest.fixture
    def job_config(self) -> BaseJobConfiguration: ...
    
    @pytest.fixture
    def flow_run(self) -> FlowRun: ...
    
    @pytest.fixture
    def flow(self) -> Flow: ...
    
    @pytest.fixture
    def deployment(self, flow: Flow) -> DeploymentResponse: ...
    
    def test_prepare_for_flow_run_without_deployment_and_flow(
        self,
        job_config: BaseJobConfiguration,
        flow_run: FlowRun
    ) -> None: ...
    
    def test_prepare_for_flow_run(
        self,
        job_config: BaseJobConfiguration,
        flow_run: FlowRun
    ) -> None: ...
    
    def test_prepare_for_flow_run_with_deployment_and_flow(
        self,
        job_config: BaseJobConfiguration,
        flow_run: FlowRun,
        deployment: DeploymentResponse,
        flow: Flow
    ) -> None: ...

async def test_get_flow_run_logger_without_worker_id_set(
    prefect_client: Any,
    worker_deployment_wq1: Any,
    work_pool: Any
) -> None: ...

async def test_get_flow_run_logger_with_worker_id_set(
    prefect_client: Any,
    worker_deployment_wq1: Any,
    work_pool: Any
) -> None: ...

class TestInfrastructureIntegration:
    async def test_worker_crashes_flow_if_infrastructure_submission_fails(
        self,
        prefect_client: Any,
        worker_deployment_infra_wq1: Any,
        work_pool: Any,
        monkeypatch: Any
    ) -> None: ...

async def test_worker_set_last_polled_time(work_pool: Any) -> None: ...

async def test_worker_last_polled_health_check(work_pool: Any) -> None: ...

class TestBaseWorkerStart:
    async def test_start_syncs_with_the_server(self, work_pool: Any) -> None: ...
    
    async def test_start_executes_flow_runs(
        self,
        prefect_client: Any,
        worker_deployment_wq1: Any,
        work_pool: Any
    ) -> None: ...

@pytest.mark.parametrize('work_pool_env, deployment_env, flow_run_env, expected_env', [({}, {'test-var': 'foo'}, {'another-var': 'boo'}, {'test-var': 'foo', 'another-var': 'boo'}), ({'A': '1', 'B': '2'}, {'A': '1', 'B': '3'}, {}, {'A': '1', 'B': '3'}), ({'A': '1', 'B': '2'}, {'C': '3', 'D': '4'}, {}, {'A': '1', 'B': '2', 'C': '3', 'D': '4'}), ({'A': '1', 'B': '2'}, {'C': '42'}, {'C': '3', 'D': '4'}, {'A': '1', 'B': '2', 'C': '3', 'D': '4'}), ({'A': '1', 'B': '2'}, {'B': ''}, {}, {'A': '1', 'B': ''})])
@pytest.mark.parametrize('use_variable_defaults', [True, False])
async def test_env_merge_logic_is_deep(
    prefect_client: Any,
    session: Any,
    flow: Any,
    work_pool: Any,
    work_pool_env: Dict[str, Any],
    deployment_env: Dict[str, Any],
    flow_run_env: Dict[str, Any],
    expected_env: Dict[str, Any],
    use_variable_defaults: bool
) -> None: ...

class TestBaseWorkerHeartbeat:
    async def test_worker_heartbeat_sends_integrations(
        self,
        work_pool: Any,
        hosted_api_server: Any
    ) -> None: ...
    
    async def test_custom_worker_can_send_arbitrary_metadata(
        self,
        work_pool: Any,
        hosted_api_server: Any
    ) -> None: ...

async def test_worker_gives_labels_to_flow_runs_when_using_cloud_api(
    prefect_client: Any,
    worker_deployment_wq1: Any,
    work_pool: Any
) -> None: ...

async def test_worker_removes_flow_run_from_submitting_when_not_ready(
    prefect_client: Any,
    worker_deployment_wq1: Any,
    work_pool: Any
) -> None: ...
```