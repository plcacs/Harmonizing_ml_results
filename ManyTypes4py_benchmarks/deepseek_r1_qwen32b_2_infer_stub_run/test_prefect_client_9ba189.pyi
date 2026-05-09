import httpx
import pendulum
import pytest
import respx
import uuid
from datetime import timedelta
from typing import Any, Dict, Generator, List, Optional, Union
from unittest.mock import MagicMock, AsyncMock, Mock
from prefect.client.schemas import (
    ArtifactCreate,
    BlockDocumentCreate,
    DeploymentScheduleCreate,
    GlobalConcurrencyLimitCreate,
    GlobalConcurrencyLimitUpdate,
    LogCreate,
    VariableCreate,
    WorkPoolCreate,
    WorkPoolUpdate,
    ArtifactFilter,
    ArtifactFilterKey,
    DeploymentFilter,
    DeploymentFilterTags,
    FlowFilter,
    FlowRunFilter,
    FlowRunFilterTags,
    FlowRunNotificationPolicyFilter,
    LogFilter,
    LogFilterFlowRunId,
    TaskRunFilter,
    TaskRunFilterFlowRunId,
    Flow,
    FlowRunNotificationPolicy,
    FlowRunPolicy,
    Integration,
    StateType,
    TaskRun,
    Variable,
    WorkerMetadata,
    WorkQueue,
    DeploymentResponse,
    OrchestrationResult,
    SetStateStatus,
    CronSchedule,
    IntervalSchedule,
)
from prefect.client.orchestration import PrefectClient, ServerType, SyncPrefectClient

class TestGetClient:
    def test_get_client_returns_client(self) -> None:
        ...
    
    def test_get_client_does_not_cache_client(self) -> None:
        ...
    
    def test_get_client_cache_uses_profile_settings(self) -> None:
        ...
    
    def test_get_client_rasises_error_when_no_api_url_and_no_ephemeral_mode(self, disable_hosted_api_server: pytest.fixture) -> None:
        ...
    
    def test_get_client_starts_subprocess_server_when_enabled(self, enable_ephemeral_server: pytest.fixture, monkeypatch: pytest.MonkeyPatch) -> None:
        ...

class TestClientProxyAwareness:
    @pytest.fixture
    def remote_https_api(self) -> Generator[httpx.URL, None, None]:
        ...
    
    def test_unproxied_remote_client_will_retry(self, remote_https_api: httpx.URL) -> None:
        ...
    
    def test_users_can_still_provide_transport(self, remote_https_api: httpx.URL) -> None:
        ...
    
    @pytest.fixture
    def https_proxy(self) -> Generator[httpcore.URL, None, None]:
        ...
    
    async def test_client_is_aware_of_https_proxy(self, remote_https_api: httpx.URL, https_proxy: httpcore.URL) -> None:
        ...
    
    @pytest.fixture
    def remote_http_api(self) -> Generator[httpx.URL, None, None]:
        ...
    
    @pytest.fixture
    def http_proxy(self) -> Generator[httpcore.URL, None, None]:
        ...
    
    async def test_client_is_aware_of_http_proxy(self, remote_http_api: httpx.URL, http_proxy: httpcore.URL) -> None:
        ...

class TestInjectClient:
    @staticmethod
    @inject_client
    async def injected_func(client: PrefectClient) -> PrefectClient:
        ...
    
    async def test_get_new_client(self) -> None:
        ...
    
    async def test_get_new_client_with_explicit_none(self) -> None:
        ...
    
    async def test_use_existing_client(self, prefect_client: PrefectClient) -> None:
        ...
    
    async def test_use_existing_client_from_flow_run_ctx(self, prefect_client: PrefectClient) -> None:
        ...
    
    async def test_use_existing_client_from_task_run_ctx(self, prefect_client: PrefectClient) -> None:
        ...
    
    async def test_use_existing_client_from_flow_run_ctx_with_null_kwarg(self, prefect_client: PrefectClient) -> None:
        ...

class TestClientContextManager:
    async def test_client_context_can_be_reentered(self) -> None:
        ...
    
    async def test_client_context_cannot_be_reused(self) -> None:
        ...
    
    async def test_client_context_manages_app_lifespan(self) -> None:
        ...
    
    async def test_client_context_calls_app_lifespan_once_despite_nesting(self) -> None:
        ...
    
    async def test_client_context_manages_app_lifespan_on_sequential_usage(self) -> None:
        ...
    
    async def test_client_context_lifespan_is_robust_to_async_concurrency(self) -> None:
        ...
    
    async def test_client_context_lifespan_is_robust_to_dependency_deadlocks(self) -> None:
        ...
    
    async def test_client_context_manages_app_lifespan_on_exception(self) -> None:
        ...
    
    async def test_client_context_manages_app_lifespan_on_anyio_cancellation(self) -> None:
        ...
    
    async def test_client_context_manages_app_lifespan_on_exception_when_nested(self) -> None:
        ...
    
    async def test_with_without_async_raises_helpful_error(self) -> None:
        ...

@pytest.mark.parametrize('enabled', [True, False])
async def test_client_runs_migrations_for_ephemeral_app_only_once(enabled: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    ...

@pytest.mark.parametrize('enabled', [True, False])
async def test_client_runs_migrations_for_two_different_ephemeral_apps(enabled: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    ...

async def test_client_does_not_run_migrations_for_hosted_app(hosted_api_server: pytest.fixture, monkeypatch: pytest.MonkeyPatch) -> None:
    ...

async def test_client_api_url() -> None:
    ...

async def test_hello(prefect_client: PrefectClient) -> None:
    ...

async def test_healthcheck(prefect_client: PrefectClient) -> None:
    ...

async def test_healthcheck_failure(prefect_client: PrefectClient, monkeypatch: pytest.MonkeyPatch) -> None:
    ...

async def test_create_then_read_flow(prefect_client: PrefectClient) -> None:
    ...

async def test_create_then_delete_flow(prefect_client: PrefectClient) -> None:
    ...

async def test_create_then_read_deployment(prefect_client: PrefectClient, storage_document_id: uuid.UUID) -> None:
    ...

async def test_read_deployment_errors_on_invalid_uuid(prefect_client: PrefectClient) -> None:
    ...

async def test_update_deployment(prefect_client: PrefectClient, storage_document_id: uuid.UUID) -> None:
    ...

async def test_update_deployment_to_remove_schedules(prefect_client: PrefectClient, storage_document_id: uuid.UUID) -> None:
    ...

async def test_read_deployment_by_name(prefect_client: PrefectClient) -> None:
    ...

@pytest.mark.parametrize('deployment_tags,filter_tags,expected_match', [
    (['tag-1'], ['tag-1'], True),
    (['tag-2'], ['tag-1'], False),
    (['tag-1', 'tag-2'], ['tag-1', 'tag-3'], True),
    (['tag-1'], ['tag-1', 'tag-2'], True),
    (['tag-2'], ['tag-1', 'tag-2'], True),
    (['tag-1'], ['tag-2', 'tag-3'], False),
    (['tag-1'], ['get-real'], False),
    ([], ['tag-1'], False),
    (['tag-1'], [], False)
], ids=['single_tag_match', 'single_tag_no_match', 'multiple_tags_partial_match', 'subset_match_1', 'subset_match_2', 'no_matching_tags', 'nonexistent_tag', 'empty_run_tags', 'empty_filter_tags'])
async def test_read_deployment_by_any_tag(prefect_client: PrefectClient, deployment_tags: List[str], filter_tags: List[str], expected_match: bool) -> None:
    ...

async def test_create_then_delete_deployment(prefect_client: PrefectClient) -> None:
    ...

async def test_read_nonexistent_deployment_by_name(prefect_client: PrefectClient) -> None:
    ...

async def test_create_then_read_concurrency_limit(prefect_client: PrefectClient) -> None:
    ...

async def test_read_nonexistent_concurrency_limit_by_tag(prefect_client: PrefectClient) -> None:
    ...

async def test_resetting_concurrency_limits(prefect_client: PrefectClient) -> None:
    ...

async def test_deleting_concurrency_limits(prefect_client: PrefectClient) -> None:
    ...

async def test_create_then_read_flow_run(prefect_client: PrefectClient) -> None:
    ...

async def test_create_flow_run_retains_parameters(prefect_client: PrefectClient) -> None:
    ...

async def test_create_flow_run_with_state(prefect_client: PrefectClient) -> None:
    ...

async def test_set_then_read_flow_run_state(prefect_client: PrefectClient) -> None:
    ...

async def test_set_flow_run_state_404_is_object_not_found(prefect_client: PrefectClient) -> None:
    ...

async def test_read_flow_runs_without_filter(prefect_client: PrefectClient) -> None:
    ...

async def test_read_flow_runs_with_filtering(prefect_client: PrefectClient) -> None:
    ...

@pytest.mark.parametrize('run_tags,filter_tags,expected_match', [
    (['tag-1'], ['tag-1'], True),
    (['tag-2'], ['tag-1'], False),
    (['tag-1', 'tag-2'], ['tag-1', 'tag-3'], True),
    (['tag-1'], ['tag-1', 'tag-2'], True),
    (['tag-2'], ['tag-1', 'tag-2'], True),
    (['tag-1'], ['tag-2', 'tag-3'], False),
    (['tag-1'], ['get-real'], False),
    ([], ['tag-1'], False),
    (['tag-1'], [], False)
], ids=['single_tag_match', 'single_tag_no_match', 'multiple_tags_partial_match', 'subset_match_1', 'subset_match_2', 'no_matching_tags', 'nonexistent_tag', 'empty_run_tags', 'empty_filter_tags'])
async def test_read_flow_runs_with_tags(prefect_client: PrefectClient, run_tags: List[str], filter_tags: List[str], expected_match: bool) -> None:
    ...

async def test_read_flows_without_filter(prefect_client: PrefectClient) -> None:
    ...

async def test_read_flows_with_filter(prefect_client: PrefectClient) -> None:
    ...

async def test_read_flow_by_name(prefect_client: PrefectClient) -> None:
    ...

async def test_create_flow_run_from_deployment(prefect_client: PrefectClient, deployment: DeploymentResponse) -> None:
    ...

async def test_create_flow_run_from_deployment_idempotency(prefect_client: PrefectClient, deployment: DeploymentResponse) -> None:
    ...

async def test_create_flow_run_from_deployment_with_options(prefect_client: PrefectClient, deployment: DeploymentResponse) -> None:
    ...

async def test_update_flow_run(prefect_client: PrefectClient) -> None:
    ...

async def test_update_flow_run_overrides_tags(prefect_client: PrefectClient) -> None:
    ...

async def test_create_then_read_task_run(prefect_client: PrefectClient) -> None:
    ...

async def test_delete_task_run(prefect_client: PrefectClient) -> None:
    ...

async def test_create_then_read_task_run_with_state(prefect_client: PrefectClient) -> None:
    ...

async def test_set_then_read_task_run_state(prefect_client: PrefectClient) -> None:
    ...

async def test_create_then_read_autonomous_task_runs(prefect_client: PrefectClient) -> None:
    ...

async def test_create_then_read_flow_run_notification_policy(prefect_client: PrefectClient, block_document: BlockDocumentCreate) -> None:
    ...

async def test_create_then_update_flow_run_notification_policy(prefect_client: PrefectClient, block_document: BlockDocumentCreate) -> None:
    ...

async def test_create_then_delete_flow_run_notification_policy(prefect_client: PrefectClient, block_document: BlockDocumentCreate) -> None:
    ...

async def test_read_filtered_logs(session: Any, prefect_client: PrefectClient, deployment: DeploymentResponse) -> None:
    ...

async def test_prefect_api_tls_insecure_skip_verify_setting_set_to_true(monkeypatch: pytest.MonkeyPatch) -> None:
    ...

async def test_prefect_api_tls_insecure_skip_verify_setting_set_to_false(monkeypatch: pytest.MonkeyPatch) -> None:
    ...

async def test_prefect_api_tls_insecure_skip_verify_default_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    ...

async def test_prefect_api_ssl_cert_file_setting_explicitly_set(monkeypatch: pytest.MonkeyPatch) -> None:
    ...

async def test_prefect_api_ssl_cert_file_default_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    ...

async def test_prefect_api_ssl_cert_file_default_setting_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    ...

class TestClientAPIVersionRequests:
    @pytest.fixture
    def versions(self) -> List[str]:
        ...
    
    @pytest.fixture
    def major_version(self, versions: List[str]) -> int:
        ...
    
    @pytest.fixture
    def minor_version(self, versions: List[str]) -> int:
        ...
    
    @pytest.fixture
    def patch_version(self, versions: List[str]) -> int:
        ...
    
    async def test_default_requests_succeeds(self) -> None:
        ...
    
    async def test_no_api_version_header_succeeds(self) -> None:
        ...
    
    async def test_major_version(self, app: FastAPI, major_version: int, minor_version: int, patch_version: int) -> None:
        ...
    
    @pytest.mark.skip(reason='This test is no longer compatible with the current API version checking logic')
    async def test_minor_version(self, app: FastAPI, major_version: int, minor_version: int, patch_version: int) -> None:
        ...
    
    @pytest.mark.skip(reason='This test is no longer compatible with the current API version checking logic')
    async def test_patch_version(self, app: FastAPI, major_version: int, minor_version: int, patch_version: int) -> None:
        ...
    
    async def test_invalid_header(self, app: FastAPI) -> None:
        ...

class TestClientAPIKey:
    @pytest.fixture
    async def test_app(self) -> FastAPI:
        ...
    
    async def test_client_passes_api_key_as_auth_header(self, test_app: FastAPI) -> None:
        ...
    
    async def test_client_no_auth_header_without_api_key(self, test_app: FastAPI) -> None:
        ...
    
    async def test_get_client_includes_api_key_from_context(self) -> None:
        ...

class TestClientAuthString:
    @pytest.fixture
    async def test_app(self) -> FastAPI:
        ...
    
    async def test_client_passes_auth_string_as_auth_header(self, test_app: FastAPI) -> None:
        ...
    
    async def test_client_no_auth_header_without_auth_string(self, test_app: FastAPI) -> None:
        ...
    
    async def test_get_client_includes_auth_string_from_context(self) -> None:
        ...

class TestClientWorkQueues:
    @pytest.fixture
    async def deployment(self, prefect_client: PrefectClient) -> uuid.UUID:
        ...
    
    async def test_create_then_read_work_queue(self, prefect_client: PrefectClient) -> None:
        ...
    
    async def test_create_and_read_includes_status(self, prefect_client: PrefectClient) -> None:
        ...
    
    async def test_create_then_read_work_queue_by_name(self, prefect_client: PrefectClient) -> None:
        ...
    
    async def test_create_queue_with_settings(self, prefect_client: PrefectClient) -> None:
        ...
    
    async def test_create_then_match_work_queues(self, prefect_client: PrefectClient) -> None:
        ...
    
    async def test_read_nonexistant_work_queue(self, prefect_client: PrefectClient) -> None:
        ...
    
    async def test_get_runs_from_queue_includes(self, prefect_client: PrefectClient, deployment: uuid.UUID) -> None:
        ...
    
    async def test_get_runs_from_queue_respects_limit(self, prefect_client: PrefectClient, deployment: uuid.UUID) -> None:
        ...

async def test_delete_flow_run(prefect_client: PrefectClient, flow_run: Flow) -> None:
    ...

def test_server_type_ephemeral(enable_ephemeral_server: pytest.fixture) -> None:
    ...

async def test_server_type_server(hosted_api_server: pytest.fixture) -> None:
    ...

async def test_server_type_cloud() -> None:
    ...

@pytest.mark.parametrize('on_create, expected_value', [(True, True), (False, False), (None, False)])
async def test_update_deployment_does_not_overwrite_paused_when_not_provided(prefect_client: PrefectClient, flow_run: Flow, on_create: Optional[bool], expected_value: bool) -> None:
    ...

@pytest.mark.parametrize('on_create, after_create, on_update, after_update', [(False, False, True, True), (True, True, False, False), (None, False, True, True)])
async def test_update_deployment_paused(prefect_client: PrefectClient, flow_run: Flow, on_create: Optional[bool], after_create: bool, on_update: bool, after_update: bool) -> None:
    ...

class TestWorkPools:
    async def test_read_work_pools(self, prefect_client: PrefectClient) -> None:
        ...
    
    async def test_create_work_pool_overwriting_existing_work_pool(self, prefect_client: PrefectClient, work_pool: WorkPoolCreate) -> None:
        ...
    
    async def test_create_work_pool_with_attempt_to_overwrite_type(self, prefect_client: PrefectClient, work_pool: WorkPoolCreate) -> None:
        ...
    
    async def test_update_work_pool(self, prefect_client: PrefectClient) -> None:
        ...
    
    async def test_update_missing_work_pool(self, prefect_client: PrefectClient) -> None:
        ...
    
    async def test_delete_work_pool(self, prefect_client: PrefectClient, work_pool: WorkPoolCreate) -> None:
        ...

class TestArtifacts:
    @pytest.fixture
    async def artifacts(self, prefect_client: PrefectClient) -> List[ArtifactCreate]:
        ...
    
    async def test_create_then_read_artifact(self, prefect_client: PrefectClient, client: httpx.Client) -> None:
        ...
    
    async def test_read_artifacts(self, prefect_client: PrefectClient, artifacts: List[ArtifactCreate]) -> None:
        ...
    
    async def test_read_artifacts_with_latest_filter(self, prefect_client: PrefectClient, artifacts: List[ArtifactCreate]) -> None:
        ...
    
    async def test_read_artifacts_with_key_filter(self, prefect_client: PrefectClient, artifacts: List[ArtifactCreate]) -> None:
        ...
    
    async def test_delete_artifact_succeeds(self, prefect_client: PrefectClient, artifacts: List[ArtifactCreate]) -> None:
        ...
    
    async def test_delete_nonexistent_artifact_raises(self, prefect_client: PrefectClient) -> None:
        ...

class TestVariables:
    @pytest.fixture
    async def variable(self, client: httpx.Client) -> Variable:
        ...
    
    @pytest.fixture
    async def variables(self, client: httpx.Client) -> List[Variable]:
        ...
    
    @pytest.mark.parametrize('value', ['string-value', '"string-value"', 123, 12.3, True, False, None, {'key': 'value'}, ['value1', 'value2'], {'key': ['value1', 'value2']})
    async def test_create_variable(self, prefect_client: PrefectClient, value: Any) -> None:
        ...
    
    async def test_read_variable_by_name(self, prefect_client: PrefectClient, variable: Variable) -> None:
        ...
    
    async def test_read_variable_by_name_doesnt_exist(self, prefect_client: PrefectClient) -> None:
        ...
    
    async def test_delete_variable_by_name(self, prefect_client: PrefectClient, variable: Variable) -> None:
        ...
    
    async def test_delete_variable_by_name_doesnt_exist(self, prefect_client: PrefectClient) -> None:
        ...
    
    async def test_read_variables(self, prefect_client: PrefectClient, variables: List[Variable]) -> None:
        ...
    
    async def test_read_variables_with_limit(self, prefect_client: PrefectClient, variables: List[Variable]) -> None:
        ...

class TestAutomations:
    @pytest.fixture
    def automation(self) -> AutomationCore:
        ...
    
    async def test_create_automation(self, cloud_client: PrefectClient, automation: AutomationCore) -> None:
        ...
    
    async def test_read_automation(self, cloud_client: PrefectClient, automation: AutomationCore) -> None:
        ...
    
    async def test_read_automation_not_found(self, cloud_client: PrefectClient, automation: AutomationCore) -> None:
        ...
    
    async def test_read_automations_by_name(self, cloud_client: PrefectClient, automation: AutomationCore) -> None:
        ...
    
    @pytest.fixture
    def automation2(self) -> AutomationCore:
        ...
    
    async def test_read_automations_by_name_multiple_same_name(self, cloud_client: PrefectClient, automation: AutomationCore, automation2: AutomationCore) -> None:
        ...
    
    async def test_read_automations_by_name_not_found(self, cloud_client: PrefectClient, automation: AutomationCore) -> None:
        ...
    
    async def test_delete_owned_automations(self, cloud_client: PrefectClient) -> None:
        ...

async def test_server_error_does_not_raise_on_client() -> None:
    ...

async def test_prefect_client_follow_redirects() -> None:
    ...

async def test_global_concurrency_limit_create(prefect_client: PrefectClient) -> None:
    ...

async def test_global_concurrency_limit_delete(prefect_client: PrefectClient) -> None:
    ...

async def test_global_concurrency_limit_update_with_integer(prefect_client: PrefectClient) -> None:
    ...

async def test_global_concurrency_limit_read_nonexistent_by_name(prefect_client: PrefectClient) -> None:
    ...

class TestPrefectClientDeploymentSchedules:
    @pytest.fixture
    async def deployment(self, prefect_client: PrefectClient) -> DeploymentResponse:
        ...
    
    async def test_create_deployment_schedule(self, prefect_client: PrefectClient, deployment: DeploymentResponse) -> None:
        ...
    
    async def test_create_multiple_deployment_schedules_success(self, prefect_client: PrefectClient, deployment: DeploymentResponse) -> None:
        ...
    
    async def test_read_deployment_schedules_success(self, prefect_client: PrefectClient, deployment: DeploymentResponse) -> None:
        ...
    
    async def test_update_deployment_schedule_only_active(self, deployment: DeploymentResponse, prefect_client: PrefectClient) -> None:
        ...
    
    async def test_update_deployment_schedule_only_schedule(self, deployment: DeploymentResponse, prefect_client: PrefectClient) -> None:
        ...
    
    async def test_update_deployment_schedule_all_fields(self, deployment: DeploymentResponse, prefect_client: PrefectClient) -> None:
        ...
    
    async def test_delete_deployment_schedule_success(self, deployment: DeploymentResponse, prefect_client: PrefectClient) -> None:
        ...
    
    async def test_create_deployment_schedules_with_invalid_schedule(self, prefect_client: PrefectClient, deployment: DeploymentResponse) -> None:
        ...
    
    async def test_read_deployment_schedule_nonexistent(self, prefect_client: PrefectClient) -> None:
        ...
    
    async def test_update_deployment_schedule_nonexistent(self, prefect_client: PrefectClient, deployment: DeploymentResponse) -> None:
        ...
    
    async def test_delete_deployment_schedule_nonexistent(self, prefect_client: PrefectClient, deployment: DeploymentResponse) -> None:
        ...

class TestPrefectClientCsrfSupport:
    def test_enabled_ephemeral(self, enable_ephemeral_server: pytest.fixture) -> None:
        ...
    
    async def test_enabled_server_type(self, hosted_api_server: pytest.fixture) -> None:
        ...
    
    async def test_not_enabled_server_type_cloud(self) -> None:
        ...
    
    async def test_disabled_setting_disabled(self, hosted_api_server: pytest.fixture) -> None:
        ...

class TestPrefectClientRaiseForAPIVersionMismatch:
    async def test_raise_for_api_version_mismatch(self, prefect_client: PrefectClient) -> None:
        ...
    
    async def test_raise_for_api_version_mismatch_when_api_unreachable(self, prefect_client: PrefectClient, monkeypatch: pytest.MonkeyPatch) -> None:
        ...
    
    async def test_raise_for_api_version_mismatch_against_cloud(self, prefect_client: PrefectClient, monkeypatch: pytest.MonkeyPatch) -> None:
        ...
    
    @pytest.mark.parametrize('client_version, api_version', [('3.0.0', '2.0.0'), ('2.0.0', '3.0.0')])
    async def test_raise_for_api_version_mismatch_with_incompatible_versions(self, prefect_client: PrefectClient, monkeypatch: pytest.MonkeyPatch, client_version: str, api_version: str) -> None:
        ...

class TestSyncClient:
    def test_get_sync_client(self) -> None:
        ...
    
    def test_fixture_is_sync(self, sync_prefect_client: SyncPrefectClient) -> None:
        ...
    
    def test_hello(self, sync_prefect_client: SyncPrefectClient) -> None:
        ...
    
    def test_api_version(self, sync_prefect_client: SyncPrefectClient) -> None:
        ...

class TestSyncClientRaiseForAPIVersionMismatch:
    def test_raise_for_api_version_mismatch(self, sync_prefect_client: SyncPrefectClient) -> None:
        ...
    
    def test_raise_for_api_version_mismatch_when_api_unreachable(self, sync_prefect_client: SyncPrefectClient, monkeypatch: pytest.MonkeyPatch) -> None:
        ...
    
    def test_raise_for_api_version_mismatch_against_cloud(self, sync_prefect_client: SyncPrefectClient, monkeypatch: pytest.MonkeyPatch) -> None:
        ...
    
    @pytest.mark.parametrize('client_version, api_version', [('3.0.0', '2.0.0'), ('2.0.0', '3.0.0')])
    def test_raise_for_api_version_mismatch_with_incompatible_versions(self, sync_prefect_client: SyncPrefectClient, monkeypatch: pytest.MonkeyPatch, client_version: str, api_version: str) -> None:
        ...

class TestPrefectClientWorkerHeartbeat:
    async def test_worker_heartbeat(self, prefect_client: PrefectClient, work_pool: WorkPoolCreate) -> None:
        ...
    
    async def test_worker_heartbeat_sends_metadata_if_passed(self, prefect_client: PrefectClient) -> None:
        ...
    
    async def test_worker_heartbeat_does_not_send_metadata_if_not_passed(self, prefect_client: PrefectClient) -> None:
        ...