```python
from typing import Any, Optional, List, Dict, Set, Tuple, Union, Generator
from uuid import UUID
from datetime import timedelta
from contextlib import asynccontextmanager
import ssl
import httpx
import httpcore
from fastapi import FastAPI
from prefect.client.schemas.actions import (
    ArtifactCreate,
    BlockDocumentCreate,
    DeploymentScheduleCreate,
    GlobalConcurrencyLimitCreate,
    GlobalConcurrencyLimitUpdate,
    LogCreate,
    VariableCreate,
    WorkPoolCreate,
    WorkPoolUpdate,
)
from prefect.client.schemas.filters import (
    ArtifactFilter,
    DeploymentFilter,
    FlowFilter,
    FlowRunFilter,
    FlowRunNotificationPolicyFilter,
    LogFilter,
    TaskRunFilter,
)
from prefect.client.schemas.objects import (
    Flow,
    FlowRunNotificationPolicy,
    FlowRunPolicy,
    StateType,
    TaskRun,
    Variable,
    WorkerMetadata,
    WorkQueue,
)
from prefect.client.schemas.responses import DeploymentResponse, OrchestrationResult
from prefect.client.schemas.schedules import CronSchedule, IntervalSchedule
from prefect.events import AutomationCore, EventTrigger
from prefect.states import State
from prefect.types import DateTime
from prefect.settings import (
    PREFECT_API_AUTH_STRING,
    PREFECT_API_DATABASE_MIGRATE_ON_START,
    PREFECT_API_KEY,
    PREFECT_API_SSL_CERT_FILE,
    PREFECT_API_TLS_INSECURE_SKIP_VERIFY,
    PREFECT_API_URL,
    PREFECT_CLIENT_CSRF_SUPPORT_ENABLED,
    PREFECT_CLOUD_API_URL,
    PREFECT_TESTING_UNIT_TEST_MODE,
)
from prefect.client.orchestration import ServerType

def get_client(
    httpx_settings: Optional[Dict[str, Any]] = ...,
    sync_client: bool = ...,
) -> Any: ...

def not_enough_open_files() -> bool: ...

class TestGetClient:
    def test_get_client_returns_client(self) -> None: ...
    def test_get_client_does_not_cache_client(self) -> None: ...
    def test_get_client_cache_uses_profile_settings(self) -> None: ...
    def test_get_client_starts_subprocess_server_when_enabled(
        self,
        enable_ephemeral_server: Any,
        monkeypatch: Any,
    ) -> None: ...
    def test_get_client_rasises_error_when_no_api_url_and_no_ephemeral_mode(
        self,
        disable_hosted_api_server: Any,
    ) -> None: ...

class TestClientProxyAwareness:
    @pytest.fixture()
    def remote_https_api(self) -> Any: ...
    def test_unproxied_remote_client_will_retry(
        self,
        remote_https_api: Any,
    ) -> None: ...
    def test_users_can_still_provide_transport(
        self,
        remote_https_api: Any,
    ) -> None: ...
    @pytest.fixture
    def https_proxy(self) -> Any: ...
    async def test_client_is_aware_of_https_proxy(
        self,
        remote_https_api: Any,
        https_proxy: Any,
    ) -> None: ...
    @pytest.fixture()
    def remote_http_api(self) -> Any: ...
    @pytest.fixture
    def http_proxy(self) -> Any: ...
    async def test_client_is_aware_of_http_proxy(
        self,
        remote_http_api: Any,
        http_proxy: Any,
    ) -> None: ...

class TestInjectClient:
    @staticmethod
    @inject_client
    async def injected_func(client: Any) -> Any: ...
    async def test_get_new_client(self) -> None: ...
    async def test_get_new_client_with_explicit_none(self) -> None: ...
    async def test_use_existing_client(
        self,
        prefect_client: Any,
    ) -> None: ...
    async def test_use_existing_client_from_flow_run_ctx(
        self,
        prefect_client: Any,
    ) -> None: ...
    async def test_use_existing_client_from_task_run_ctx(
        self,
        prefect_client: Any,
    ) -> None: ...
    async def test_use_existing_client_from_flow_run_ctx_with_null_kwarg(
        self,
        prefect_client: Any,
    ) -> None: ...

def make_lifespan(
    startup: Any,
    shutdown: Any,
) -> Any: ...

class TestClientContextManager:
    async def test_client_context_can_be_reentered(self) -> None: ...
    async def test_client_context_cannot_be_reused(self) -> None: ...
    async def test_client_context_manages_app_lifespan(self) -> None: ...
    async def test_client_context_calls_app_lifespan_once_despite_nesting(self) -> None: ...
    async def test_client_context_manages_app_lifespan_on_sequential_usage(self) -> None: ...
    async def test_client_context_lifespan_is_robust_to_async_concurrency(self) -> None: ...
    async def test_client_context_lifespan_is_robust_to_dependency_deadlocks(self) -> None: ...
    async def test_client_context_manages_app_lifespan_on_exception(self) -> None: ...
    async def test_client_context_manages_app_lifespan_on_anyio_cancellation(self) -> None: ...
    async def test_client_context_manages_app_lifespan_on_exception_when_nested(self) -> None: ...
    async def test_with_without_async_raises_helpful_error(self) -> None: ...

@pytest.mark.parametrize('enabled', [True, False])
async def test_client_runs_migrations_for_ephemeral_app_only_once(
    enabled: bool,
    monkeypatch: Any,
) -> None: ...

@pytest.mark.parametrize('enabled', [True, False])
async def test_client_runs_migrations_for_two_different_ephemeral_apps(
    enabled: bool,
    monkeypatch: Any,
) -> None: ...

async def test_client_does_not_run_migrations_for_hosted_app(
    hosted_api_server: Any,
    monkeypatch: Any,
) -> None: ...

async def test_client_api_url() -> None: ...

async def test_hello(prefect_client: Any) -> None: ...

async def test_healthcheck(prefect_client: Any) -> None: ...

async def test_healthcheck_failure(
    prefect_client: Any,
    monkeypatch: Any,
) -> None: ...

async def test_create_then_read_flow(prefect_client: Any) -> None: ...

async def test_create_then_delete_flow(prefect_client: Any) -> None: ...

async def test_create_then_read_deployment(
    prefect_client: Any,
    storage_document_id: Any,
) -> None: ...

async def test_read_deployment_errors_on_invalid_uuid(
    prefect_client: Any,
) -> None: ...

async def test_update_deployment(
    prefect_client: Any,
    storage_document_id: Any,
) -> None: ...

async def test_update_deployment_to_remove_schedules(
    prefect_client: Any,
    storage_document_id: Any,
) -> None: ...

async def test_read_deployment_by_name(prefect_client: Any) -> None: ...

@pytest.mark.parametrize(
    'deployment_tags,filter_tags,expected_match',
    [
        (['tag-1'], ['tag-1'], True),
        (['tag-2'], ['tag-1'], False),
        (['tag-1', 'tag-2'], ['tag-1', 'tag-3'], True),
        (['tag-1'], ['tag-1', 'tag-2'], True),
        (['tag-2'], ['tag-1', 'tag-2'], True),
        (['tag-1'], ['tag-2', 'tag-3'], False),
        (['tag-1'], ['get-real'], False),
        ([], ['tag-1'], False),
        (['tag-1'], [], False),
    ],
)
async def test_read_deployment_by_any_tag(
    prefect_client: Any,
    deployment_tags: List[str],
    filter_tags: List[str],
    expected_match: bool,
) -> None: ...

async def test_create_then_delete_deployment(prefect_client: Any) -> None: ...

async def test_read_nonexistent_deployment_by_name(prefect_client: Any) -> None: ...

async def test_create_then_read_concurrency_limit(prefect_client: Any) -> None: ...

async def test_read_nonexistent_concurrency_limit_by_tag(
    prefect_client: Any,
) -> None: ...

async def test_resetting_concurrency_limits(prefect_client: Any) -> None: ...

async def test_deleting_concurrency_limits(prefect_client: Any) -> None: ...

async def test_create_then_read_flow_run(prefect_client: Any) -> None: ...

async def test_create_flow_run_retains_parameters(prefect_client: Any) -> None: ...

async def test_create_flow_run_with_state(prefect_client: Any) -> None: ...

async def test_set_then_read_flow_run_state(prefect_client: Any) -> None: ...

async def test_set_flow_run_state_404_is_object_not_found(
    prefect_client: Any,
) -> None: ...

async def test_read_flow_runs_without_filter(prefect_client: Any) -> None: ...

async def test_read_flow_runs_with_filtering(prefect_client: Any) -> None: ...

@pytest.mark.parametrize(
    'run_tags,filter_tags,expected_match',
    [
        (['tag-1'], ['tag-1'], True),
        (['tag-2'], ['tag-1'], False),
        (['tag-1', 'tag-2'], ['tag-1', 'tag-3'], True),
        (['tag-1'], ['tag-1', 'tag-2'], True),
        (['tag-2'], ['tag-1', 'tag-2'], True),
        (['tag-1'], ['tag-2', 'tag-3'], False),
        (['tag-1'], ['get-real'], False),
        ([], ['tag-1'], False),
        (['tag-1'], [], False),
    ],
)
async def test_read_flow_runs_with_tags(
    prefect_client: Any,
    run_tags: List[str],
    filter_tags: List[str],
    expected_match: bool,
) -> None: ...

async def test_read_flows_without_filter(prefect_client: Any) -> None: ...

async def test_read_flows_with_filter(prefect_client: Any) -> None: ...

async def test_read_flow_by_name(prefect_client: Any) -> None: ...

async def test_create_flow_run_from_deployment(
    prefect_client: Any,
    deployment: Any,
) -> None: ...

async def test_create_flow_run_from_deployment_idempotency(
    prefect_client: Any,
    deployment: Any,
) -> None: ...

async def test_create_flow_run_from_deployment_with_options(
    prefect_client: Any,
    deployment: Any,
) -> None: ...

async def test_update_flow_run(prefect_client: Any) -> None: ...

async def test_update_flow_run_overrides_tags(prefect_client: Any) -> None: ...

async def test_create_then_read_task_run(prefect_client: Any) -> None: ...

async def test_delete_task_run(prefect_client: Any) -> None: ...

async def test_create_then_read_task_run_with_state(
    prefect_client: Any,
) -> None: ...

async def test_set_then_read_task_run_state(prefect_client: Any) -> None: ...

async def test_create_then_read_autonomous_task_runs(
    prefect_client: Any,
) -> None: ...

async def test_create_then_read_flow_run_notification_policy(
    prefect_client: Any,
    block_document: Any,
) -> None: ...

async def test_create_then_update_flow_run_notification_policy(
    prefect_client: Any,
    block_document: Any,
) -> None: ...

async def test_create_then_delete_flow_run_notification_policy(
    prefect_client: Any,
    block_document: Any,
) -> None: ...

async def test_read_filtered_logs(
    session: Any,
    prefect_client: Any,
    deployment: Any,
) -> None: ...

async def test_prefect_api_tls_insecure_skip_verify_setting_set_to_true(
    monkeypatch: Any,
) -> None: ...

async def test_prefect_api_tls_insecure_skip_verify_setting_set_to_false(
    monkeypatch: Any,
) -> None: ...

async def test_prefect_api_tls_insecure_skip_verify_default_setting(
    monkeypatch: Any,
) -> None: ...

async def test_prefect_api_ssl_cert_file_setting_explicitly_set(
    monkeypatch: Any,
) -> None: ...

async def test_prefect_api_ssl_cert_file_default_setting(
    monkeypatch: Any,
) -> None: ...

async def test_prefect_api_ssl_cert_file_default_setting_fallback(
    monkeypatch: Any,
) -> None: ...

class TestClientAPIVersionRequests:
    @pytest.fixture
    def versions(self) -> Any: ...
    @pytest.fixture
    def major_version(self, versions: Any) -> Any: ...
    @pytest.fixture
    def minor_version(self, versions: Any) -> Any: ...
    @pytest.fixture
    def patch_version(self, versions: Any) -> Any: ...
    async def test_default_requests_succeeds(self) -> None: ...
    async def test_no_api_version_header_succeeds(self) -> None: ...
    async def test_major_version(
        self,
        app: Any,
        major_version: Any,
        minor_version: Any,
        patch_version: Any,
    ) -> None: ...
    @pytest.mark.skip(reason='This test is no longer compatible with the current API version checking logic')
    async def test_minor_version(
        self,
        app: Any,
        major_version: Any,
        minor_version: Any,
        patch_version: Any,
    ) -> None: ...
    @pytest.mark.skip(reason='This test is no longer compatible with the current API version checking logic')
    async def test_patch_version(
        self,
        app: Any,
        major_version: Any,
        minor_version: Any,
        patch_version: Any,
    ) -> None: ...
    async def test_invalid_header(self, app: Any) -> None: ...

class TestClientAPIKey:
    @pytest.fixture
    async def test_app(self) -> Any: ...
    async def test_client_passes_api_key_as_auth_header(
        self,
        test_app: Any,
    ) -> None: ...
    async def test_client_no_auth_header_without_api_key(
        self,
        test_app: Any,
    ) -> None: ...
    async def test_get_client_includes_api_key_from_context(self) -> None: ...

class TestClientAuthString:
    @pytest.fixture
    async def test_app(self) -> Any: ...
    async def test_client_passes_auth_string_as_auth_header(
        self,
        test_app: Any,
    ) -> None: ...
    async def test_client_no_auth_header_without_auth_string(
        self,
        test_app: Any,
    ) -> None: ...
    async def test_get_client_includes_auth_string_from_context(self) -> None: ...

class TestClientWorkQueues:
    @pytest.fixture
    async def deployment(self, prefect_client: Any) -> Any: ...
    async def test_create_then_read_work_queue(
        self,
        prefect_client: Any,
    ) -> None: ...
    async def test_create_and_read_includes_status(
        self,
        prefect_client: Any,
    ) -> None: ...
    async def test_create_then_read_work_queue_by_name(
        self,
        prefect_client: Any,
    ) -> None: ...
    async def test_create_queue_with_settings(
        self,
        prefect_client: Any,
    ) -> None: ...
    async def test_create_then_match_work_queues(
        self,
        prefect_client: Any,
    ) -> None: ...
    async def test_read_nonexistant_work_queue(
        self,
        prefect_client: Any,
    ) -> None: ...
    async def test_get_runs_from_queue_includes(
        self,
        prefect_client: Any,
        deployment: Any,
    ) -> None: ...
    async def test_get_runs_from_queue_respects_limit(
        self,
        prefect_client: Any,
        deployment: Any,
    ) -> None: ...

async def test_delete_flow_run(
    prefect_client: Any,
    flow_run: Any,
) -> None: ...

def test_server_type_ephemeral(enable_ephemeral_server: Any) -> None: ...

async def test_server_type_server(hosted_api_server: Any) -> None: ...

async def test_server_type_cloud() -> None: ...

@pytest.mark.parametrize(
    'on_create, expected_value',
    [(True, True), (False, False), (None, False)],
)
async def test_update_deployment_does_not_overwrite_paused_when_not_provided(
    prefect_client: Any,
    flow_run: Any,
    on_create: Optional[bool],
    expected_value: bool,
) -> None: ...

@pytest.mark.parametrize(
    'on_create, after_create, on_update, after_update',
    [
        (False, False, True, True),
        (True, True, False, False),
        (None, False, True, True),
    ],
)
async def test_update_deployment_paused(
    prefect_client: Any,
    flow_run: Any,
    on_create: Optional[bool],
    after_create: bool,
    on_update: Optional[bool],
    after_update: bool,
) -> None: ...

class TestWorkPools:
    async def test_read_work_pools(self, prefect_client: Any) -> None: ...
    async def test_create_work_pool_overwriting_existing_work_pool(
        self,
        prefect_client: Any,
        work_pool: Any,
    ) -> None: ...
    async def test_create_work_pool_with_attempt_to_overwrite_type(
        self,
        prefect_client: Any,
        work_pool: Any,
    ) -> None: ...
    async def test_update_work_pool(self, pre