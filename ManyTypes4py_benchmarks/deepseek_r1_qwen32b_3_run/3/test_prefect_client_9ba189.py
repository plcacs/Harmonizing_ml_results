import json
import os
import ssl
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Generator, List
from unittest import mock
from unittest.mock import MagicMock, Mock
from uuid import UUID, uuid4
import anyio
import certifi
import httpcore
import httpx
import pendulum
import pydantic
import pytest
import respx
from fastapi import Depends, FastAPI, status
from fastapi.security import HTTPBasic, HTTPBearer
import prefect.client.schemas as client_schemas
import prefect.context
import prefect.exceptions
import prefect.server.api
from prefect import flow, tags
from prefect.client.constants import SERVER_API_VERSION
from prefect.client.orchestration import PrefectClient, ServerType, SyncPrefectClient, get_client
from prefect.client.schemas.actions import ArtifactCreate, BlockDocumentCreate, DeploymentScheduleCreate, GlobalConcurrencyLimitCreate, GlobalConcurrencyLimitUpdate, LogCreate, VariableCreate, WorkPoolCreate, WorkPoolUpdate
from prefect.client.schemas.filters import ArtifactFilter, ArtifactFilterKey, DeploymentFilter, DeploymentFilterTags, FlowFilter, FlowRunFilter, FlowRunFilterTags, FlowRunNotificationPolicyFilter, LogFilter, LogFilterFlowRunId, TaskRunFilter, TaskRunFilterFlowRunId
from prefect.client.schemas.objects import Flow, FlowRunNotificationPolicy, FlowRunPolicy, Integration, StateType, TaskRun, Variable, WorkerMetadata, WorkQueue
from prefect.client.schemas.responses import DeploymentResponse, OrchestrationResult, SetStateStatus
from prefect.client.schemas.schedules import CronSchedule, IntervalSchedule
from prefect.client.utilities import inject_client
from prefect.events import AutomationCore, EventTrigger, Posture
from prefect.server.api.server import create_app
from prefect.server.database.orm_models import WorkPool
from prefect.settings import PREFECT_API_AUTH_STRING, PREFECT_API_DATABASE_MIGRATE_ON_START, PREFECT_API_KEY, PREFECT_API_SSL_CERT_FILE, PREFECT_API_TLS_INSECURE_SKIP_VERIFY, PREFECT_API_URL, PREFECT_CLIENT_CSRF_SUPPORT_ENABLED, PREFECT_CLOUD_API_URL, PREFECT_TESTING_UNIT_TEST_MODE, temporary_settings
from prefect.states import Completed, Pending, Running, Scheduled, State
from prefect.tasks import task
from prefect.testing.utilities import AsyncMock, exceptions_equal
from prefect.types import DateTime
from prefect.utilities.pydantic import parse_obj_as

class TestGetClient:

    def test_get_client_returns_client(self) -> None:
        assert isinstance(get_client(), PrefectClient)

    def test_get_client_does_not_cache_client(self) -> None:
        assert get_client() is not get_client()

    def test_get_client_cache_uses_profile_settings(self) -> None:
        client = get_client()
        with temporary_settings(updates={PREFECT_API_KEY: 'FOO'}):
            new_client = get_client()
            assert isinstance(new_client, PrefectClient)
            assert new_client is not client

    def test_get_client_starts_subprocess_server_when_enabled(self, enable_ephemeral_server, monkeypatch) -> None:
        subprocess_server_mock = MagicMock()
        monkeypatch.setattr(prefect.server.api.server, 'SubprocessASGIServer', subprocess_server_mock)
        get_client()
        assert subprocess_server_mock.call_count == 1
        assert subprocess_server_mock.return_value.start.call_count == 1

    def test_get_client_rasises_error_when_no_api_url_and_no_ephemeral_mode(self, disable_hosted_api_server) -> None:
        with pytest.raises(ValueError, match='API URL'):
            get_client()

class TestClientProxyAwareness:

    @pytest.fixture()
    def remote_https_api(self) -> Generator[httpx.URL, None, None]:
        api_url = 'https://127.0.0.1:4242/'
        with temporary_settings(updates={PREFECT_API_URL: api_url}):
            yield httpx.URL(api_url)

    def test_unproxied_remote_client_will_retry(self, remote_https_api) -> None:
        httpx_client = get_client()._client
        assert isinstance(httpx_client, httpx.AsyncClient)
        transport_for_api = httpx_client._transport_for_url(remote_https_api)
        assert isinstance(transport_for_api, httpx.AsyncHTTPTransport)
        pool = transport_for_api._pool
        assert isinstance(pool, httpcore.AsyncConnectionPool)
        assert pool._retries == 3

    def test_users_can_still_provide_transport(self, remote_https_api) -> None:
        httpx_settings = {'transport': httpx.AsyncHTTPTransport(retries=11)}
        httpx_client = get_client(httpx_settings)._client
        assert isinstance(httpx_client, httpx.AsyncClient)
        transport_for_api = httpx_client._transport_for_url(remote_https_api)
        assert isinstance(transport_for_api, httpx.AsyncHTTPTransport)
        pool = transport_for_api._pool
        assert isinstance(pool, httpcore.AsyncConnectionPool)
        assert pool._retries == 11

    @pytest.fixture
    def https_proxy(self) -> Generator[httpcore.URL, None, None]:
        original = os.environ.get('HTTPS_PROXY')
        try:
            os.environ['HTTPS_PROXY'] = 'https://127.0.0.1:6666'
            yield httpcore.URL(os.environ['HTTPS_PROXY'])
        finally:
            if original is None:
                del os.environ['HTTPS_PROXY']
            else:
                os.environ['HTTPS_PROXY'] = original

    async def test_client_is_aware_of_https_proxy(self, remote_https_api, https_proxy) -> None:
        httpx_client = get_client()._client
        assert isinstance(httpx_client, httpx.AsyncClient)
        transport_for_api = httpx_client._transport_for_url(remote_https_api)
        assert isinstance(transport_for_api, httpx.AsyncHTTPTransport)
        pool = transport_for_api._pool
        assert isinstance(pool, httpcore.AsyncHTTPProxy)
        assert pool._proxy_url == https_proxy
        assert pool._retries == 3

    @pytest.fixture()
    def remote_http_api(self) -> Generator[httpx.URL, None, None]:
        api_url = 'http://127.0.0.1:4242/'
        with temporary_settings(updates={PREFECT_API_URL: api_url}):
            yield httpx.URL(api_url)

    @pytest.fixture
    def http_proxy(self) -> Generator[httpcore.URL, None, None]:
        original = os.environ.get('HTTP_PROXY')
        try:
            os.environ['HTTP_PROXY'] = 'http://127.0.0.1:6666'
            yield httpcore.URL(os.environ['HTTP_PROXY'])
        finally:
            if original is None:
                del os.environ['HTTP_PROXY']
            else:
                os.environ['HTTP_PROXY'] = original

    async def test_client_is_aware_of_http_proxy(self, remote_http_api, http_proxy) -> None:
        httpx_client = get_client()._client
        assert isinstance(httpx_client, httpx.AsyncClient)
        transport_for_api = httpx_client._transport_for_url(remote_http_api)
        assert isinstance(transport_for_api, httpx.AsyncHTTPTransport)
        pool = transport_for_api._pool
        assert isinstance(pool, httpcore.AsyncHTTPProxy)
        assert pool._proxy_url == http_proxy
        assert pool._retries == 3

class TestInjectClient:

    @staticmethod
    @inject_client
    async def injected_func(client: PrefectClient) -> PrefectClient:
        assert client._started, 'Client should be started during function'
        assert not client._closed, 'Client should be closed during function'
        await client.api_healthcheck()
        return client

    async def test_get_new_client(self) -> None:
        client = await TestInjectClient.injected_func()
        assert isinstance(client, PrefectClient)
        assert client._closed, 'Client should be closed after function returns'

    async def test_get_new_client_with_explicit_none(self) -> None:
        client = await TestInjectClient.injected_func(client=None)
        assert isinstance(client, PrefectClient)
        assert client._closed, 'Client should be closed after function returns'

    async def test_use_existing_client(self, prefect_client: PrefectClient) -> None:
        client = await TestInjectClient.injected_func(client=prefect_client)
        assert client is prefect_client, 'Client should be the same object'
        assert not client._closed, 'Client should not be closed after function returns'

    async def test_use_existing_client_from_flow_run_ctx(self, prefect_client: PrefectClient) -> None:
        with prefect.context.FlowRunContext.model_construct(client=prefect_client):
            client = await TestInjectClient.injected_func()
        assert client is prefect_client, 'Client should be the same object'
        assert not client._closed, 'Client should not be closed after function returns'

    async def test_use_existing_client_from_task_run_ctx(self, prefect_client: PrefectClient) -> None:
        with prefect.context.FlowRunContext.model_construct(client=prefect_client):
            client = await TestInjectClient.injected_func()
        assert client is prefect_client, 'Client should be the same object'
        assert not client._closed, 'Client should not be closed after function returns'

    async def test_use_existing_client_from_flow_run_ctx_with_null_kwarg(self, prefect_client: PrefectClient) -> None:
        with prefect.context.FlowRunContext.model_construct(client=prefect_client):
            client = await TestInjectClient.injected_func(client=None)
        assert client is prefect_client, 'Client should be the same object'
        assert not client._closed, 'Client should not be closed after function returns'

def not_enough_open_files() -> bool:
    try:
        import resource
    except ImportError:
        return False
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    return soft_limit < 512 or hard_limit < 512

def make_lifespan(startup: callable, shutdown: callable) -> callable:
    async def lifespan(app: FastAPI) -> Generator[None, None, None]:
        try:
            startup()
            yield
        finally:
            shutdown()
    return asynccontextmanager(lifespan)

class TestClientContextManager:

    async def test_client_context_can_be_reentered(self) -> None:
        client = PrefectClient('http://foo.test')
        client._exit_stack.__aenter__ = AsyncMock()
        client._exit_stack.__aexit__ = AsyncMock()
        assert client._exit_stack.__aenter__.call_count == 0
        assert client._exit_stack.__aexit__.call_count == 0
        async with client as c1:
            async with client as c2:
                assert c1 is c2
        assert client._exit_stack.__aenter__.call_count == 1
        assert client._exit_stack.__aexit__.call_count == 1

    async def test_client_context_cannot_be_reused(self) -> None:
        client = PrefectClient('http://foo.test')
        async with client:
            pass
        with pytest.raises(RuntimeError, match='cannot be started again after closing'):
            async with client:
                pass

    async def test_client_context_manages_app_lifespan(self) -> None:
        startup, shutdown = (MagicMock(), MagicMock())
        app = FastAPI(lifespan=make_lifespan(startup, shutdown))
        client = PrefectClient(app)
        startup.assert_not_called()
        shutdown.assert_not_called()
        async with client:
            startup.assert_called_once()
            shutdown.assert_not_called()
        startup.assert_called_once()
        shutdown.assert_called_once()

    async def test_client_context_calls_app_lifespan_once_despite_nesting(self) -> None:
        startup, shutdown = (MagicMock(), MagicMock())
        app = FastAPI(lifespan=make_lifespan(startup, shutdown))
        startup.assert_not_called()
        shutdown.assert_not_called()
        async with PrefectClient(app):
            async with PrefectClient(app):
                async with PrefectClient(app):
                    startup.assert_called_once()
            shutdown.assert_not_called()
        startup.assert_called_once()
        shutdown.assert_called_once()

    async def test_client_context_manages_app_lifespan_on_sequential_usage(self) -> None:
        startup, shutdown = (MagicMock(), MagicMock())
        app = FastAPI(lifespan=make_lifespan(startup, shutdown))
        async with PrefectClient(app):
            pass
        assert startup.call_count == 1
        assert shutdown.call_count == 1
        async with PrefectClient(app):
            assert startup.call_count == 2
            assert shutdown.call_count == 1
        assert startup.call_count == 2
        assert shutdown.call_count == 2

    async def test_client_context_lifespan_is_robust_to_async_concurrency(self) -> None:
        startup = MagicMock(side_effect=lambda: print('Startup called!'))
        shutdown = MagicMock(side_effect=lambda: print('Shutdown called!!'))
        app = FastAPI(lifespan=make_lifespan(startup, shutdown))
        one_started = anyio.Event()
        one_exited = anyio.Event()
        two_started = anyio.Event()

        async def one():
            async with PrefectClient(app):
                print('Started one')
                one_started.set()
                startup.assert_called_once()
                shutdown.assert_not_called()
                print('Waiting for two to start...')
                await two_started.wait()
                print('Exiting one...')
            one_exited.set()

        async def two():
            await one_started.wait()
            async with PrefectClient(app):
                print('Started two')
                two_started.set()
                await anyio.sleep(1)
                startup.assert_called_once()
                shutdown.assert_not_called()
                print('Exiting two...')
        with anyio.fail_after(5):
            async with anyio.create_task_group() as tg:
                tg.start_soon(one)
                tg.start_soon(two)
        startup.assert_called_once()
        shutdown.assert_called_once()

    async def test_client_context_lifespan_is_robust_to_dependency_deadlocks(self) -> None:
        startup = MagicMock(side_effect=lambda: print('Startup called!'))
        shutdown = MagicMock(side_effect=lambda: print('Shutdown called!!'))
        app = FastAPI(lifespan=make_lifespan(startup, shutdown))
        one_started = anyio.Event()
        one_exited = anyio.Event()
        two_started = anyio.Event()

        async def one():
            async with PrefectClient(app):
                print('Started one')
                one_started.set()
                startup.assert_called_once()
                shutdown.assert_not_called()
                print('Waiting for two to start...')
                await two_started.wait()
                print('Exiting one...')
            one_exited.set()

        async def two():
            await one_started.wait()
            async with PrefectClient(app):
                print('Started two')
                two_started.set()
                await one_exited.wait()
                startup.assert_called_once()
                shutdown.assert_not_called()
                print('Exiting two...')
        with anyio.fail_after(5):
            async with anyio.create_task_group() as tg:
                tg.start_soon(one)
                tg.start_soon(two)
        startup.assert_called_once()
        shutdown.assert_called_once()

    async def test_client_context_manages_app_lifespan_on_exception(self) -> None:
        startup, shutdown = (MagicMock(), MagicMock())
        app = FastAPI(lifespan=make_lifespan(startup, shutdown))
        client = PrefectClient(app)
        with pytest.raises(ValueError):
            async with client:
                raise ValueError()
        startup.assert_called_once()
        shutdown.assert_called_once()

    async def test_client_context_manages_app_lifespan_on_anyio_cancellation(self) -> None:
        startup, shutdown = (MagicMock(), MagicMock())
        app = FastAPI(lifespan=make_lifespan(startup, shutdown))

        async def enter_client(task_status):
            async with PrefectClient(app):
                task_status.started()
                await anyio.sleep_forever()
        async with anyio.create_task_group() as tg:
            await tg.start(enter_client)
            await tg.start(enter_client)
            await tg.start(enter_client)
            tg.cancel_scope.cancel()
        startup.assert_called_once()
        shutdown.assert_called_once()

    async def test_client_context_manages_app_lifespan_on_exception_when_nested(self) -> None:
        startup, shutdown = (MagicMock(), MagicMock())
        app = FastAPI(lifespan=make_lifespan(startup, shutdown))
        with pytest.raises(ValueError):
            async with PrefectClient(app):
                try:
                    async with PrefectClient(app):
                        raise ValueError()
                finally:
                    shutdown.assert_not_called()
        startup.assert_called_once()
        shutdown.assert_called_once()

    async def test_with_without_async_raises_helpful_error(self) -> None:
        with pytest.raises(RuntimeError, match='must be entered with an async context'):
            with PrefectClient('http://foo.test'):
                pass

@pytest.mark.parametrize('enabled', [True, False])
async def test_client_runs_migrations_for_ephemeral_app_only_once(enabled: bool, monkeypatch: MonkeyPatch) -> None:
    with temporary_settings(updates={PREFECT_API_DATABASE_MIGRATE_ON_START: enabled}):
        monkeypatch.setattr(prefect.server.api.server, 'LIFESPAN_RAN_FOR_APP', set())
        app = create_app(ephemeral=True, ignore_cache=True)
        mock = AsyncMock()
        monkeypatch.setattr('prefect.server.database.interface.PrefectDBInterface.create_db', mock)
        async with PrefectClient(app):
            if enabled:
                mock.assert_awaited_once_with()
        async with PrefectClient(app):
            if enabled:
                mock.assert_awaited_once_with()
        if not enabled:
            mock.assert_not_awaited()

@pytest.mark.parametrize('enabled', [True, False])
async def test_client_runs_migrations_for_two_different_ephemeral_apps(enabled: bool, monkeypatch: MonkeyPatch) -> None:
    with temporary_settings(updates={PREFECT_API_DATABASE_MIGRATE_ON_START: enabled}):
        monkeypatch.setattr(prefect.server.api.server, 'LIFESPAN_RAN_FOR_APP', set())
        app = create_app(ephemeral=True, ignore_cache=True)
        app2 = create_app(ephemeral=True, ignore_cache=True)
        mock = AsyncMock()
        monkeypatch.setattr('prefect.server.database.interface.PrefectDBInterface.create_db', mock)
        async with PrefectClient(app):
            if enabled:
                mock.assert_awaited_once_with()
        async with PrefectClient(app2):
            if enabled:
                assert mock.await_count == 2
        if not enabled:
            mock.assert_not_awaited()

async def test_client_does_not_run_migrations_for_hosted_app(hosted_api_server: FastAPI, monkeypatch: MonkeyPatch) -> None:
    with temporary_settings(updates={PREFECT_API_DATABASE_MIGRATE_ON_START: True}):
        mock = AsyncMock()
        monkeypatch.setattr('prefect.server.database.interface.PrefectDBInterface.create_db', mock)
        async with PrefectClient(hosted_api_server):
            pass
    mock.assert_not_awaited()

async def test_client_api_url() -> None:
    url = PrefectClient('http://foo.test/bar').api_url
    assert isinstance(url, httpx.URL)
    assert str(url) == 'http://foo.test/bar/'
    assert PrefectClient(FastAPI()).api_url is not None

async def test_hello(prefect_client: PrefectClient) -> None:
    response = await prefect_client.hello()
    assert response.json() == '👋'

async def test_healthcheck(prefect_client: PrefectClient) -> None:
    assert await prefect_client.api_healthcheck() is None

async def test_healthcheck_failure(prefect_client: PrefectClient, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(prefect_client._client, 'get', AsyncMock(side_effect=ValueError('test')))
    assert exceptions_equal(await prefect_client.api_healthcheck(), ValueError('test'))

async def test_create_then_read_flow(prefect_client: PrefectClient) -> None:
    @flow
    def foo():
        pass
    flow_id = await prefect_client.create_flow(foo)
    assert isinstance(flow_id, UUID)
    lookup = await prefect_client.read_flow(flow_id)
    assert isinstance(lookup, Flow)
    assert lookup.name == foo.name

async def test_create_then_delete_flow(prefect_client: PrefectClient) -> None:
    @flow
    def foo():
        pass
    flow_id = await prefect_client.create_flow(foo)
    assert isinstance(flow_id, UUID)
    await prefect_client.delete_flow(flow_id)
    with pytest.raises(prefect.exceptions.PrefectHTTPStatusError, match='404'):
        await prefect_client.read_flow(flow_id)

async def test_create_then_read_deployment(prefect_client: PrefectClient, storage_document_id: UUID) -> None:
    @flow
    def foo():
        pass
    flow_id = await prefect_client.create_flow(foo)
    schedule = DeploymentScheduleCreate(schedule=IntervalSchedule(interval=timedelta(days=1)))
    deployment_id = await prefect_client.create_deployment(flow_id=flow_id, name='test-deployment', version='git-commit-hash', schedules=[schedule], concurrency_limit=42, parameters={'foo': 'bar'}, tags=['foo', 'bar'], storage_document_id=storage_document_id, parameter_openapi_schema={})
    lookup = await prefect_client.read_deployment(deployment_id)
    assert isinstance(lookup, DeploymentResponse)
    assert lookup.name == 'test-deployment'
    assert lookup.version == 'git-commit-hash'
    assert len(lookup.schedules) == 1
    assert lookup.schedules[0].schedule == schedule.schedule
    assert lookup.schedules[0].active == schedule.active
    assert lookup.schedules[0].deployment_id == deployment_id
    assert lookup.global_concurrency_limit.limit == 42
    assert lookup.parameters == {'foo': 'bar'}
    assert lookup.tags == ['foo', 'bar']
    assert lookup.storage_document_id == storage_document_id
    assert lookup.parameter_openapi_schema == {}

async def test_read_deployment_errors_on_invalid_uuid(prefect_client: PrefectClient) -> None:
    with pytest.raises(ValueError, match='Invalid deployment ID: not-a-real-deployment'):
        await prefect_client.read_deployment('not-a-real-deployment')

async def test_update_deployment(prefect_client: PrefectClient, storage_document_id: UUID) -> None:
    @flow
    def foo():
        pass
    flow_id = await prefect_client.create_flow(foo)
    deployment_id = await prefect_client.create_deployment(flow_id=flow_id, name='test-deployment', version='git-commit-hash', parameters={'foo': 'bar'}, tags=['foo', 'bar'], paused=True, storage_document_id=storage_document_id, parameter_openapi_schema={})
    deployment = await prefect_client.read_deployment(deployment_id)
    await prefect_client.update_deployment(deployment_id=deployment_id, deployment=client_schemas.actions.DeploymentUpdate(tags=['new', 'tags'], concurrency_limit=42))
    updated_deployment = await prefect_client.read_deployment(deployment_id)
    assert updated_deployment.tags == ['new', 'tags']
    assert updated_deployment.global_concurrency_limit.limit == 42
    assert updated_deployment.id == deployment.id
    assert updated_deployment.name == deployment.name
    assert updated_deployment.version == deployment.version
    assert updated_deployment.parameters == deployment.parameters
    assert updated_deployment.paused == deployment.paused
    assert updated_deployment.storage_document_id == deployment.storage_document_id
    assert updated_deployment.parameter_openapi_schema == deployment.parameter_openapi_schema

async def test_update_deployment_to_remove_schedules(prefect_client: PrefectClient, storage_document_id: UUID) -> None:
    @flow
    def foo():
        pass
    flow_id = await prefect_client.create_flow(foo)
    schedule = DeploymentScheduleCreate(schedule=IntervalSchedule(interval=timedelta(days=1)))
    deployment_id = await prefect_client.create_deployment(flow_id=flow_id, name='test-deployment', version='git-commit-hash', schedules=[schedule], parameters={'foo': 'bar'}, tags=['foo', 'bar'], storage_document_id=storage_document_id, parameter_openapi_schema={})
    deployment = await prefect_client.read_deployment(deployment_id)
    assert len(deployment.schedules) == 1
    await prefect_client.update_deployment(deployment_id=deployment_id, deployment=client_schemas.actions.DeploymentUpdate(schedules=[]))
    updated_deployment = await prefect_client.read_deployment(deployment_id)
    assert len(updated_deployment.schedules) == 0

async def test_read_deployment_by_name(prefect_client: PrefectClient) -> None:
    @flow
    def foo():
        pass
    flow_id = await prefect_client.create_flow(foo)
    deployment_id = await prefect_client.create_deployment(flow_id=flow_id, name='test-deployment')
    lookup = await prefect_client.read_deployment_by_name('foo/test-deployment')
    assert isinstance(lookup, DeploymentResponse)
    assert lookup.id == deployment_id
    assert lookup.name == 'test-deployment'

@pytest.mark.parametrize('deployment_tags,filter_tags,expected_match', [(['tag-1'], ['tag-1'], True), (['tag-2'], ['tag-1'], False), (['tag-1', 'tag-2'], ['tag-1', 'tag-3'], True), (['tag-1'], ['tag-1', 'tag-2'], True), (['tag-2'], ['tag-1', 'tag-2'], True), (['tag-1'], ['tag-2', 'tag-3'], False), (['tag-1'], ['get-real'], False), ([], ['tag-1'], False), (['tag-1'], [], False)], ids=['single_tag_match', 'single_tag_no_match', 'multiple_tags_partial_match', 'subset_match_1', 'subset_match_2', 'no_matching_tags', 'nonexistent_tag', 'empty_run_tags', 'empty_filter_tags'])
async def test_read_deployment_by_any_tag(prefect_client: PrefectClient, deployment_tags: List[str], filter_tags: List[str], expected_match: bool) -> None:
    @flow
    def moo_deng():
        pass
    flow_id = await prefect_client.create_flow(moo_deng)
    await prefect_client.create_deployment(flow_id=flow_id, name='moisturized-deployment', tags=deployment_tags)
    deployment_responses = await prefect_client.read_deployments(deployment_filter=DeploymentFilter(tags=DeploymentFilterTags(any_=filter_tags)))
    if expected_match:
        assert len(deployment_responses) == 1
        assert deployment_responses[0].name == 'moisturized-deployment'
    else:
        assert len(deployment_responses) == 0

async def test_create_then_delete_deployment(prefect_client: PrefectClient) -> None:
    @flow
    def foo():
        pass
    flow_id = await prefect_client.create_flow(foo)
    deployment_id = await prefect_client.create_deployment(flow_id=flow_id, name='test-deployment')
    await prefect_client.delete_deployment(deployment_id)
    with pytest.raises(prefect.exceptions.ObjectNotFound):
        await prefect_client.read_deployment(deployment_id)

async def test_read_nonexistent_deployment_by_name(prefect_client: PrefectClient) -> None:
    with pytest.raises((prefect.exceptions.ObjectNotFound, ValueError)):
        await prefect_client.read_deployment_by_name('not-a-real-deployment')

async def test_create_then_read_concurrency_limit(prefect_client: PrefectClient) -> None:
    cl_id = await prefect_client.create_concurrency_limit(tag='client-created', concurrency_limit=12345)
    lookup = await prefect_client.read_concurrency_limit_by_tag('client-created')
    assert lookup.id == cl_id
    assert lookup.concurrency_limit == 12345

async def test_read_nonexistent_concurrency_limit_by_tag(prefect_client: PrefectClient) -> None:
    with pytest.raises(prefect.exceptions.ObjectNotFound):
        await prefect_client.read_concurrency_limit_by_tag('not-a-real-tag')

async def test_resetting_concurrency_limits(prefect_client: PrefectClient) -> None:
    await prefect_client.create_concurrency_limit(tag='an-unimportant-limit', concurrency_limit=100)
    await prefect_client.reset_concurrency_limit_by_tag('an-unimportant-limit', slot_override=[uuid4(), uuid4(), uuid4()])
    first_lookup = await prefect_client.read_concurrency_limit_by_tag('an-unimportant-limit')
    assert len(first_lookup.active_slots) == 3
    await prefect_client.reset_concurrency_limit_by_tag('an-unimportant-limit')
    reset_lookup = await prefect_client.read_concurrency_limit_by_tag('an-unimportant-limit')
    assert len(reset_lookup.active_slots) == 0

async def test_deleting_concurrency_limits(prefect_client: PrefectClient) -> None:
    await prefect_client.create_concurrency_limit(tag='dead-limit-walking', concurrency_limit=10)
    assert await prefect_client.read_concurrency_limit_by_tag('dead-limit-walking')
    await prefect_client.delete_concurrency_limit_by_tag('dead-limit-walking')
    with pytest.raises(prefect.exceptions.ObjectNotFound):
        await prefect_client.read_concurrency_limit_by_tag('dead-limit-walking')

async def test_create_then_read_flow_run(prefect_client: PrefectClient) -> None:
    @flow
    def foo():
        pass
    flow_run = await prefect_client.create_flow_run(foo, name='zachs-flow-run')
    assert isinstance(flow_run, client_schemas.FlowRun)
    lookup = await prefect_client.read_flow_run(flow_run.id)
    lookup.estimated_start_time_delta = flow_run.estimated_start_time_delta
    lookup.estimated_run_time = flow_run.estimated_run_time
    assert lookup == flow_run

async def test_create_flow_run_retains_parameters(prefect_client: PrefectClient) -> None:
    @flow
    def foo():
        pass
    parameters = {'x': 1, 'y': [1, 2, 3]}
    flow_run = await prefect_client.create_flow_run(foo, name='zachs-flow-run', parameters=parameters)
    assert parameters == flow_run.parameters, 'Parameter contents are equal'
    assert id(flow_run.parameters) == id(parameters), 'Original objects retained'

async def test_create_flow_run_with_state(prefect_client: PrefectClient) -> None:
    @flow
    def foo():
        pass
    flow_run = await prefect_client.create_flow_run(foo, state=Running())
    assert flow_run.state.is_running()

async def test_set_then_read_flow_run_state(prefect_client: PrefectClient) -> None:
    @flow
    def foo():
        pass
    flow_run_id = (await prefect_client.create_flow_run(foo)).id
    response = await prefect_client.set_flow_run_state(flow_run_id, state=Completed(message='Test!'))
    assert isinstance(response, OrchestrationResult)
    assert response.status == SetStateStatus.ACCEPT
    states = await prefect_client.read_flow_run_states(flow_run_id)
    assert len(states) == 2
    assert states[0].is_pending()
    assert states[1].is_completed()
    assert states[1].message == 'Test!'

async def test_set_flow_run_state_404_is_object_not_found(prefect_client: PrefectClient) -> None:
    @flow
    def foo():
        pass
    await prefect_client.create_flow_run(foo)
    with pytest.raises(prefect.exceptions.ObjectNotFound):
        await prefect_client.set_flow_run_state(uuid4(), state=Completed(message='Test!'))

async def test_read_flow_runs_without_filter(prefect_client: PrefectClient) -> None:
    @flow
    def foo():
        pass
    fr_id_1 = (await prefect_client.create_flow_run(foo)).id
    fr_id_2 = (await prefect_client.create_flow_run(foo)).id
    flow_runs = await prefect_client.read_flow_runs()
    assert len(flow_runs) == 2
    assert all((isinstance(flow_run, client_schemas.FlowRun) for flow_run in flow_runs))
    assert {flow_run.id for flow_run in flow_runs} == {fr_id_1, fr_id_2}

async def test_read_flow_runs_with_filtering(prefect_client: PrefectClient) -> None:
    @flow
    def foo():
        pass

    @flow
    def bar():
        pass
    (await prefect_client.create_flow_run(foo, state=Pending())).id
    (await prefect_client.create_flow_run(foo, state=Scheduled())).id
    (await prefect_client.create_flow_run(bar, state=Pending())).id
    fr_id_4 = (await prefect_client.create_flow_run(bar, state=Scheduled())).id
    fr_id_5 = (await prefect_client.create_flow_run(bar, state=Running())).id
    flow_runs = await prefect_client.read_flow_runs(flow_filter=FlowFilter(name=dict(any_=['bar'])), flow_run_filter=FlowRunFilter(state=dict(type=dict(any_=[StateType.SCHEDULED, StateType.RUNNING]))))
    assert len(flow_runs) == 2
    assert all((isinstance(flow, client_schemas.FlowRun) for flow in flow_runs))
    assert {flow_run.id for flow_run in flow_runs} == {fr_id_4, fr_id_5}

@pytest.mark.parametrize('run_tags,filter_tags,expected_match', [(['tag-1'], ['tag-1'], True), (['tag-2'], ['tag-1'], False), (['tag-1', 'tag-2'], ['tag-1', 'tag-3'], True), (['tag-1'], ['tag-1', 'tag-2'], True), (['tag-2'], ['tag-1', 'tag-2'], True), (['tag-1'], ['tag-2', 'tag-3'], False), (['tag-1'], ['get-real'], False), ([], ['tag-1'], False), (['tag-1'], [], False)], ids=['single_tag_match', 'single_tag_no_match', 'multiple_tags_partial_match', 'subset_match_1', 'subset_match_2', 'no_matching_tags', 'nonexistent_tag', 'empty_run_tags', 'empty_filter_tags'])
async def test_read_flow_runs_with_tags(prefect_client: PrefectClient, run_tags: List[str], filter_tags: List[str], expected_match: bool) -> None:
    @flow
    def foo():
        pass
    flow_run = await prefect_client.create_flow_run(foo, tags=run_tags)
    flow_runs = await prefect_client.read_flow_runs(flow_run_filter=FlowRunFilter(tags=FlowRunFilterTags(any_=filter_tags)))
    if expected_match:
        assert len(flow_runs) == 1
        assert flow_runs[0].id == flow_run.id
    else:
        assert len(flow_runs) == 0

async def test_read_flows_without_filter(prefect_client: PrefectClient) -> None:
    @flow
    def foo():
        pass

    @flow
    def bar():
        pass
    flow_id_1 = await prefect_client.create_flow(foo)
    flow_id_2 = await prefect_client.create_flow(bar)
    flows = await prefect_client.read_flows()
    assert len(flows) == 2
    assert all((isinstance(flow, Flow) for flow in flows))
    assert {flow.id for flow in flows} == {flow_id_1, flow_id_2}

async def test_read_flows_with_filter(prefect_client: PrefectClient) -> None:
    @flow
    def foo():
        pass

    @flow
    def bar():
        pass

    @flow
    def foobar():
        pass
    flow_id_1 = await prefect_client.create_flow(foo)
    flow_id_2 = await prefect_client.create_flow(bar)
    await prefect_client.create_flow(foobar)
    flows = await prefect_client.read_flows(flow_filter=FlowFilter(name=dict(any_=['foo', 'bar'])))
    assert len(flows) == 2
    assert all((isinstance(flow, Flow) for flow in flows))
    assert {flow.id for flow in flows} == {flow_id_1, flow_id_2}

async def test_read_flow_by_name(prefect_client: PrefectClient) -> None:
    @flow(name='null-flow')
    def do_nothing():
        pass
    flow_id = await prefect_client.create_flow(do_nothing)
    the_flow = await prefect_client.read_flow_by_name('null-flow')
    assert the_flow.id == flow_id

async def test_create_flow_run_from_deployment(prefect_client: PrefectClient, deployment: DeploymentResponse) -> None:
    start_time = pendulum.now('utc')
    flow_run = await prefect_client.create_flow_run_from_deployment(deployment.id)
    assert flow_run.deployment_id == deployment.id
    assert flow_run.flow_id == deployment.flow_id
    assert flow_run.work_queue_name == deployment.work_queue_name
    assert flow_run.work_queue_name
    assert flow_run.flow_version is None
    assert flow_run.state.type == StateType.SCHEDULED
    assert start_time <= flow_run.state.state_details.scheduled_time <= pendulum.now('utc')

async def test_create_flow_run_from_deployment_idempotency(prefect_client: PrefectClient, deployment: DeploymentResponse) -> None:
    flow_run_1 = await prefect_client.create_flow_run_from_deployment(deployment.id, idempotency_key='foo')
    flow_run_2 = await prefect_client.create_flow_run_from_deployment(deployment.id, idempotency_key='foo')
    assert flow_run_2.id == flow_run_1.id
    flow_run_3 = await prefect_client.create_flow_run_from_deployment(deployment.id, idempotency_key='bar')
    assert flow_run_3.id != flow_run_1.id

async def test_create_flow_run_from_deployment_with_options(prefect_client: PrefectClient, deployment: DeploymentResponse) -> None:
    job_variables = {'foo': 'bar'}
    flow_run = await prefect_client.create_flow_run_from_deployment(deployment.id, name='test-run-name', tags={'foo', 'bar'}, state=Pending(message='test'), parameters={'foo': 'bar'}, job_variables=job_variables)
    assert flow_run.name == 'test-run-name'
    assert set(flow_run.tags) == {'foo', 'bar'}.union(deployment.tags)
    assert flow_run.state.type == StateType.PENDING
    assert flow_run.state.message == 'test'
    assert flow_run.parameters == {'foo': 'bar'}
    assert flow_run.job_variables == job_variables

async def test_update_flow_run(prefect_client: PrefectClient) -> None:
    @flow
    def foo():
        pass
    flow_run = await prefect_client.create_flow_run(foo)
    exclude = {'updated', 'lateness_estimate', 'estimated_start_time_delta'}
    await prefect_client.update_flow_run(flow_run.id)
    unchanged_flow_run = await prefect_client.read_flow_run(flow_run.id)
    assert unchanged_flow_run.model_dump(exclude=exclude) == flow_run.model_dump(exclude=exclude)
    await prefect_client.update_flow_run(flow_run.id, flow_version='foo', parameters={'foo': 'bar'}, name='test', tags=['hello', 'world'], empirical_policy=FlowRunPolicy(retries=1, retry_delay=2), infrastructure_pid='infrastructure-123:1029')
    updated_flow_run = await prefect_client.read_flow_run(flow_run.id)
    assert updated_flow_run.flow_version == 'foo'
    assert updated_flow_run.parameters == {'foo': 'bar'}
    assert updated_flow_run.name == 'test'
    assert updated_flow_run.tags == ['hello', 'world']
    assert updated_flow_run.empirical_policy == FlowRunPolicy(retries=1, retry_delay=2)
    assert updated_flow_run.infrastructure_pid == 'infrastructure-123:1029'

async def test_update_flow_run_overrides_tags(prefect_client: PrefectClient) -> None:
    @flow(name='test_update_flow_run_tags__flow')
    def hello(name):
        return f'Hello {name}'
    with tags('goodbye', 'cruel', 'world'):
        state = hello('Marvin', return_state=True)
    flow_run = await prefect_client.read_flow_run(state.state_details.flow_run_id)
    await prefect_client.update_flow_run(flow_run.id, tags=['hello', 'world'])
    updated_flow_run = await prefect_client.read_flow_run(flow_run.id)
    assert updated_flow_run.tags == ['hello', 'world']

async def test_create_then_read_task_run(prefect_client: PrefectClient) -> None:
    @flow
    def foo():
        pass

    @task(tags=['a', 'b'], retries=3)
    def bar(prefect_client):
        pass
    flow_run = await prefect_client.create_flow_run(foo)
    task_run = await prefect_client.create_task_run(bar, flow_run_id=flow_run.id, dynamic_key='0')
    assert isinstance(task_run, TaskRun)
    lookup = await prefect_client.read_task_run(task_run.id)
    lookup.estimated_start_time_delta = task_run.estimated_start_time_delta
    lookup.estimated_run_time = task_run.estimated_run_time
    assert lookup == task_run

async def test_delete_task_run(prefect_client: PrefectClient) -> None:
    @task
    def bar():
        pass
    task_run = await prefect_client.create_task_run(bar, flow_run_id=None, dynamic_key='0')
    await prefect_client.delete_task_run(task_run.id)
    with pytest.raises(prefect.exceptions.ObjectNotFound):
        await prefect_client.read_task_run(task_run.id)

async def test_create_then_read_task_run_with_state(prefect_client: PrefectClient) -> None:
    @flow
    def foo():
        pass

    @task(tags=['a', 'b'], retries=3)
    def bar(prefect_client):
        pass
    flow_run = await prefect_client.create_flow_run(foo)
    task_run = await prefect_client.create_task_run(bar, flow_run_id=flow_run.id, state=Running(), dynamic_key='0')
    assert task_run.state.is_running()

async def test_set_then_read_task_run_state(prefect_client: PrefectClient) -> None:
    @flow
    def foo():
        pass

    @task
    def bar(prefect_client):
        pass
    flow_run = await prefect_client.create_flow_run(foo)
    task_run = await prefect_client.create_task_run(bar, flow_run_id=flow_run.id, dynamic_key='0')
    response = await prefect_client.set_task_run_state(task_run.id, Completed(message='Test!'))
    assert isinstance(response, OrchestrationResult)
    assert response.status == SetStateStatus.ACCEPT
    run = await prefect_client.read_task_run(task_run.id)
    assert isinstance(run.state, State)
    assert run.state.type == StateType.COMPLETED
    assert run.state.message == 'Test!'

async def test_create_then_read_autonomous_task_runs(prefect_client: PrefectClient) -> None:
    @task
    def foo():
        pass
    flow_run = await prefect_client.create_flow_run(foo)
    task_run_1 = await prefect_client.create_task_run(foo, flow_run_id=None, dynamic_key='0')
    task_run_2 = await prefect_client.create_task_run(foo, flow_run_id=None, dynamic_key='1')
    task_run_3 = await prefect_client.create_task_run(foo, flow_run_id=flow_run.id, dynamic_key='2')
    assert all((isinstance(task_run, TaskRun) for task_run in [task_run_1, task_run_2, task_run_3]))
    autonotask_runs = await prefect_client.read_task_runs(task_run_filter=TaskRunFilter(flow_run_id=TaskRunFilterFlowRunId(is_null_=True)))
    assert len(autonotask_runs) == 2
    assert {task_run.id for task_run in autonotask_runs} == {task_run_1.id, task_run_2.id}

async def test_create_then_read_flow_run_notification_policy(prefect_client: PrefectClient, block_document: BlockDocumentCreate) -> None:
    message_template = 'Test message template!'
    state_names = ['COMPLETED']
    notification_policy_id = await prefect_client.create_flow_run_notification_policy(block_document_id=block_document.id, is_active=True, tags=[], state_names=state_names, message_template=message_template)
    response = await prefect_client.read_flow_run_notification_policies(FlowRunNotificationPolicyFilter(is_active={'eq_': True}))
    assert len(response) == 1
    assert response[0].id == notification_policy_id
    assert response[0].block_document_id == block_document.id
    assert response[0].message_template == message_template
    assert response[0].is_active
    assert response[0].tags == []
    assert response[0].state_names == state_names

async def test_create_then_update_flow_run_notification_policy(prefect_client: PrefectClient, block_document: BlockDocumentCreate) -> None:
    message_template = 'Updated test message template!'
    state_names = ['FAILED']
    tags = ['1.0']
    notification_policy_id = await prefect_client.create_flow_run_notification_policy(block_document_id=block_document.id, is_active=True, tags=[], state_names=['COMPLETED'], message_template='Test message template!')
    new_block_document = await prefect_client.create_block_document(block_document=BlockDocumentCreate(data={'url': 'http://127.0.0.1'}, block_schema_id=block_document.block_schema_id, block_type_id=block_document.block_type_id, is_anonymous=True))
    await prefect_client.update_flow_run_notification_policy(id=notification_policy_id, block_document_id=new_block_document.id, is_active=False, tags=tags, state_names=state_names, message_template=message_template)
    response = await prefect_client.read_flow_run_notification_policies(FlowRunNotificationPolicyFilter(is_active={'eq_': False}))
    assert len(response) == 1
    assert response[0].id == notification_policy_id
    assert response[0].block_document_id == new_block_document.id
    assert response[0].message_template == message_template
    assert not response[0].is_active
    assert response[0].tags == tags
    assert response[0].state_names == state_names

async def test_create_then_delete_flow_run_notification_policy(prefect_client: PrefectClient, block_document: BlockDocumentCreate) -> None:
    message_template = 'Test message template!'
    state_names = ['COMPLETED']
    notification_policy_id = await prefect_client.create_flow_run_notification_policy(block_document_id=block_document.id, is_active=True, tags=[], state_names=state_names, message_template=message_template)
    await prefect_client.delete_flow_run_notification_policy(notification_policy_id)
    response = await prefect_client.read_flow_run_notification_policies(FlowRunNotificationPolicyFilter(is_active={'eq_': True}))
    assert len(response) == 0

async def test_read_filtered_logs(session, prefect_client: PrefectClient, deployment: DeploymentResponse) -> None:
    flow_runs = [uuid4() for i in range(5)]
    logs = [LogCreate(name='prefect.flow_runs', level=20, message=f'Log from flow_run {id}.', timestamp=DateTime.now(), flow_run_id=id) for id in flow_runs]
    await prefect_client.create_logs(logs)
    logs = await prefect_client.read_logs(log_filter=LogFilter(flow_run_id=LogFilterFlowRunId(any_=flow_runs[:3])))
    for log in logs:
        assert log.flow_run_id in flow_runs[:3]
        assert log.flow_run_id not in flow_runs[3:]

async def test_prefect_api_tls_insecure_skip_verify_setting_set_to_true(monkeypatch: MonkeyPatch) -> None:
    with temporary_settings(updates={PREFECT_API_TLS_INSECURE_SKIP_VERIFY: True}):
        mock = Mock()
        monkeypatch.setattr('prefect.client.orchestration.PrefectHttpxAsyncClient', mock)
        get_client()
    call_kwargs = mock.call_args[1]
    verify_ctx = call_kwargs['verify']
    assert isinstance(verify_ctx, ssl.SSLContext)
    assert verify_ctx.verify_mode == ssl.CERT_NONE
    assert verify_ctx.check_hostname is False

async def test_prefect_api_tls_insecure_skip_verify_setting_set_to_false(monkeypatch: MonkeyPatch) -> None:
    with temporary_settings(updates={PREFECT_API_TLS_INSECURE_SKIP_VERIFY: False}):
        mock = Mock()
        monkeypatch.setattr('prefect.client.orchestration.PrefectHttpxAsyncClient', mock)
        get_client()
    call_kwargs = mock.call_args[1]
    verify_ctx = call_kwargs['verify']
    assert isinstance(verify_ctx, ssl.SSLContext)
    assert verify_ctx.verify_mode == ssl.CERT_REQUIRED
    assert verify_ctx.check_hostname is True

async def test_prefect_api_tls_insecure_skip_verify_default_setting(monkeypatch: MonkeyPatch) -> None:
    mock = Mock()
    monkeypatch.setattr('prefect.client.orchestration.PrefectHttpxAsyncClient', mock)
    get_client()
    call_kwargs = mock.call_args[1]
    verify_ctx = call_kwargs['verify']
    assert isinstance(verify_ctx, ssl.SSLContext)
    assert verify_ctx.verify_mode == ssl.CERT_REQUIRED
    assert verify_ctx.check_hostname is True

async def test_prefect_api_ssl_cert_file_setting_explicitly_set(monkeypatch: MonkeyPatch) -> None:
    cert_path = 'my_cert.pem'
    mock_context = Mock()
    mock_create_default_context = Mock(return_value=mock_context)
    monkeypatch.setattr('ssl.create_default_context', mock_create_default_context)
    with temporary_settings(updates={PREFECT_API_TLS_INSECURE_SKIP_VERIFY: False, PREFECT_API_SSL_CERT_FILE: cert_path}):
        mock_client = Mock()
        monkeypatch.setattr('prefect.client.orchestration.PrefectHttpxAsyncClient', mock_client)
        get_client()
    mock_create_default_context.assert_called_once_with(cafile=cert_path)
    call_kwargs = mock_client.call_args[1]
    verify_ctx = call_kwargs['verify']
    assert verify_ctx == mock_context

async def test_prefect_api_ssl_cert_file_default_setting(monkeypatch: MonkeyPatch) -> None:
    os.environ['SSL_CERT_FILE'] = 'my_cert.pem'
    mock_context = Mock()
    mock_create_default_context = Mock(return_value=mock_context)
    monkeypatch.setattr('ssl.create_default_context', mock_create_default_context)
    with temporary_settings(updates={PREFECT_API_TLS_INSECURE_SKIP_VERIFY: False}, set_defaults={PREFECT_API_SSL_CERT_FILE: os.environ.get('SSL_CERT_FILE')}):
        mock_client = Mock()
        monkeypatch.setattr('prefect.client.orchestration.PrefectHttpxAsyncClient', mock_client)
        get_client()
    mock_create_default_context.assert_called_once_with(cafile='my_cert.pem')
    call_kwargs = mock_client.call_args[1]
    verify_ctx = call_kwargs['verify']
    assert verify_ctx == mock_context

async def test_prefect_api_ssl_cert_file_default_setting_fallback(monkeypatch: MonkeyPatch) -> None:
    os.environ['SSL_CERT_FILE'] = ''
    mock_context = Mock()
    mock_create_default_context = Mock(return_value=mock_context)
    monkeypatch.setattr('ssl.create_default_context', mock_create_default_context)
    with temporary_settings(updates={PREFECT_API_TLS_INSECURE_SKIP_VERIFY: False}, set_defaults={PREFECT_API_SSL_CERT_FILE: os.environ.get('SSL_CERT_FILE')}):
        mock_client = Mock()
        monkeypatch.setattr('prefect.client.orchestration.PrefectHttpxAsyncClient', mock_client)
        get_client()
    mock_create_default_context.assert_called_once_with(cafile=certifi.where())
    call_kwargs = mock_client.call_args[1]
    verify_ctx = call_kwargs['verify']
    assert verify_ctx == mock_context

class TestClientAPIVersionRequests:

    @pytest.fixture
    def versions(self) -> List[str]:
        return SERVER_API_VERSION.split('.')

    @pytest.fixture
    def major_version(self, versions: List[str]) -> int:
        return int(versions[0])

    @pytest.fixture
    def minor_version(self, versions: List[str]) -> int:
        return int(versions[1])

    @pytest.fixture
    def patch_version(self, versions: List[str]) -> int:
        return int(versions[2])

    async def test_default_requests_succeeds(self) -> None:
        async with get_client() as client:
            res = await client.hello()
            assert res.status_code == status.HTTP_200_OK

    async def test_no_api_version_header_succeeds(self) -> None:
        async with get_client() as client:
            client._client.headers = {}
            res = await client.hello()
            assert res.status_code == status.HTTP_200_OK

    async def test_major_version(self, app: FastAPI, major_version: int, minor_version: int, patch_version: int) -> None:
        api_version = f'{major_version + 1}.{minor_version}.{patch_version}'
        async with PrefectClient(app, api_version=api_version) as client:
            res = await client.hello()
            assert res.status_code == status.HTTP_200_OK
        api_version = f'{major_version - 1}.{minor_version}.{patch_version}'
        async with PrefectClient(app, api_version=api_version) as client:
            with pytest.raises(httpx.HTTPStatusError, match=str(status.HTTP_400_BAD_REQUEST)):
                await client.hello()

    @pytest.mark.skip(reason='This test is no longer compatible with the current API version checking logic')
    async def test_minor_version(self, app: FastAPI, major_version: int, minor_version: int, patch_version: int) -> None:
        api_version = f'{major_version}.{minor_version + 1}.{patch_version}'
        async with PrefectClient(app, api_version=api_version) as client:
            res = await client.hello()
            assert res.status_code == status.HTTP_200_OK
        api_version = f'{major_version}.{minor_version - 1}.{patch_version}'
        res = await client.hello()
        async with PrefectClient(app, api_version=api_version) as client:
            with pytest.raises(httpx.HTTPStatusError, match=str(status.HTTP_400_BAD_REQUEST)):
                await client.hello()

    @pytest.mark.skip(reason='This test is no longer compatible with the current API version checking logic')
    async def test_patch_version(self, app: FastAPI, major_version: int, minor_version: int, patch_version: int) -> None:
        api_version = f'{major_version}.{minor_version}.{patch_version + 1}'
        async with PrefectClient(app, api_version=api_version) as client:
            res = await client.hello()
            assert res.status_code == status.HTTP_200_OK
        api_version = f'{major_version}.{minor_version}.{patch_version - 1}'
        res = await client.hello()
        async with PrefectClient(app, api_version=api_version) as client:
            with pytest.raises(httpx.HTTPStatusError, match=str(status.HTTP_400_BAD_REQUEST)):
                await client.hello()

    async def test_invalid_header(self, app: FastAPI) -> None:
        api_version = 'not a real version header'
        async with PrefectClient(app, api_version=api_version) as client:
            with pytest.raises(httpx.HTTPStatusError, match=str(status.HTTP_400_BAD_REQUEST)) as e:
                await client.hello()
            assert 'Invalid X-PREFECT-API-VERSION header format.' in e.value.response.json()['detail']

class TestClientAPIKey:

    @pytest.fixture
    async def test_app(self) -> FastAPI:
        app = FastAPI()
        bearer = HTTPBearer()

        @app.get('/api/check_for_auth_header')
        async def check_for_auth_header(credentials=Depends(bearer)) -> str:
            return credentials.credentials
        return app

    async def test_client_passes_api_key_as_auth_header(self, test_app: FastAPI) -> None:
        api_key = 'validAPIkey'
        async with PrefectClient(test_app, api_key=api_key) as client:
            res = await client._client.get('/check_for_auth_header')
        assert res.status_code == status.HTTP_200_OK
        assert res.json() == api_key

    async def test_client_no_auth_header_without_api_key(self, test_app: FastAPI) -> None:
        async with PrefectClient(test_app) as client:
            with pytest.raises(httpx.HTTPStatusError, match=str(status.HTTP_403_FORBIDDEN)):
                await client._client.get('/check_for_auth_header')

    async def test_get_client_includes_api_key_from_context(self) -> None:
        with temporary_settings(updates={PREFECT_API_KEY: 'test'}):
            client = get_client()
        assert client._client.headers['Authorization'] == 'Bearer test'

class TestClientAuthString:

    @pytest.fixture
    async def test_app(self) -> FastAPI:
        app = FastAPI()
        basic = HTTPBasic()

        @app.get('/api/check_for_auth_header')
        async def check_for_auth_header(credentials=Depends(basic)) -> dict:
            return {'username': credentials.username, 'password': credentials.password}
        return app

    async def test_client_passes_auth_string_as_auth_header(self, test_app: FastAPI) -> None:
        auth_string = 'admin:admin'
        async with PrefectClient(test_app, auth_string=auth_string) as client:
            res = await client._client.get('/check_for_auth_header')
        assert res.status_code == status.HTTP_200_OK
        assert res.json() == {'username': 'admin', 'password': 'admin'}

    async def test_client_no_auth_header_without_auth_string(self, test_app: FastAPI) -> None:
        async with PrefectClient(test_app) as client:
            with pytest.raises(httpx.HTTPStatusError, match='401'):
                await client._client.get('/check_for_auth_header')

    async def test_get_client_includes_auth_string_from_context(self) -> None:
        with temporary_settings(updates={PREFECT_API_AUTH_STRING: 'admin:test'}):
            client = get_client()
        assert client._client.headers['Authorization'].startswith('Basic')

class TestClientWorkQueues:

    @pytest.fixture
    async def deployment(self, prefect_client: PrefectClient) -> DeploymentResponse:
        foo = flow