import json
import os
import ssl
from contextlib import asynccontextmanager, AbstractAsyncContextManager
from datetime import timedelta
from typing import Any, AsyncIterator, Callable, Generator, List, Optional, Tuple, Union
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
)
from prefect.client.schemas.objects import Flow, FlowRunNotificationPolicy, FlowRunPolicy, Integration, StateType, TaskRun, Variable, WorkerMetadata, WorkQueue
from prefect.client.schemas.responses import DeploymentResponse, OrchestrationResult, SetStateStatus
from prefect.client.schemas.schedules import CronSchedule, IntervalSchedule
from prefect.client.utilities import inject_client
from prefect.events import AutomationCore, EventTrigger, Posture
from prefect.server.api.server import create_app
from prefect.server.database.orm_models import WorkPool
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
    temporary_settings,
)
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
        with temporary_settings(updates={PREFECT_API_KEY: "FOO"}):
            new_client = get_client()
            assert isinstance(new_client, PrefectClient)
            assert new_client is not client

    def test_get_client_starts_subprocess_server_when_enabled(self, enable_ephemeral_server: Any, monkeypatch: Any) -> None:
        subprocess_server_mock: Any = MagicMock()
        monkeypatch.setattr(prefect.server.api.server, "SubprocessASGIServer", subprocess_server_mock)
        get_client()
        assert subprocess_server_mock.call_count == 1
        assert subprocess_server_mock.return_value.start.call_count == 1

    def test_get_client_rasises_error_when_no_api_url_and_no_ephemeral_mode(self, disable_hosted_api_server: Any) -> None:
        with pytest.raises(ValueError, match="API URL"):
            get_client()


class TestClientProxyAwareness:
    """Regression test for https://github.com/PrefectHQ/nebula/issues/2356, where
    a customer reported that the Cloud client supported proxies, but the client
    did not.  This test suite is implementation-specific to httpx/httpcore, as there are
    no other inexpensive ways to confirm both the proxy-awareness and preserving the
    retry behavior without probing into the implementation details of the libraries."""

    @pytest.fixture()
    def remote_https_api(self) -> Generator[httpx.URL, None, None]:
        api_url: str = "https://127.0.0.1:4242/"
        with temporary_settings(updates={PREFECT_API_URL: api_url}):
            yield httpx.URL(api_url)

    def test_unproxied_remote_client_will_retry(self, remote_https_api: httpx.URL) -> None:
        httpx_client = get_client()._client
        assert isinstance(httpx_client, httpx.AsyncClient)
        transport_for_api = httpx_client._transport_for_url(remote_https_api)
        assert isinstance(transport_for_api, httpx.AsyncHTTPTransport)
        pool = transport_for_api._pool
        assert isinstance(pool, httpcore.AsyncConnectionPool)
        assert pool._retries == 3

    def test_users_can_still_provide_transport(self, remote_https_api: httpx.URL) -> None:
        httpx_settings: dict = {"transport": httpx.AsyncHTTPTransport(retries=11)}
        httpx_client = get_client(httpx_settings)._client
        assert isinstance(httpx_client, httpx.AsyncClient)
        transport_for_api = httpx_client._transport_for_url(remote_https_api)
        assert isinstance(transport_for_api, httpx.AsyncHTTPTransport)
        pool = transport_for_api._pool
        assert isinstance(pool, httpcore.AsyncConnectionPool)
        assert pool._retries == 11

    @pytest.fixture
    def https_proxy(self) -> Generator[httpcore.URL, None, None]:
        original: Optional[str] = os.environ.get("HTTPS_PROXY")
        try:
            os.environ["HTTPS_PROXY"] = "https://127.0.0.1:6666"
            yield httpcore.URL(os.environ["HTTPS_PROXY"])
        finally:
            if original is None:
                del os.environ["HTTPS_PROXY"]
            else:
                os.environ["HTTPS_PROXY"] = original

    async def test_client_is_aware_of_https_proxy(self, remote_https_api: httpx.URL, https_proxy: httpcore.URL) -> None:
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
        api_url: str = "http://127.0.0.1:4242/"
        with temporary_settings(updates={PREFECT_API_URL: api_url}):
            yield httpx.URL(api_url)

    @pytest.fixture
    def http_proxy(self) -> Generator[httpcore.URL, None, None]:
        original: Optional[str] = os.environ.get("HTTP_PROXY")
        try:
            os.environ["HTTP_PROXY"] = "http://127.0.0.1:6666"
            yield httpcore.URL(os.environ["HTTP_PROXY"])
        finally:
            if original is None:
                del os.environ["HTTP_PROXY"]
            else:
                os.environ["HTTP_PROXY"] = original

    async def test_client_is_aware_of_http_proxy(self, remote_http_api: httpx.URL, http_proxy: httpcore.URL) -> None:
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
    async def injected_func(client: Optional[PrefectClient] = None) -> PrefectClient:
        assert client is not None
        assert client._started, "Client should be started during function"
        assert not client._closed, "Client should be closed during function"
        await client.api_healthcheck()
        return client

    async def test_get_new_client(self) -> None:
        client: PrefectClient = await TestInjectClient.injected_func()
        assert isinstance(client, PrefectClient)
        assert client._closed, "Client should be closed after function returns"

    async def test_get_new_client_with_explicit_none(self) -> None:
        client: PrefectClient = await TestInjectClient.injected_func(client=None)
        assert isinstance(client, PrefectClient)
        assert client._closed, "Client should be closed after function returns"

    async def test_use_existing_client(self, prefect_client: PrefectClient) -> None:
        client: PrefectClient = await TestInjectClient.injected_func(client=prefect_client)
        assert client is prefect_client, "Client should be the same object"
        assert not client._closed, "Client should not be closed after function returns"

    async def test_use_existing_client_from_flow_run_ctx(self, prefect_client: PrefectClient) -> None:
        with prefect.context.FlowRunContext.model_construct(client=prefect_client):
            client: PrefectClient = await TestInjectClient.injected_func()
        assert client is prefect_client, "Client should be the same object"
        assert not client._closed, "Client should not be closed after function returns"

    async def test_use_existing_client_from_task_run_ctx(self, prefect_client: PrefectClient) -> None:
        with prefect.context.FlowRunContext.model_construct(client=prefect_client):
            client: PrefectClient = await TestInjectClient.injected_func()
        assert client is prefect_client, "Client should be the same object"
        assert not client._closed, "Client should not be closed after function returns"

    async def test_use_existing_client_from_flow_run_ctx_with_null_kwarg(self, prefect_client: PrefectClient) -> None:
        with prefect.context.FlowRunContext.model_construct(client=prefect_client):
            client: PrefectClient = await TestInjectClient.injected_func(client=None)
        assert client is prefect_client, "Client should be the same object"
        assert not client._closed, "Client should not be closed after function returns"


def not_enough_open_files() -> bool:
    """
    The current process does not currently allow enough open files for this test.
    You can increase the number of open files with `ulimit -n 512`.
    """
    try:
        import resource
    except ImportError:
        return False
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    return soft_limit < 512 or hard_limit < 512


def make_lifespan(startup: Callable[[], Any], shutdown: Callable[[], Any]) -> AbstractAsyncContextManager[None]:
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            startup()
            yield
        finally:
            shutdown()

    return asynccontextmanager(lifespan)()


class TestClientContextManager:
    async def test_client_context_can_be_reentered(self) -> None:
        client = PrefectClient("http://foo.test")
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
        client = PrefectClient("http://foo.test")
        async with client:
            pass
        with pytest.raises(RuntimeError, match="cannot be started again after closing"):
            async with client:
                pass

    async def test_client_context_manages_app_lifespan(self) -> None:
        startup = MagicMock()
        shutdown = MagicMock()
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
        startup = MagicMock()
        shutdown = MagicMock()
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
        startup = MagicMock()
        shutdown = MagicMock()
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
        startup = MagicMock(side_effect=lambda: print("Startup called!"))
        shutdown = MagicMock(side_effect=lambda: print("Shutdown called!!"))
        app = FastAPI(lifespan=make_lifespan(startup, shutdown))
        one_started: anyio.Event = anyio.Event()
        one_exited: anyio.Event = anyio.Event()
        two_started: anyio.Event = anyio.Event()

        async def one() -> None:
            async with PrefectClient(app):
                print("Started one")
                one_started.set()
                startup.assert_called_once()
                shutdown.assert_not_called()
                print("Waiting for two to start...")
                await two_started.wait()
                print("Exiting one...")
            one_exited.set()

        async def two() -> None:
            await one_started.wait()
            async with PrefectClient(app):
                print("Started two")
                two_started.set()
                await anyio.sleep(1)
                startup.assert_called_once()
                shutdown.assert_not_called()
                print("Exiting two...")

        with anyio.fail_after(5):
            async with anyio.create_task_group() as tg:
                tg.start_soon(one)
                tg.start_soon(two)
        startup.assert_called_once()
        shutdown.assert_called_once()

    async def test_client_context_lifespan_is_robust_to_dependency_deadlocks(self) -> None:
        """
        If you have two concurrrent contexts which are used as follows:

        --> Context A is entered (manages a new lifespan)
        -----> Context B is entered (uses the lifespan from A)
        -----> Context A exits
        -----> Context B exits

        We must ensure that the lifespan shutdown hooks are not called on exit of A and
        wait for all clients to be done consuming them (e.g. after B exits). We must
        also ensure that we do not deadlock by having dependent waits during this
        interleaved case.
        """
        startup = MagicMock(side_effect=lambda: print("Startup called!"))
        shutdown = MagicMock(side_effect=lambda: print("Shutdown called!!"))
        app = FastAPI(lifespan=make_lifespan(startup, shutdown))
        one_started: anyio.Event = anyio.Event()
        one_exited: anyio.Event = anyio.Event()
        two_started: anyio.Event = anyio.Event()

        async def one() -> None:
            async with PrefectClient(app):
                print("Started one")
                one_started.set()
                startup.assert_called_once()
                shutdown.assert_not_called()
                print("Waiting for two to start...")
                await two_started.wait()
                print("Exiting one...")
            one_exited.set()

        async def two() -> None:
            await one_started.wait()
            async with PrefectClient(app):
                print("Started two")
                two_started.set()
                await one_exited.wait()
                startup.assert_called_once()
                shutdown.assert_not_called()
                print("Exiting two...")

        with anyio.fail_after(5):
            async with anyio.create_task_group() as tg:
                tg.start_soon(one)
                tg.start_soon(two)
        startup.assert_called_once()
        shutdown.assert_called_once()

    async def test_client_context_manages_app_lifespan_on_exception(self) -> None:
        startup = MagicMock()
        shutdown = MagicMock()
        app = FastAPI(lifespan=make_lifespan(startup, shutdown))
        client = PrefectClient(app)
        with pytest.raises(ValueError):
            async with client:
                raise ValueError()
        startup.assert_called_once()
        shutdown.assert_called_once()

    async def test_client_context_manages_app_lifespan_on_anyio_cancellation(self) -> None:
        startup = MagicMock()
        shutdown = MagicMock()
        app = FastAPI(lifespan=make_lifespan(startup, shutdown))

        async def enter_client(task_status: Any) -> None:
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
        startup = MagicMock()
        shutdown = MagicMock()
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
        with pytest.raises(RuntimeError, match="must be entered with an async context"):
            with PrefectClient("http://foo.test"):
                pass


@pytest.mark.parametrize("enabled", [True, False])
async def test_client_runs_migrations_for_ephemeral_app_only_once(enabled: bool, monkeypatch: Any) -> None:
    with temporary_settings(updates={PREFECT_API_DATABASE_MIGRATE_ON_START: enabled}):
        monkeypatch.setattr(prefect.server.api.server, "LIFESPAN_RAN_FOR_APP", set())
        app = create_app(ephemeral=True, ignore_cache=True)
        mock = AsyncMock()
        monkeypatch.setattr("prefect.server.database.interface.PrefectDBInterface.create_db", mock)
        async with PrefectClient(app):
            if enabled:
                mock.assert_awaited_once_with()
        async with PrefectClient(app):
            if enabled:
                mock.assert_awaited_once_with()
        if not enabled:
            mock.assert_not_awaited()

@pytest.mark.parametrize("enabled", [True, False])
async def test_client_runs_migrations_for_two_different_ephemeral_apps(enabled: bool, monkeypatch: Any) -> None:
    with temporary_settings(updates={PREFECT_API_DATABASE_MIGRATE_ON_START: enabled}):
        monkeypatch.setattr(prefect.server.api.server, "LIFESPAN_RAN_FOR_APP", set())
        app = create_app(ephemeral=True, ignore_cache=True)
        app2 = create_app(ephemeral=True, ignore_cache=True)
        mock = AsyncMock()
        monkeypatch.setattr("prefect.server.database.interface.PrefectDBInterface.create_db", mock)
        async with PrefectClient(app):
            if enabled:
                mock.assert_awaited_once_with()
        async with PrefectClient(app2):
            if enabled:
                assert mock.await_count == 2
        if not enabled:
            mock.assert_not_awaited()

async def test_client_does_not_run_migrations_for_hosted_app(hosted_api_server: Any, monkeypatch: Any) -> None:
    with temporary_settings(updates={PREFECT_API_DATABASE_MIGRATE_ON_START: True}):
        mock = AsyncMock()
        monkeypatch.setattr("prefect.server.database.interface.PrefectDBInterface.create_db", mock)
        async with PrefectClient(hosted_api_server):
            pass
    mock.assert_not_awaited()

async def test_client_api_url() -> None:
    url: httpx.URL = PrefectClient("http://foo.test/bar").api_url
    assert isinstance(url, httpx.URL)
    assert str(url) == "http://foo.test/bar/"
    assert PrefectClient(FastAPI()).api_url is not None

async def test_hello(prefect_client: PrefectClient) -> None:
    response = await prefect_client.hello()
    assert response.json() == "👋"

async def test_healthcheck(prefect_client: PrefectClient) -> None:
    assert await prefect_client.api_healthcheck() is None

async def test_healthcheck_failure(prefect_client: PrefectClient, monkeypatch: Any) -> None:
    monkeypatch.setattr(prefect_client._client, "get", AsyncMock(side_effect=ValueError("test")))
    assert exceptions_equal(await prefect_client.api_healthcheck(), ValueError("test"))

async def test_create_then_read_flow(prefect_client: PrefectClient) -> None:
    @flow
    def foo() -> None:
        pass
    flow_id: UUID = await prefect_client.create_flow(foo)
    assert isinstance(flow_id, UUID)
    lookup: Flow = await prefect_client.read_flow(flow_id)
    assert isinstance(lookup, Flow)
    assert lookup.name == foo.name

async def test_create_then_delete_flow(prefect_client: PrefectClient) -> None:
    @flow
    def foo() -> None:
        pass
    flow_id: UUID = await prefect_client.create_flow(foo)
    assert isinstance(flow_id, UUID)
    await prefect_client.delete_flow(flow_id)
    with pytest.raises(prefect.exceptions.PrefectHTTPStatusError, match="404"):
        await prefect_client.read_flow(flow_id)

async def test_create_then_read_deployment(prefect_client: PrefectClient, storage_document_id: str) -> None:
    @flow
    def foo() -> None:
        pass
    flow_id: UUID = await prefect_client.create_flow(foo)
    schedule: DeploymentScheduleCreate = DeploymentScheduleCreate(schedule=IntervalSchedule(interval=timedelta(days=1)))
    deployment_id: UUID = await prefect_client.create_deployment(
        flow_id=flow_id,
        name="test-deployment",
        version="git-commit-hash",
        schedules=[schedule],
        concurrency_limit=42,
        parameters={"foo": "bar"},
        tags=["foo", "bar"],
        storage_document_id=storage_document_id,
        parameter_openapi_schema={},
    )
    lookup: DeploymentResponse = await prefect_client.read_deployment(deployment_id)
    assert isinstance(lookup, DeploymentResponse)
    assert lookup.name == "test-deployment"
    assert lookup.version == "git-commit-hash"
    assert len(lookup.schedules) == 1
    assert lookup.schedules[0].schedule == schedule.schedule
    assert lookup.schedules[0].active == schedule.active
    assert lookup.schedules[0].deployment_id == deployment_id
    assert lookup.global_concurrency_limit.limit == 42
    assert lookup.parameters == {"foo": "bar"}
    assert lookup.tags == ["foo", "bar"]
    assert lookup.storage_document_id == storage_document_id
    assert lookup.parameter_openapi_schema == {}

async def test_read_deployment_errors_on_invalid_uuid(prefect_client: PrefectClient) -> None:
    with pytest.raises(ValueError, match="Invalid deployment ID: not-a-real-deployment"):
        await prefect_client.read_deployment("not-a-real-deployment")

async def test_update_deployment(prefect_client: PrefectClient, storage_document_id: str) -> None:
    @flow
    def foo() -> None:
        pass
    flow_id: UUID = await prefect_client.create_flow(foo)
    deployment_id: UUID = await prefect_client.create_deployment(
        flow_id=flow_id,
        name="test-deployment",
        version="git-commit-hash",
        parameters={"foo": "bar"},
        tags=["foo", "bar"],
        paused=True,
        storage_document_id=storage_document_id,
        parameter_openapi_schema={},
    )
    deployment: DeploymentResponse = await prefect_client.read_deployment(deployment_id)
    await prefect_client.update_deployment(
        deployment_id=deployment_id, deployment=client_schemas.actions.DeploymentUpdate(tags=["new", "tags"], concurrency_limit=42)
    )
    updated_deployment: DeploymentResponse = await prefect_client.read_deployment(deployment_id)
    assert updated_deployment.tags == ["new", "tags"]
    assert updated_deployment.global_concurrency_limit.limit == 42
    assert updated_deployment.id == deployment.id
    assert updated_deployment.name == deployment.name
    assert updated_deployment.version == deployment.version
    assert updated_deployment.parameters == deployment.parameters
    assert updated_deployment.paused == deployment.paused
    assert updated_deployment.storage_document_id == deployment.storage_document_id
    assert updated_deployment.parameter_openapi_schema == deployment.parameter_openapi_schema

async def test_update_deployment_to_remove_schedules(prefect_client: PrefectClient, storage_document_id: str) -> None:
    @flow
    def foo() -> None:
        pass
    flow_id: UUID = await prefect_client.create_flow(foo)
    schedule: DeploymentScheduleCreate = DeploymentScheduleCreate(schedule=IntervalSchedule(interval=timedelta(days=1)))
    deployment_id: UUID = await prefect_client.create_deployment(
        flow_id=flow_id,
        name="test-deployment",
        version="git-commit-hash",
        schedules=[schedule],
        parameters={"foo": "bar"},
        tags=["foo", "bar"],
        storage_document_id=storage_document_id,
        parameter_openapi_schema={},
    )
    deployment: DeploymentResponse = await prefect_client.read_deployment(deployment_id)
    assert len(deployment.schedules) == 1
    await prefect_client.update_deployment(deployment_id=deployment_id, deployment=client_schemas.actions.DeploymentUpdate(schedules=[]))
    updated_deployment: DeploymentResponse = await prefect_client.read_deployment(deployment_id)
    assert len(updated_deployment.schedules) == 0

async def test_read_deployment_by_name(prefect_client: PrefectClient) -> None:
    @flow
    def foo() -> None:
        pass
    flow_id: UUID = await prefect_client.create_flow(foo)
    deployment_id: UUID = await prefect_client.create_deployment(flow_id=flow_id, name="test-deployment")
    lookup: DeploymentResponse = await prefect_client.read_deployment_by_name("foo/test-deployment")
    assert isinstance(lookup, DeploymentResponse)
    assert lookup.id == deployment_id
    assert lookup.name == "test-deployment"

@pytest.mark.parametrize(
    "deployment_tags,filter_tags,expected_match",
    [
        (["tag-1"], ["tag-1"], True),
        (["tag-2"], ["tag-1"], False),
        (["tag-1", "tag-2"], ["tag-1", "tag-3"], True),
        (["tag-1"], ["tag-1", "tag-2"], True),
        (["tag-2"], ["tag-1", "tag-2"], True),
        (["tag-1"], ["tag-2", "tag-3"], False),
        (["tag-1"], ["get-real"], False),
        ([], ["tag-1"], False),
        (["tag-1"], [], False),
    ],
    ids=[
        "single_tag_match",
        "single_tag_no_match",
        "multiple_tags_partial_match",
        "subset_match_1",
        "subset_match_2",
        "no_matching_tags",
        "nonexistent_tag",
        "empty_run_tags",
        "empty_filter_tags",
    ],
)
async def test_read_deployment_by_any_tag(
    prefect_client: PrefectClient, deployment_tags: List[str], filter_tags: List[str], expected_match: bool
) -> None:
    @flow
    def moo_deng() -> None:
        pass
    flow_id: UUID = await prefect_client.create_flow(moo_deng)
    await prefect_client.create_deployment(flow_id=flow_id, name="moisturized-deployment", tags=deployment_tags)
    deployment_responses: List[DeploymentResponse] = await prefect_client.read_deployments(
        deployment_filter=DeploymentFilter(tags=DeploymentFilterTags(any_=filter_tags))
    )
    if expected_match:
        assert len(deployment_responses) == 1
        assert deployment_responses[0].name == "moisturized-deployment"
    else:
        assert len(deployment_responses) == 0

async def test_create_then_delete_deployment(prefect_client: PrefectClient) -> None:
    @flow
    def foo() -> None:
        pass
    flow_id: UUID = await prefect_client.create_flow(foo)
    deployment_id: UUID = await prefect_client.create_deployment(flow_id=flow_id, name="test-deployment")
    await prefect_client.delete_deployment(deployment_id)
    with pytest.raises(prefect.exceptions.ObjectNotFound):
        await prefect_client.read_deployment(deployment_id)

async def test_read_nonexistent_deployment_by_name(prefect_client: PrefectClient) -> None:
    with pytest.raises((prefect.exceptions.ObjectNotFound, ValueError)):
        await prefect_client.read_deployment_by_name("not-a-real-deployment")

async def test_create_then_read_concurrency_limit(prefect_client: PrefectClient) -> None:
    cl_id: UUID = await prefect_client.create_concurrency_limit(tag="client-created", concurrency_limit=12345)
    lookup = await prefect_client.read_concurrency_limit_by_tag("client-created")
    assert lookup.id == cl_id
    assert lookup.concurrency_limit == 12345

async def test_read_nonexistent_concurrency_limit_by_tag(prefect_client: PrefectClient) -> None:
    with pytest.raises(prefect.exceptions.ObjectNotFound):
        await prefect_client.read_concurrency_limit_by_tag("not-a-real-tag")

async def test_resetting_concurrency_limits(prefect_client: PrefectClient) -> None:
    await prefect_client.create_concurrency_limit(tag="an-unimportant-limit", concurrency_limit=100)
    await prefect_client.reset_concurrency_limit_by_tag("an-unimportant-limit", slot_override=[uuid4(), uuid4(), uuid4()])
    first_lookup = await prefect_client.read_concurrency_limit_by_tag("an-unimportant-limit")
    assert len(first_lookup.active_slots) == 3
    await prefect_client.reset_concurrency_limit_by_tag("an-unimportant-limit")
    reset_lookup = await prefect_client.read_concurrency_limit_by_tag("an-unimportant-limit")
    assert len(reset_lookup.active_slots) == 0

async def test_deleting_concurrency_limits(prefect_client: PrefectClient) -> None:
    await prefect_client.create_concurrency_limit(tag="dead-limit-walking", concurrency_limit=10)
    assert await prefect_client.read_concurrency_limit_by_tag("dead-limit-walking")
    await prefect_client.delete_concurrency_limit_by_tag("dead-limit-walking")
    with pytest.raises(prefect.exceptions.ObjectNotFound):
        await prefect_client.read_concurrency_limit_by_tag("dead-limit-walking")

async def test_create_then_read_flow_run(prefect_client: PrefectClient) -> None:
    @flow
    def foo() -> None:
        pass
    flow_run = await prefect_client.create_flow_run(foo, name="zachs-flow-run")
    assert isinstance(flow_run, client_schemas.FlowRun)
    lookup = await prefect_client.read_flow_run(flow_run.id)
    lookup.estimated_start_time_delta = flow_run.estimated_start_time_delta
    lookup.estimated_run_time = flow_run.estimated_run_time
    assert lookup == flow_run

async def test_create_flow_run_retains_parameters(prefect_client: PrefectClient) -> None:
    @flow
    def foo() -> None:
        pass
    parameters: dict = {"x": 1, "y": [1, 2, 3]}
    flow_run = await prefect_client.create_flow_run(foo, name="zachs-flow-run", parameters=parameters)
    assert parameters == flow_run.parameters, "Parameter contents are equal"
    assert id(flow_run.parameters) == id(parameters), "Original objects retained"

async def test_create_flow_run_with_state(prefect_client: PrefectClient) -> None:
    @flow
    def foo() -> None:
        pass
    flow_run = await prefect_client.create_flow_run(foo, state=Running())
    assert flow_run.state.is_running()

async def test_set_then_read_flow_run_state(prefect_client: PrefectClient) -> None:
    @flow
    def foo() -> None:
        pass
    flow_run_id: UUID = (await prefect_client.create_flow_run(foo)).id
    response: OrchestrationResult = await prefect_client.set_flow_run_state(flow_run_id, state=Completed(message="Test!"))
    assert isinstance(response, OrchestrationResult)
    assert response.status == SetStateStatus.ACCEPT
    states: List[State] = await prefect_client.read_flow_run_states(flow_run_id)
    assert len(states) == 2
    assert states[0].is_pending()
    assert states[1].is_completed()
    assert states[1].message == "Test!"

async def test_set_flow_run_state_404_is_object_not_found(prefect_client: PrefectClient) -> None:
    @flow
    def foo() -> None:
        pass
    await prefect_client.create_flow_run(foo)
    with pytest.raises(prefect.exceptions.ObjectNotFound):
        await prefect_client.set_flow_run_state(uuid4(), state=Completed(message="Test!"))

async def test_read_flow_runs_without_filter(prefect_client: PrefectClient) -> None:
    @flow
    def foo() -> None:
        pass
    fr_id_1: UUID = (await prefect_client.create_flow_run(foo)).id
    fr_id_2: UUID = (await prefect_client.create_flow_run(foo)).id
    flow_runs: List[client_schemas.FlowRun] = await prefect_client.read_flow_runs()
    assert len(flow_runs) == 2
    assert all((isinstance(flow_run, client_schemas.FlowRun) for flow_run in flow_runs))
    assert {flow_run.id for flow_run in flow_runs} == {fr_id_1, fr_id_2}

async def test_read_flow_runs_with_filtering(prefect_client: PrefectClient) -> None:
    @flow
    def foo() -> None:
        pass

    @flow
    def bar() -> None:
        pass

    (await prefect_client.create_flow_run(foo, state=Pending())).id
    (await prefect_client.create_flow_run(foo, state=Scheduled())).id
    (await prefect_client.create_flow_run(bar, state=Pending())).id
    fr_id_4: UUID = (await prefect_client.create_flow_run(bar, state=Scheduled())).id
    fr_id_5: UUID = (await prefect_client.create_flow_run(bar, state=Running())).id
    flow_runs: List[client_schemas.FlowRun] = await prefect_client.read_flow_runs(
        flow_filter=FlowFilter(name=dict(any_=["bar"])),
        flow_run_filter=FlowRunFilter(state=dict(type=dict(any_=[StateType.SCHEDULED, StateType.RUNNING]))),
    )
    assert len(flow_runs) == 2
    assert all((isinstance(flow, client_schemas.FlowRun) for flow in flow_runs))
    assert {flow_run.id for flow_run in flow_runs} == {fr_id_4, fr_id_5}

@pytest.mark.parametrize(
    "run_tags,filter_tags,expected_match",
    [
        (["tag-1"], ["tag-1"], True),
        (["tag-2"], ["tag-1"], False),
        (["tag-1", "tag-2"], ["tag-1", "tag-3"], True),
        (["tag-1"], ["tag-1", "tag-2"], True),
        (["tag-2"], ["tag-1", "tag-2"], True),
        (["tag-1"], ["tag-2", "tag-3"], False),
        (["tag-1"], ["get-real"], False),
        ([], ["tag-1"], False),
        (["tag-1"], [], False),
    ],
    ids=[
        "single_tag_match",
        "single_tag_no_match",
        "multiple_tags_partial_match",
        "subset_match_1",
        "subset_match_2",
        "no_matching_tags",
        "nonexistent_tag",
        "empty_run_tags",
        "empty_filter_tags",
    ],
)
async def test_read_flow_runs_with_tags(
    prefect_client: PrefectClient, run_tags: List[str], filter_tags: List[str], expected_match: bool
) -> None:
    @flow
    def foo() -> None:
        pass
    flow_run = await prefect_client.create_flow_run(foo, tags=run_tags)
    flow_runs: List[client_schemas.FlowRun] = await prefect_client.read_flow_runs(
        flow_run_filter=FlowRunFilter(tags=FlowRunFilterTags(any_=filter_tags))
    )
    if expected_match:
        assert len(flow_runs) == 1
        assert flow_runs[0].id == flow_run.id
    else:
        assert len(flow_runs) == 0

async def test_read_flows_without_filter(prefect_client: PrefectClient) -> None:
    @flow
    def foo() -> None:
        pass

    @flow
    def bar() -> None:
        pass
    flow_id_1: UUID = await prefect_client.create_flow(foo)
    flow_id_2: UUID = await prefect_client.create_flow(bar)
    flows: List[Flow] = await prefect_client.read_flows()
    assert len(flows) == 2
    assert all((isinstance(flow, Flow) for flow in flows))
    assert {flow.id for flow in flows} == {flow_id_1, flow_id_2}

async def test_read_flows_with_filter(prefect_client: PrefectClient) -> None:
    @flow
    def foo() -> None:
        pass

    @flow
    def bar() -> None:
        pass

    @flow
    def foobar() -> None:
        pass
    flow_id_1: UUID = await prefect_client.create_flow(foo)
    flow_id_2: UUID = await prefect_client.create_flow(bar)
    await prefect_client.create_flow(foobar)
    flows: List[Flow] = await prefect_client.read_flows(flow_filter=FlowFilter(name=dict(any_=["foo", "bar"])))
    assert len(flows) == 2
    assert all((isinstance(flow, Flow) for flow in flows))
    assert {flow.id for flow in flows} == {flow_id_1, flow_id_2}

async def test_read_flow_by_name(prefect_client: PrefectClient) -> None:
    @flow(name="null-flow")
    def do_nothing() -> None:
        pass
    flow_id: UUID = await prefect_client.create_flow(do_nothing)
    the_flow: Flow = await prefect_client.read_flow_by_name("null-flow")
    assert the_flow.id == flow_id

async def test_create_flow_run_from_deployment(prefect_client: PrefectClient, deployment: Any) -> None:
    start_time = pendulum.now("utc")
    flow_run = await prefect_client.create_flow_run_from_deployment(deployment.id)
    assert flow_run.deployment_id == deployment.id
    assert flow_run.flow_id == deployment.flow_id
    assert flow_run.work_queue_name == deployment.work_queue_name
    assert flow_run.work_queue_name
    assert flow_run.flow_version is None
    assert flow_run.state.type == StateType.SCHEDULED
    assert start_time <= flow_run.state.state_details.scheduled_time <= pendulum.now("utc")

async def test_create_flow_run_from_deployment_idempotency(prefect_client: PrefectClient, deployment: Any) -> None:
    flow_run_1 = await prefect_client.create_flow_run_from_deployment(deployment.id, idempotency_key="foo")
    flow_run_2 = await prefect_client.create_flow_run_from_deployment(deployment.id, idempotency_key="foo")
    assert flow_run_2.id == flow_run_1.id
    flow_run_3 = await prefect_client.create_flow_run_from_deployment(deployment.id, idempotency_key="bar")
    assert flow_run_3.id != flow_run_1.id

async def test_create_flow_run_from_deployment_with_options(prefect_client: PrefectClient, deployment: Any) -> None:
    job_variables: dict = {"foo": "bar"}
    flow_run = await prefect_client.create_flow_run_from_deployment(
        deployment.id,
        name="test-run-name",
        tags={"foo", "bar"},
        state=Pending(message="test"),
        parameters={"foo": "bar"},
        job_variables=job_variables,
    )
    assert flow_run.name == "test-run-name"
    assert set(flow_run.tags) == {"foo", "bar"}.union(deployment.tags)
    assert flow_run.state.type == StateType.PENDING
    assert flow_run.state.message == "test"
    assert flow_run.parameters == {"foo": "bar"}
    assert flow_run.job_variables == job_variables

async def test_update_flow_run(prefect_client: PrefectClient) -> None:
    @flow
    def foo() -> None:
        pass
    flow_run = await prefect_client.create_flow_run(foo)
    exclude = {"updated", "lateness_estimate", "estimated_start_time_delta"}
    await prefect_client.update_flow_run(flow_run.id)
    unchanged_flow_run = await prefect_client.read_flow_run(flow_run.id)
    assert unchanged_flow_run.model_dump(exclude=exclude) == flow_run.model_dump(exclude=exclude)
    await prefect_client.update_flow_run(
        flow_run.id,
        flow_version="foo",
        parameters={"foo": "bar"},
        name="test",
        tags=["hello", "world"],
        empirical_policy=FlowRunPolicy(retries=1, retry_delay=2),
        infrastructure_pid="infrastructure-123:1029",
    )
    updated_flow_run = await prefect_client.read_flow_run(flow_run.id)
    assert updated_flow_run.flow_version == "foo"
    assert updated_flow_run.parameters == {"foo": "bar"}
    assert updated_flow_run.name == "test"
    assert updated_flow_run.tags == ["hello", "world"]
    assert updated_flow_run.empirical_policy == FlowRunPolicy(retries=1, retry_delay=2)
    assert updated_flow_run.infrastructure_pid == "infrastructure-123:1029"

async def test_update_flow_run_overrides_tags(prefect_client: PrefectClient) -> None:
    @flow(name="test_update_flow_run_tags__flow")
    def hello(name: str) -> str:
        return f"Hello {name}"
    with tags("goodbye", "cruel", "world"):
        state = hello("Marvin", return_state=True)
    flow_run = await prefect_client.read_flow_run(state.state_details.flow_run_id)
    await prefect_client.update_flow_run(flow_run.id, tags=["hello", "world"])
    updated_flow_run = await prefect_client.read_flow_run(flow_run.id)
    assert updated_flow_run.tags == ["hello", "world"]

async def test_create_then_read_task_run(prefect_client: PrefectClient) -> None:
    @flow
    def foo() -> None:
        pass

    @task(tags=["a", "b"], retries=3)
    def bar(prefect_client: Any) -> None:
        pass
    flow_run = await prefect_client.create_flow_run(foo)
    task_run: TaskRun = await prefect_client.create_task_run(bar, flow_run_id=flow_run.id, dynamic_key="0")
    assert isinstance(task_run, TaskRun)
    lookup: TaskRun = await prefect_client.read_task_run(task_run.id)
    lookup.estimated_start_time_delta = task_run.estimated_start_time_delta
    lookup.estimated_run_time = task_run.estimated_run_time
    assert lookup == task_run

async def test_delete_task_run(prefect_client: PrefectClient) -> None:
    @task
    def bar() -> None:
        pass
    task_run: TaskRun = await prefect_client.create_task_run(bar, flow_run_id=None, dynamic_key="0")
    await prefect_client.delete_task_run(task_run.id)
    with pytest.raises(prefect.exceptions.ObjectNotFound):
        await prefect_client.read_task_run(task_run.id)

async def test_create_then_read_task_run_with_state(prefect_client: PrefectClient) -> None:
    @flow
    def foo() -> None:
        pass

    @task(tags=["a", "b"], retries=3)
    def bar(prefect_client: Any) -> None:
        pass
    flow_run = await prefect_client.create_flow_run(foo)
    task_run: TaskRun = await prefect_client.create_task_run(bar, flow_run_id=flow_run.id, state=Running(), dynamic_key="0")
    assert task_run.state.is_running()

async def test_set_then_read_task_run_state(prefect_client: PrefectClient) -> None:
    @flow
    def foo() -> None:
        pass

    @task
    def bar(prefect_client: Any) -> None:
        pass
    flow_run = await prefect_client.create_flow_run(foo)
    task_run: TaskRun = await prefect_client.create_task_run(bar, flow_run_id=flow_run.id, dynamic_key="0")
    response: OrchestrationResult = await prefect_client.set_task_run_state(task_run.id, Completed(message="Test!"))
    assert isinstance(response, OrchestrationResult)
    assert response.status == SetStateStatus.ACCEPT
    run: TaskRun = await prefect_client.read_task_run(task_run.id)
    assert isinstance(run.state, State)
    assert run.state.type == StateType.COMPLETED
    assert run.state.message == "Test!"

async def test_create_then_read_autonomous_task_runs(prefect_client: PrefectClient) -> None:
    @task
    def foo() -> None:
        pass
    flow_run = await prefect_client.create_flow_run(foo)
    task_run_1: TaskRun = await prefect_client.create_task_run(foo, flow_run_id=None, dynamic_key="0")
    task_run_2: TaskRun = await prefect_client.create_task_run(foo, flow_run_id=None, dynamic_key="1")
    task_run_3: TaskRun = await prefect_client.create_task_run(foo, flow_run_id=flow_run.id, dynamic_key="2")
    assert all((isinstance(task_run, TaskRun) for task_run in [task_run_1, task_run_2, task_run_3]))
    autonotask_runs: List[TaskRun] = await prefect_client.read_task_runs(
        task_run_filter=TaskRunFilter(flow_run_id=TaskRunFilterFlowRunId(is_null_=True))
    )
    assert len(autonotask_runs) == 2
    assert {task_run.id for task_run in autonotask_runs} == {task_run_1.id, task_run_2.id}

async def test_create_then_read_flow_run_notification_policy(prefect_client: PrefectClient, block_document: Any) -> None:
    message_template: str = "Test message template!"
    state_names: List[str] = ["COMPLETED"]
    notification_policy_id: UUID = await prefect_client.create_flow_run_notification_policy(
        block_document_id=block_document.id, is_active=True, tags=[], state_names=state_names, message_template=message_template
    )
    response: List[FlowRunNotificationPolicy] = await prefect_client.read_flow_run_notification_policies(
        FlowRunNotificationPolicyFilter(is_active={"eq_": True})
    )
    assert len(response) == 1
    assert response[0].id == notification_policy_id
    assert response[0].block_document_id == block_document.id
    assert response[0].message_template == message_template
    assert response[0].is_active
    assert response[0].tags == []
    assert response[0].state_names == state_names

async def test_create_then_update_flow_run_notification_policy(prefect_client: PrefectClient, block_document: Any) -> None:
    message_template: str = "Updated test message template!"
    state_names: List[str] = ["FAILED"]
    tags: List[str] = ["1.0"]
    notification_policy_id: UUID = await prefect_client.create_flow_run_notification_policy(
        block_document_id=block_document.id, is_active=True, tags=[], state_names=["COMPLETED"], message_template="Test message template!"
    )
    new_block_document: Any = await prefect_client.create_block_document(
        block_document=BlockDocumentCreate(
            data={"url": "http://127.0.0.1"},
            block_schema_id=block_document.block_schema_id,
            block_type_id=block_document.block_type_id,
            is_anonymous=True,
        )
    )
    await prefect_client.update_flow_run_notification_policy(
        id=notification_policy_id,
        block_document_id=new_block_document.id,
        is_active=False,
        tags=tags,
        state_names=state_names,
        message_template=message_template,
    )
    response: List[FlowRunNotificationPolicy] = await prefect_client.read_flow_run_notification_policies(
        FlowRunNotificationPolicyFilter(is_active={"eq_": False})
    )
    assert len(response) == 1
    assert response[0].id == notification_policy_id
    assert response[0].block_document_id == new_block_document.id
    assert response[0].message_template == message_template
    assert not response[0].is_active
    assert response[0].tags == tags
    assert response[0].state_names == state_names

async def test_create_then_delete_flow_run_notification_policy(prefect_client: PrefectClient, block_document: Any) -> None:
    message_template: str = "Test message template!"
    state_names: List[str] = ["COMPLETED"]
    notification_policy_id: UUID = await prefect_client.create_flow_run_notification_policy(
        block_document_id=block_document.id, is_active=True, tags=[], state_names=state_names, message_template=message_template
    )
    await prefect_client.delete_flow_run_notification_policy(notification_policy_id)
    response: List[FlowRunNotificationPolicy] = await prefect_client.read_flow_run_notification_policies(
        FlowRunNotificationPolicyFilter(is_active={"eq_": True})
    )
    assert len(response) == 0

async def test_read_filtered_logs(session: Any, prefect_client: PrefectClient, deployment: Any) -> None:
    flow_runs: List[UUID] = [uuid4() for i in range(5)]
    logs = [
        LogCreate(
            name="prefect.flow_runs",
            level=20,
            message=f"Log from flow_run {id}.",
            timestamp=DateTime.now(),
            flow_run_id=id,
        )
        for id in flow_runs
    ]
    await prefect_client.create_logs(logs)
    logs = await prefect_client.read_logs(log_filter=LogFilter(flow_run_id=LogFilterFlowRunId(any_=flow_runs[:3])))
    for log in logs:
        assert log.flow_run_id in flow_runs[:3]
        assert log.flow_run_id not in flow_runs[3:]

async def test_prefect_api_tls_insecure_skip_verify_setting_set_to_true(monkeypatch: Any) -> None:
    with temporary_settings(updates={PREFECT_API_TLS_INSECURE_SKIP_VERIFY: True}):
        mock = Mock()
        monkeypatch.setattr("prefect.client.orchestration.PrefectHttpxAsyncClient", mock)
        get_client()
    call_kwargs = mock.call_args[1]
    verify_ctx: ssl.SSLContext = call_kwargs["verify"]
    assert isinstance(verify_ctx, ssl.SSLContext)
    assert verify_ctx.verify_mode == ssl.CERT_NONE
    assert verify_ctx.check_hostname is False

async def test_prefect_api_tls_insecure_skip_verify_setting_set_to_false(monkeypatch: Any) -> None:
    with temporary_settings(updates={PREFECT_API_TLS_INSECURE_SKIP_VERIFY: False}):
        mock = Mock()
        monkeypatch.setattr("prefect.client.orchestration.PrefectHttpxAsyncClient", mock)
        get_client()
    call_kwargs = mock.call_args[1]
    verify_ctx: ssl.SSLContext = call_kwargs["verify"]
    assert isinstance(verify_ctx, ssl.SSLContext)
    assert verify_ctx.verify_mode == ssl.CERT_REQUIRED
    assert verify_ctx.check_hostname is True

async def test_prefect_api_tls_insecure_skip_verify_default_setting(monkeypatch: Any) -> None:
    mock = Mock()
    monkeypatch.setattr("prefect.client.orchestration.PrefectHttpxAsyncClient", mock)
    get_client()
    call_kwargs = mock.call_args[1]
    verify_ctx: ssl.SSLContext = call_kwargs["verify"]
    assert isinstance(verify_ctx, ssl.SSLContext)
    assert verify_ctx.verify_mode == ssl.CERT_REQUIRED
    assert verify_ctx.check_hostname is True

async def test_prefect_api_ssl_cert_file_setting_explicitly_set(monkeypatch: Any) -> None:
    cert_path: str = "my_cert.pem"
    mock_context = Mock()
    mock_create_default_context = Mock(return_value=mock_context)
    monkeypatch.setattr("ssl.create_default_context", mock_create_default_context)
    with temporary_settings(updates={PREFECT_API_TLS_INSECURE_SKIP_VERIFY: False, PREFECT_API_SSL_CERT_FILE: cert_path}):
        mock_client = Mock()
        monkeypatch.setattr("prefect.client.orchestration.PrefectHttpxAsyncClient", mock_client)
        get_client()
    mock_create_default_context.assert_called_once_with(cafile=cert_path)
    call_kwargs = mock_client.call_args[1]
    verify_ctx = call_kwargs["verify"]
    assert verify_ctx == mock_context

async def test_prefect_api_ssl_cert_file_default_setting(monkeypatch: Any) -> None:
    os.environ["SSL_CERT_FILE"] = "my_cert.pem"
    mock_context = Mock()
    mock_create_default_context = Mock(return_value=mock_context)
    monkeypatch.setattr("ssl.create_default_context", mock_create_default_context)
    with temporary_settings(updates={PREFECT_API_TLS_INSECURE_SKIP_VERIFY: False}, set_defaults={PREFECT_API_SSL_CERT_FILE: os.environ.get("SSL_CERT_FILE")}):

        mock_client = Mock()
        monkeypatch.setattr("prefect.client.orchestration.PrefectHttpxAsyncClient", mock_client)
        get_client()
    mock_create_default_context.assert_called_once_with(cafile="my_cert.pem")
    call_kwargs = mock_client.call_args[1]
    verify_ctx = call_kwargs["verify"]
    assert verify_ctx == mock_context

async def test_prefect_api_ssl_cert_file_default_setting_fallback(monkeypatch: Any) -> None:
    os.environ["SSL_CERT_FILE"] = ""
    mock_context = Mock()
    mock_create_default_context = Mock(return_value=mock_context)
    monkeypatch.setattr("ssl.create_default_context", mock_create_default_context)
    with temporary_settings(updates={PREFECT_API_TLS_INSECURE_SKIP_VERIFY: False}, set_defaults={PREFECT_API_SSL_CERT_FILE: os.environ.get("SSL_CERT_FILE")}):

        mock_client = Mock()
        monkeypatch.setattr("prefect.client.orchestration.PrefectHttpxAsyncClient", mock_client)
        get_client()
    mock_create_default_context.assert_called_once_with(cafile=certifi.where())
    call_kwargs = mock_client.call_args[1]
    verify_ctx = call_kwargs["verify"]
    assert verify_ctx == mock_context


class TestClientAPIVersionRequests:
    @pytest.fixture
    def versions(self) -> List[str]:
        return SERVER_API_VERSION.split(".")

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
        api_version: str = f"{major_version + 1}.{minor_version}.{patch_version}"
        async with PrefectClient(app, api_version=api_version) as client:
            res = await client.hello()
            assert res.status_code == status.HTTP_200_OK
        api_version = f"{major_version - 1}.{minor_version}.{patch_version}"
        async with PrefectClient(app, api_version=api_version) as client:
            with pytest.raises(httpx.HTTPStatusError, match=str(status.HTTP_400_BAD_REQUEST)):
                await client.hello()

    @pytest.mark.skip(reason="This test is no longer compatible with the current API version checking logic")
    async def test_minor_version(self, app: FastAPI, major_version: int, minor_version: int, patch_version: int) -> None:
        api_version: str = f"{major_version}.{minor_version + 1}.{patch_version}"
        async with PrefectClient(app, api_version=api_version) as client:
            res = await client.hello()
            assert res.status_code == status.HTTP_200_OK
        api_version = f"{major_version}.{minor_version - 1}.{patch_version}"
        async with PrefectClient(app, api_version=api_version) as client:
            with pytest.raises(httpx.HTTPStatusError, match=str(status.HTTP_400_BAD_REQUEST)):
                await client.hello()

    @pytest.mark.skip(reason="This test is no longer compatible with the current API version checking logic")
    async def test_patch_version(self, app: FastAPI, major_version: int, minor_version: int, patch_version: int) -> None:
        api_version: str = f"{major_version}.{minor_version}.{patch_version + 1}"
        async with PrefectClient(app, api_version=api_version) as client:
            res = await client.hello()
            assert res.status_code == status.HTTP_200_OK
        api_version = f"{major_version}.{minor_version}.{patch_version - 1}"
        async with PrefectClient(app, api_version=api_version) as client:
            with pytest.raises(httpx.HTTPStatusError, match=str(status.HTTP_400_BAD_REQUEST)):
                await client.hello()

    async def test_invalid_header(self, app: FastAPI) -> None:
        api_version: str = "not a real version header"
        async with PrefectClient(app, api_version=api_version) as client:
            with pytest.raises(httpx.HTTPStatusError, match=str(status.HTTP_400_BAD_REQUEST)) as e:
                await client.hello()
            assert "Invalid X-PREFECT-API-VERSION header format." in e.value.response.json()["detail"]


class TestClientAPIKey:
    @pytest.fixture
    async def test_app(self) -> FastAPI:
        app = FastAPI()
        bearer = HTTPBearer()

        @app.get("/api/check_for_auth_header")
        async def check_for_auth_header(credentials=Depends(bearer)) -> str:
            return credentials.credentials

        return app

    async def test_client_passes_api_key_as_auth_header(self, test_app: FastAPI) -> None:
        api_key: str = "validAPIkey"
        async with PrefectClient(test_app, api_key=api_key) as client:
            res = await client._client.get("/check_for_auth_header")
        assert res.status_code == status.HTTP_200_OK
        assert res.json() == api_key

    async def test_client_no_auth_header_without_api_key(self, test_app: FastAPI) -> None:
        async with PrefectClient(test_app) as client:
            with pytest.raises(httpx.HTTPStatusError, match=str(status.HTTP_403_FORBIDDEN)):
                await client._client.get("/check_for_auth_header")

    async def test_get_client_includes_api_key_from_context(self) -> None:
        with temporary_settings(updates={PREFECT_API_KEY: "test"}):
            client = get_client()
        assert client._client.headers["Authorization"] == "Bearer test"


class TestClientAuthString:
    @pytest.fixture
    async def test_app(self) -> FastAPI:
        app = FastAPI()
        basic = HTTPBasic()

        @app.get("/api/check_for_auth_header")
        async def check_for_auth_header(credentials=Depends(basic)) -> dict:
            return {"username": credentials.username, "password": credentials.password}

        return app

    async def test_client_passes_auth_string_as_auth_header(self, test_app: FastAPI) -> None:
        auth_string: str = "admin:admin"
        async with PrefectClient(test_app, auth_string=auth_string) as client:
            res = await client._client.get("/check_for_auth_header")
        assert res.status_code == status.HTTP_200_OK
        assert res.json() == {"username": "admin", "password": "admin"}

    async def test_client_no_auth_header_without_auth_string(self, test_app: FastAPI) -> None:
        async with PrefectClient(test_app) as client:
            with pytest.raises(httpx.HTTPStatusError, match="401"):
                await client._client.get("/check_for_auth_header")

    async def test_get_client_includes_auth_string_from_context(self) -> None:
        with temporary_settings(updates={PREFECT_API_AUTH_STRING: "admin:test"}):
            client = get_client()
        assert client._client.headers["Authorization"].startswith("Basic")


class TestClientWorkQueues:
    @pytest.fixture
    async def deployment(self, prefect_client: PrefectClient) -> UUID:
        foo = flow(lambda: None, name="foo")
        flow_id: UUID = await prefect_client.create_flow(foo)
        schedule: IntervalSchedule = IntervalSchedule(interval=timedelta(days=1), anchor_date=pendulum.datetime(2020, 1, 1))
        deployment_id: UUID = await prefect_client.create_deployment(
            flow_id=flow_id, name="test-deployment", schedules=[DeploymentScheduleCreate(schedule=schedule)], parameters={"foo": "bar"}, work_queue_name="wq"
        )
        return deployment_id

    async def test_create_then_read_work_queue(self, prefect_client: PrefectClient) -> None:
        queue = await prefect_client.create_work_queue(name="foo")
        assert isinstance(queue.id, UUID)
        lookup: WorkQueue = await prefect_client.read_work_queue(queue.id)
        assert isinstance(lookup, WorkQueue)
        assert lookup.name == "foo"

    async def test_create_and_read_includes_status(self, prefect_client: PrefectClient) -> None:
        queue = await prefect_client.create_work_queue(name="foo")
        assert hasattr(queue, "status")
        assert queue.status == "NOT_READY"
        lookup: WorkQueue = await prefect_client.read_work_queue(queue.id)
        assert hasattr(lookup, "status")
        assert lookup.status == "NOT_READY"

    async def test_create_then_read_work_queue_by_name(self, prefect_client: PrefectClient) -> None:
        queue = await prefect_client.create_work_queue(name="foo")
        assert isinstance(queue.id, UUID)
        lookup: WorkQueue = await prefect_client.read_work_queue_by_name("foo")
        assert lookup.name == "foo"

    async def test_create_queue_with_settings(self, prefect_client: PrefectClient) -> None:
        queue = await prefect_client.create_work_queue(name="foo", concurrency_limit=1, is_paused=True, priority=2, description="such queue")
        assert queue.concurrency_limit == 1
        assert queue.is_paused is True
        assert queue.priority == 2
        assert queue.description == "such queue"

    async def test_create_then_match_work_queues(self, prefect_client: PrefectClient) -> None:
        await prefect_client.create_work_queue(name="one of these things is not like the other")
        await prefect_client.create_work_queue(name="one of these things just doesn't belong")
        await prefect_client.create_work_queue(name="can you tell which thing is not like the others")
        matched_queues = await prefect_client.match_work_queues(["one of these things"])
        assert len(matched_queues) == 2

    async def test_read_nonexistant_work_queue(self, prefect_client: PrefectClient) -> None:
        with pytest.raises(prefect.exceptions.ObjectNotFound):
            await prefect_client.read_work_queue_by_name("foo")

    async def test_get_runs_from_queue_includes(self, prefect_client: PrefectClient, deployment: UUID) -> None:
        wq_1 = await prefect_client.read_work_queue_by_name(name="wq")
        wq_2 = await prefect_client.create_work_queue(name="wq2")
        run = await prefect_client.create_flow_run_from_deployment(deployment)
        assert run.id
        runs_1 = await prefect_client.get_runs_in_work_queue(wq_1.id)
        assert runs_1[0].id == run.id
        runs_2 = await prefect_client.get_runs_in_work_queue(wq_2.id)
        assert runs_2 == []

    async def test_get_runs_from_queue_respects_limit(self, prefect_client: PrefectClient, deployment: UUID) -> None:
        queue = await prefect_client.read_work_queue_by_name(name="wq")
        runs: List[Any] = []
        for _ in range(10):
            run = await prefect_client.create_flow_run_from_deployment(deployment)
            runs.append(run)
        output = await prefect_client.get_runs_in_work_queue(queue.id, limit=1)
        assert len(output) == 1
        assert output[0].id in [r.id for r in runs]
        output = await prefect_client.get_runs_in_work_queue(queue.id, limit=8)
        assert len(output) == 8
        assert {o.id for o in output} < {r.id for r in runs}
        output = await prefect_client.get_runs_in_work_queue(queue.id, limit=20)
        assert len(output) == 10
        assert {o.id for o in output} == {r.id for r in runs}


async def test_delete_flow_run(prefect_client: PrefectClient, flow_run: client_schemas.FlowRun) -> None:
    print(f"Type: {type(flow_run)}")
    lookup: client_schemas.FlowRun = await prefect_client.read_flow_run(flow_run.id)
    assert isinstance(lookup, client_schemas.FlowRun)
    await prefect_client.delete_flow_run(flow_run.id)
    with pytest.raises(prefect.exceptions.ObjectNotFound):
        await prefect_client.read_flow_run(flow_run.id)
    with pytest.raises(prefect.exceptions.ObjectNotFound):
        await prefect_client.delete_flow_run(flow_run.id)

def test_server_type_ephemeral(enable_ephemeral_server: Any) -> None:
    prefect_client = get_client()
    assert prefect_client.server_type == ServerType.EPHEMERAL

async def test_server_type_server(hosted_api_server: Any) -> None:
    async with PrefectClient(hosted_api_server) as prefect_client:
        assert prefect_client.server_type == ServerType.SERVER

async def test_server_type_cloud() -> None:
    async with PrefectClient(PREFECT_CLOUD_API_URL.value()) as prefect_client:
        assert prefect_client.server_type == ServerType.CLOUD

@pytest.mark.parametrize("on_create, expected_value", [(True, True), (False, False), (None, False)])
async def test_update_deployment_does_not_overwrite_paused_when_not_provided(
    prefect_client: PrefectClient, flow_run: client_schemas.FlowRun, on_create: Optional[bool], expected_value: bool
) -> None:
    deployment_id: UUID = await prefect_client.create_deployment(
        flow_id=flow_run.flow_id, name="test-deployment", parameters={"foo": "bar"}, work_queue_name="wq", paused=on_create
    )
    deployment: DeploymentResponse = await prefect_client.read_deployment(deployment_id)
    assert deployment.paused == expected_value
    await prefect_client.update_deployment(deployment_id, client_schemas.actions.DeploymentUpdate(tags=["new-tag"]))
    deployment = await prefect_client.read_deployment(deployment_id)
    assert deployment.paused == expected_value

@pytest.mark.parametrize("on_create, after_create, on_update, after_update", [(False, False, True, True), (True, True, False, False), (None, False, True, True)])
async def test_update_deployment_paused(
    prefect_client: PrefectClient, flow_run: client_schemas.FlowRun, on_create: Optional[bool], after_create: bool, on_update: Optional[bool], after_update: bool
) -> None:
    deployment_id: UUID = await prefect_client.create_deployment(
        flow_id=flow_run.flow_id, name="test-deployment", parameters={"foo": "bar"}, work_queue_name="wq", paused=on_create
    )
    deployment: DeploymentResponse = await prefect_client.read_deployment(deployment_id)
    assert deployment.paused == after_create
    await prefect_client.update_deployment(deployment_id, client_schemas.actions.DeploymentUpdate(paused=on_update))
    deployment = await prefect_client.read_deployment(deployment_id)
    assert deployment.paused == after_update


class TestWorkPools:
    async def test_read_work_pools(self, prefect_client: PrefectClient) -> None:
        pools = await prefect_client.read_work_pools()
        existing_name = set([p.name for p in pools])
        existing_ids = set([p.id for p in pools])
        work_pool_1 = await prefect_client.create_work_pool(work_pool=WorkPoolCreate(name="test-pool-1"))
        work_pool_2 = await prefect_client.create_work_pool(work_pool=WorkPoolCreate(name="test-pool-2"))
        pools = await prefect_client.read_work_pools()
        names_after_adding = set([p.name for p in pools])
        ids_after_adding = set([p.id for p in pools])
        assert names_after_adding.symmetric_difference(existing_name) == {work_pool_1.name, work_pool_2.name}
        assert ids_after_adding.symmetric_difference(existing_ids) == {work_pool_1.id, work_pool_2.id}

    async def test_create_work_pool_overwriting_existing_work_pool(self, prefect_client: PrefectClient, work_pool: WorkPool) -> None:
        await prefect_client.create_work_pool(
            work_pool=WorkPoolCreate(name=work_pool.name, type=work_pool.type, description="new description"), overwrite=True
        )
        updated_work_pool = await prefect_client.read_work_pool(work_pool.name)
        assert updated_work_pool.description == "new description"

    async def test_create_work_pool_with_attempt_to_overwrite_type(self, prefect_client: PrefectClient, work_pool: WorkPool) -> None:
        with pytest.warns(UserWarning, match="Overwriting work pool type is not supported"):
            await prefect_client.create_work_pool(
                work_pool=WorkPoolCreate(name=work_pool.name, type="kubernetes", description=work_pool.description), overwrite=True
            )
        updated_work_pool = await prefect_client.read_work_pool(work_pool.name)
        assert updated_work_pool.type == work_pool.type

    async def test_update_work_pool(self, prefect_client: PrefectClient) -> None:
        work_pool = await prefect_client.create_work_pool(work_pool=WorkPoolCreate(name="test-pool-1"))
        assert work_pool.description is None
        await prefect_client.update_work_pool(work_pool_name=work_pool.name, work_pool=WorkPoolUpdate(description="Foo description"))
        result = await prefect_client.read_work_pool(work_pool_name=work_pool.name)
        assert result.description == "Foo description"

    async def test_update_missing_work_pool(self, prefect_client: PrefectClient) -> None:
        with pytest.raises(prefect.exceptions.ObjectNotFound):
            await prefect_client.update_work_pool(work_pool_name="abcdefg", work_pool=WorkPoolUpdate())

    async def test_delete_work_pool(self, prefect_client: PrefectClient, work_pool: WorkPool) -> None:
        await prefect_client.delete_work_pool(work_pool.name)
        with pytest.raises(prefect.exceptions.ObjectNotFound):
            await prefect_client.read_work_pool(work_pool.id)


class TestArtifacts:
    @pytest.fixture
    async def artifacts(self, prefect_client: PrefectClient) -> List[Any]:
        artifact1 = await prefect_client.create_artifact(artifact=ArtifactCreate(key="voltaic", data=1, type="table", description="# This is a markdown description title"))
        artifact2 = await prefect_client.create_artifact(artifact=ArtifactCreate(key="voltaic", data=2, type="table", description="# This is a markdown description title"))
        artifact3 = await prefect_client.create_artifact(artifact=ArtifactCreate(key="lotus", data=3, type="markdown", description="# This is a markdown description title"))
        return [artifact1, artifact2, artifact3]

    async def test_create_then_read_artifact(self, prefect_client: PrefectClient, client: Any) -> None:
        artifact_schema = ArtifactCreate(key="voltaic", data=1, description="# This is a markdown description title", metadata_={"data": "opens many doors"})
        artifact = await prefect_client.create_artifact(artifact=artifact_schema)
        response = await client.get(f"/artifacts/{artifact.id}")
        assert response.status_code == 200
        assert response.json()["key"] == artifact.key
        assert response.json()["description"] == artifact.description

    async def test_read_artifacts(self, prefect_client: PrefectClient, artifacts: List[Any]) -> None:
        artifact_list = await prefect_client.read_artifacts()
        assert len(artifact_list) == 3
        keyed_data = {(r.key, r.data) for r in artifact_list}
        assert keyed_data == {("voltaic", 1), ("voltaic", 2), ("lotus", 3)}

    async def test_read_artifacts_with_latest_filter(self, prefect_client: PrefectClient, artifacts: List[Any]) -> None:
        artifact_list = await prefect_client.read_latest_artifacts()
        assert len(artifact_list) == 2
        keyed_data = {(r.key, r.data) for r in artifact_list}
        assert keyed_data == {("voltaic", 2), ("lotus", 3)}

    async def test_read_artifacts_with_key_filter(self, prefect_client: PrefectClient, artifacts: List[Any]) -> None:
        key_artifact_filter = ArtifactFilter(key=ArtifactFilterKey(any_=["voltaic"]))
        artifact_list = await prefect_client.read_artifacts(artifact_filter=key_artifact_filter)
        assert len(artifact_list) == 2
        keyed_data = {(r.key, r.data) for r in artifact_list}
        assert keyed_data == {("voltaic", 1), ("voltaic", 2)}

    async def test_delete_artifact_succeeds(self, prefect_client: PrefectClient, artifacts: List[Any]) -> None:
        await prefect_client.delete_artifact(artifacts[1].id)
        artifact_list = await prefect_client.read_artifacts()
        assert len(artifact_list) == 2
        keyed_data = {(r.key, r.data) for r in artifact_list}
        assert keyed_data == {("voltaic", 1), ("lotus", 3)}

    async def test_delete_nonexistent_artifact_raises(self, prefect_client: PrefectClient) -> None:
        with pytest.raises(prefect.exceptions.ObjectNotFound):
            await prefect_client.delete_artifact(uuid4())


class TestVariables:
    @pytest.fixture
    async def variable(self, client: Any) -> Variable:
        res = await client.post("/variables/", json=VariableCreate(name="my_variable", value="my-value", tags=["123", "456"]).model_dump(mode="json"))
        assert res.status_code == 201
        return parse_obj_as(Variable, res.json())

    @pytest.fixture
    async def variables(self, client: Any) -> List[Variable]:
        variables = [
            VariableCreate(name="my_variable1", value="my-value1", tags=["1"]),
            VariableCreate(name="my_variable2", value="my-value2", tags=["2"]),
            VariableCreate(name="my_variable3", value="my-value3", tags=["3"]),
        ]
        results = []
        for variable in variables:
            res = await client.post("/variables/", json=variable.model_dump(mode="json"))
            assert res.status_code == 201
            results.append(res.json())
        return parse_obj_as(List[Variable], results)

    @pytest.mark.parametrize(
        "value",
        [
            "string-value",
            '"string-value"',
            123,
            12.3,
            True,
            False,
            None,
            {"key": "value"},
            ["value1", "value2"],
            {"key": ["value1", "value2"]},
        ],
    )
    async def test_create_variable(self, prefect_client: PrefectClient, value: Union[str, int, float, bool, None, dict, list]) -> None:
        created_variable = await prefect_client.create_variable(variable=VariableCreate(name="my_variable", value=value))
        assert created_variable
        assert created_variable.name == "my_variable"
        assert created_variable.value == value
        res = await prefect_client.read_variable_by_name(created_variable.name)
        assert res.name == created_variable.name
        assert res.value == value

    async def test_read_variable_by_name(self, prefect_client: PrefectClient, variable: Variable) -> None:
        res = await prefect_client.read_variable_by_name(variable.name)
        assert res.name == variable.name
        assert res.value == variable.value
        assert res.tags == variable.tags

    async def test_read_variable_by_name_doesnt_exist(self, prefect_client: PrefectClient) -> None:
        res = await prefect_client.read_variable_by_name("doesnt_exist")
        assert res is None

    async def test_delete_variable_by_name(self, prefect_client: PrefectClient, variable: Variable) -> None:
        await prefect_client.delete_variable_by_name(variable.name)
        res = await prefect_client.read_variable_by_name(variable.name)
        assert not res

    async def test_delete_variable_by_name_doesnt_exist(self, prefect_client: PrefectClient) -> None:
        with pytest.raises(prefect.exceptions.ObjectNotFound):
            await prefect_client.delete_variable_by_name("doesnt_exist")

    async def test_read_variables(self, prefect_client: PrefectClient, variables: List[Variable]) -> None:
        res = await prefect_client.read_variables()
        assert len(res) == len(variables)
        assert {r.name for r in res} == {v.name for v in variables}

    async def test_read_variables_with_limit(self, prefect_client: PrefectClient, variables: List[Variable]) -> None:
        res = await prefect_client.read_variables(limit=1)
        assert len(res) == 1
        assert res[0].name == variables[0].name


class TestAutomations:
    @pytest.fixture
    def automation(self) -> AutomationCore:
        return AutomationCore(
            name="test-automation",
            trigger=EventTrigger(match={"flow_run_id": "123"}, posture=Posture.Reactive, threshold=1, within=0),
            actions=[],
        )

    async def test_create_automation(self, cloud_client: Any, automation: AutomationCore) -> None:
        with respx.mock(base_url=PREFECT_CLOUD_API_URL.value(), using="httpx") as router:
            created_automation = automation.model_dump(mode="json")
            created_automation["id"] = str(uuid4())
            create_route = router.post("/automations/").mock(return_value=httpx.Response(200, json=created_automation))
            automation_id = await cloud_client.create_automation(automation)
            assert create_route.called
            assert json.loads(create_route.calls[0].request.content) == automation.model_dump(mode="json")
            assert automation_id == UUID(created_automation["id"])

    async def test_read_automation(self, cloud_client: Any, automation: AutomationCore) -> None:
        with respx.mock(base_url=PREFECT_CLOUD_API_URL.value(), using="httpx") as router:
            created_automation = automation.model_dump(mode="json")
            created_automation["id"] = str(uuid4())
            created_automation_id = created_automation["id"]
            read_route = router.get(f"/automations/{created_automation_id}").mock(return_value=httpx.Response(200, json=created_automation))
            read_automation = await cloud_client.read_automation(created_automation_id)
            assert read_route.called
            assert read_automation.id == UUID(created_automation["id"])

    async def test_read_automation_not_found(self, cloud_client: Any, automation: AutomationCore) -> None:
        with respx.mock(base_url=PREFECT_CLOUD_API_URL.value(), using="httpx") as router:
            created_automation = automation.model_dump(mode="json")
            created_automation["id"] = str(uuid4())
            created_automation_id = created_automation["id"]
            read_route = router.get(f"/automations/{created_automation_id}").mock(return_value=httpx.Response(404))
            with pytest.raises(prefect.exceptions.PrefectHTTPStatusError, match="404"):
                await cloud_client.read_automation(created_automation_id)
            assert read_route.called

    async def test_read_automations_by_name(self, cloud_client: Any, automation: AutomationCore) -> None:
        with respx.mock(base_url=PREFECT_CLOUD_API_URL.value(), using="httpx") as router:
            created_automation = automation.model_dump(mode="json")
            created_automation["id"] = str(uuid4())
            read_route = router.post("/automations/filter").mock(return_value=httpx.Response(200, json=[created_automation]))
            read_automation = await cloud_client.read_automations_by_name(automation.name)
            assert read_route.called
            assert len(read_automation) == 1
            assert read_automation[0].id == UUID(created_automation["id"])
            assert read_automation[0].name == automation.name == created_automation["name"]

    @pytest.fixture
    def automation2(self) -> AutomationCore:
        return AutomationCore(
            name="test-automation",
            trigger=EventTrigger(match={"flow_run_id": "234"}, posture=Posture.Reactive, threshold=1, within=0),
            actions=[],
        )

    async def test_read_automations_by_name_multiple_same_name(self, cloud_client: Any, automation: AutomationCore, automation2: AutomationCore) -> None:
        with respx.mock(base_url=PREFECT_CLOUD_API_URL.value(), using="httpx") as router:
            created_automation = automation.model_dump(mode="json")
            created_automation["id"] = str(uuid4())
            created_automation2 = automation2.model_dump(mode="json")
            created_automation2["id"] = str(uuid4())
            read_route = router.post("/automations/filter").mock(return_value=httpx.Response(200, json=[created_automation, created_automation2]))
            read_automation = await cloud_client.read_automations_by_name(automation.name)
            assert read_route.called
            assert len(read_automation) == 2, "Expected two automations with the same name"
            assert all([automation.name == created_automation["name"] for automation in read_automation]), "Expected all automations to have the same name"

    async def test_read_automations_by_name_not_found(self, cloud_client: Any, automation: AutomationCore) -> None:
        with respx.mock(base_url=PREFECT_CLOUD_API_URL.value(), using="httpx") as router:
            created_automation = automation.model_dump(mode="json")
            created_automation["id"] = str(uuid4())
            created_automation["name"] = "nonexistent"
            read_route = router.post("/automations/filter").mock(return_value=httpx.Response(200, json=[]))
            nonexistent_automation = await cloud_client.read_automations_by_name(name="nonexistent")
            assert read_route.called
            assert nonexistent_automation == []

    async def test_delete_owned_automations(self, cloud_client: Any) -> None:
        with respx.mock(base_url=PREFECT_CLOUD_API_URL.value(), using="httpx") as router:
            resource_id: str = f"prefect.deployment.{uuid4()}"
            delete_route = router.delete(f"/automations/owned-by/{resource_id}").mock(return_value=httpx.Response(204))
            await cloud_client.delete_resource_owned_automations(resource_id)
            assert delete_route.called

async def test_server_error_does_not_raise_on_client() -> None:
    async def raise_error() -> None:
        raise ValueError("test")
    app = create_app(ephemeral=True)
    app.api_app.add_api_route("/raise_error", raise_error)
    async with PrefectClient(api=app) as client:
        with pytest.raises(prefect.exceptions.HTTPStatusError, match="500"):
            await client._client.get("/raise_error")

async def test_prefect_client_follow_redirects() -> None:
    app = create_app(ephemeral=True)
    httpx_settings: dict = {"follow_redirects": True}
    async with PrefectClient(api=app, httpx_settings=httpx_settings) as client:
        assert client._client.follow_redirects is True
    httpx_settings = {"follow_redirects": False}
    async with PrefectClient(api=app, httpx_settings=httpx_settings) as client:
        assert client._client.follow_redirects is False
    with temporary_settings({PREFECT_TESTING_UNIT_TEST_MODE: False}):
        async with PrefectClient(api=app) as client:
            assert client._client.follow_redirects is True
    async with PrefectClient(api=app) as client:
        assert client._client.follow_redirects is False

async def test_global_concurrency_limit_create(prefect_client: PrefectClient) -> None:
    for slot_decay_per_second in [1, 1.2]:
        global_concurrency_limit_name: str = f"global-create-test-{slot_decay_per_second}"
        response_uuid: UUID = await prefect_client.create_global_concurrency_limit(
            GlobalConcurrencyLimitCreate(name=global_concurrency_limit_name, limit=42, slot_decay_per_second=slot_decay_per_second)
        )
        concurrency_limit = await prefect_client.read_global_concurrency_limit_by_name(name=global_concurrency_limit_name)
        assert concurrency_limit.id == response_uuid
        assert concurrency_limit.slot_decay_per_second == slot_decay_per_second

async def test_global_concurrency_limit_delete(prefect_client: PrefectClient) -> None:
    await prefect_client.create_global_concurrency_limit(GlobalConcurrencyLimitCreate(name="global-delete-test", limit=42))
    assert len(await prefect_client.read_global_concurrency_limits()) == 1
    await prefect_client.delete_global_concurrency_limit_by_name(name="global-delete-test")
    assert len(await prefect_client.read_global_concurrency_limits()) == 0
    with pytest.raises(prefect.exceptions.ObjectNotFound):
        await prefect_client.delete_global_concurrency_limit_by_name(name="global-delete-test")

async def test_global_concurrency_limit_update_with_integer(prefect_client: PrefectClient) -> None:
    for index, slot_decay_per_second in enumerate([1, 1.2]):
        created_global_concurrency_limit_name: str = f"global-update-test-{slot_decay_per_second}"
        updated_global_concurrency_limit_name: str = f"global-create-test-{slot_decay_per_second}-new"
        await prefect_client.create_global_concurrency_limit(GlobalConcurrencyLimitCreate(name=created_global_concurrency_limit_name, limit=42, slot_decay_per_second=slot_decay_per_second))
        await prefect_client.update_global_concurrency_limit(
            name=created_global_concurrency_limit_name, concurrency_limit=GlobalConcurrencyLimitUpdate(limit=1, name=updated_global_concurrency_limit_name)
        )
        assert len(await prefect_client.read_global_concurrency_limits()) == index + 1
        assert (await prefect_client.read_global_concurrency_limit_by_name(name=updated_global_concurrency_limit_name)).limit == 1
        assert (await prefect_client.read_global_concurrency_limit_by_name(name=updated_global_concurrency_limit_name)).slot_decay_per_second == slot_decay_per_second
        with pytest.raises(prefect.exceptions.ObjectNotFound):
            await prefect_client.update_global_concurrency_limit(name=created_global_concurrency_limit_name, concurrency_limit=GlobalConcurrencyLimitUpdate(limit=1))

async def test_global_concurrency_limit_read_nonexistent_by_name(prefect_client: PrefectClient) -> None:
    with pytest.raises(prefect.exceptions.ObjectNotFound):
        await prefect_client.read_global_concurrency_limit_by_name(name="not-here")


class TestPrefectClientDeploymentSchedules:
    @pytest.fixture
    async def deployment(self, prefect_client: PrefectClient) -> DeploymentResponse:
        foo = flow(lambda: None, name="foo")
        flow_id: UUID = await prefect_client.create_flow(foo)
        schedule: IntervalSchedule = IntervalSchedule(interval=timedelta(days=1), anchor_date=pendulum.datetime(2020, 1, 1))
        deployment_id: UUID = await prefect_client.create_deployment(
            flow_id=flow_id, name="test-deployment", schedules=[DeploymentScheduleCreate(schedule=schedule)], parameters={"foo": "bar"}, work_queue_name="wq"
        )
        deployment = await prefect_client.read_deployment(deployment_id)
        return deployment

    async def test_create_deployment_schedule(self, prefect_client: PrefectClient, deployment: DeploymentResponse) -> None:
        deployment_id: str = str(deployment.id)
        cron_schedule: CronSchedule = CronSchedule(cron="* * * * *")
        schedules: List[Tuple[Union[CronSchedule, IntervalSchedule], bool]] = [(cron_schedule, True)]
        result = await prefect_client.create_deployment_schedules(deployment_id, schedules)
        assert len(result) == 1
        assert result[0].id
        assert result[0].schedule == cron_schedule
        assert result[0].active is True

    async def test_create_multiple_deployment_schedules_success(self, prefect_client: PrefectClient, deployment: DeploymentResponse) -> None:
        deployment_id: str = str(deployment.id)
        cron_schedule: CronSchedule = CronSchedule(cron="0 12 * * *")
        interval_schedule: IntervalSchedule = IntervalSchedule(interval=timedelta(hours=1))
        schedules: List[Tuple[Union[CronSchedule, IntervalSchedule], bool]] = [(cron_schedule, True), (interval_schedule, False)]
        result = await prefect_client.create_deployment_schedules(deployment_id, schedules)
        assert len(result) == 2
        assert result[0].schedule == cron_schedule
        assert result[0].active is True
        assert result[1].schedule == interval_schedule
        assert result[1].active is False

    async def test_read_deployment_schedules_success(self, prefect_client: PrefectClient, deployment: DeploymentResponse) -> None:
        result = await prefect_client.read_deployment_schedules(deployment.id)
        assert len(result) == 1
        assert result[0].schedule == IntervalSchedule(interval=timedelta(days=1), anchor_date=pendulum.datetime(2020, 1, 1))
        assert result[0].active is True

    async def test_update_deployment_schedule_only_active(self, deployment: DeploymentResponse, prefect_client: PrefectClient) -> None:
        result = await prefect_client.read_deployment_schedules(deployment.id)
        assert result[0].active is True
        await prefect_client.update_deployment_schedule(deployment.id, deployment.schedules[0].id, active=False)
        result = await prefect_client.read_deployment_schedules(deployment.id)
        assert len(result) == 1
        assert result[0].active is False

    async def test_update_deployment_schedule_only_schedule(self, deployment: DeploymentResponse, prefect_client: PrefectClient) -> None:
        result = await prefect_client.read_deployment_schedules(deployment.id)
        assert result[0].schedule == IntervalSchedule(interval=timedelta(days=1), anchor_date=pendulum.datetime(2020, 1, 1))
        await prefect_client.update_deployment_schedule(deployment.id, deployment.schedules[0].id, schedule=IntervalSchedule(interval=timedelta(minutes=15)))
        result = await prefect_client.read_deployment_schedules(deployment.id)
        assert len(result) == 1
        assert result[0].schedule.interval == timedelta(minutes=15)

    async def test_update_deployment_schedule_all_fields(self, deployment: DeploymentResponse, prefect_client: PrefectClient) -> None:
        """
        A regression test for #13243
        """
        result = await prefect_client.read_deployment_schedules(deployment.id)
        assert result[0].schedule == IntervalSchedule(interval=timedelta(days=1), anchor_date=pendulum.datetime(2020, 1, 1))
        assert result[0].active is True
        await prefect_client.update_deployment_schedule(deployment.id, deployment.schedules[0].id, schedule=IntervalSchedule(interval=timedelta(minutes=15)), active=False)
        result = await prefect_client.read_deployment_schedules(deployment.id)
        assert len(result) == 1
        assert result[0].schedule.interval == timedelta(minutes=15)
        assert result[0].active is False

    async def test_delete_deployment_schedule_success(self, deployment: DeploymentResponse, prefect_client: PrefectClient) -> None:
        await prefect_client.delete_deployment_schedule(deployment.id, deployment.schedules[0].id)
        result = await prefect_client.read_deployment_schedules(deployment.id)
        assert len(result) == 0

    async def test_create_deployment_schedules_with_invalid_schedule(self, prefect_client: PrefectClient, deployment: DeploymentResponse) -> None:
        deployment_id: str = str(deployment.id)
        invalid_schedule: Any = "not a valid schedule"
        schedules = [(invalid_schedule, True)]
        with pytest.raises(pydantic.ValidationError):
            await prefect_client.create_deployment_schedules(deployment_id, schedules)

    async def test_read_deployment_schedule_nonexistent(self, prefect_client: PrefectClient) -> None:
        nonexistent_deployment_id: str = str(uuid4())
        with pytest.raises(prefect.exceptions.ObjectNotFound):
            await prefect_client.read_deployment_schedules(nonexistent_deployment_id)

    async def test_update_deployment_schedule_nonexistent(self, prefect_client: PrefectClient, deployment: DeploymentResponse) -> None:
        nonexistent_schedule_id: str = str(uuid4())
        with pytest.raises(prefect.exceptions.ObjectNotFound):
            await prefect_client.update_deployment_schedule(deployment.id, nonexistent_schedule_id, active=False)

    async def test_delete_deployment_schedule_nonexistent(self, prefect_client: PrefectClient, deployment: DeploymentResponse) -> None:
        nonexistent_schedule_id: str = str(uuid4())
        with pytest.raises(prefect.exceptions.ObjectNotFound):
            await prefect_client.delete_deployment_schedule(deployment.id, nonexistent_schedule_id)


class TestPrefectClientCsrfSupport:
    def test_enabled_ephemeral(self, enable_ephemeral_server: Any) -> None:
        prefect_client = get_client()
        assert prefect_client.server_type == ServerType.EPHEMERAL
        assert prefect_client._client.enable_csrf_support

    async def test_enabled_server_type(self, hosted_api_server: Any) -> None:
        async with PrefectClient(hosted_api_server) as prefect_client:
            assert prefect_client.server_type == ServerType.SERVER
            assert prefect_client._client.enable_csrf_support

    async def test_not_enabled_server_type_cloud(self) -> None:
        async with PrefectClient(PREFECT_CLOUD_API_URL.value()) as prefect_client:
            assert prefect_client.server_type == ServerType.CLOUD
            assert not prefect_client._client.enable_csrf_support

    async def test_disabled_setting_disabled(self, hosted_api_server: Any) -> None:
        with temporary_settings({PREFECT_CLIENT_CSRF_SUPPORT_ENABLED: False}):
            async with PrefectClient(hosted_api_server) as prefect_client:
                assert prefect_client.server_type == ServerType.SERVER
                assert not prefect_client._client.enable_csrf_support


class TestPrefectClientRaiseForAPIVersionMismatch:
    async def test_raise_for_api_version_mismatch(self, prefect_client: PrefectClient) -> None:
        await prefect_client.raise_for_api_version_mismatch()

    async def test_raise_for_api_version_mismatch_when_api_unreachable(self, prefect_client: PrefectClient, monkeypatch: Any) -> None:
        async def something_went_wrong(*args: Any, **kwargs: Any) -> Any:
            raise httpx.ConnectError
        monkeypatch.setattr(prefect_client, "api_version", something_went_wrong)
        with pytest.raises(RuntimeError) as e:
            await prefect_client.raise_for_api_version_mismatch()
        assert "Failed to reach API" in str(e.value)

    async def test_raise_for_api_version_mismatch_against_cloud(self, prefect_client: PrefectClient, monkeypatch: Any) -> None:
        monkeypatch.setattr(prefect_client, "server_type", ServerType.CLOUD)
        api_version_mock = AsyncMock()
        monkeypatch.setattr(prefect_client, "api_version", api_version_mock)
        await prefect_client.raise_for_api_version_mismatch()
        api_version_mock.assert_not_called()

    @pytest.mark.parametrize("client_version, api_version", [("3.0.0", "2.0.0"), ("2.0.0", "3.0.0")])
    async def test_raise_for_api_version_mismatch_with_incompatible_versions(
        self, prefect_client: PrefectClient, monkeypatch: Any, client_version: str, api_version: str
    ) -> None:
        monkeypatch.setattr(prefect_client, "api_version", AsyncMock(return_value=api_version))
        monkeypatch.setattr(prefect_client, "client_version", Mock(return_value=client_version))
        with pytest.raises(RuntimeError) as e:
            await prefect_client.raise_for_api_version_mismatch()
        assert f"Found incompatible versions: client: {client_version}, server: {api_version}. " in str(e.value)


class TestSyncClient:
    def test_get_sync_client(self) -> None:
        client = get_client(sync_client=True)
        assert isinstance(client, SyncPrefectClient)

    def test_fixture_is_sync(self, sync_prefect_client: SyncPrefectClient) -> None:
        assert isinstance(sync_prefect_client, SyncPrefectClient)

    def test_hello(self, sync_prefect_client: SyncPrefectClient) -> None:
        response = sync_prefect_client.hello()
        assert response.json() == "👋"

    def test_api_version(self, sync_prefect_client: SyncPrefectClient) -> None:
        version = sync_prefect_client.api_version()
        assert prefect.__version__
        assert version == prefect.__version__


class TestSyncClientRaiseForAPIVersionMismatch:
    def test_raise_for_api_version_mismatch(self, sync_prefect_client: SyncPrefectClient) -> None:
        sync_prefect_client.raise_for_api_version_mismatch()

    def test_raise_for_api_version_mismatch_when_api_unreachable(self, sync_prefect_client: SyncPrefectClient, monkeypatch: Any) -> None:
        def something_went_wrong(*args: Any, **kwargs: Any) -> Any:
            raise httpx.ConnectError
        monkeypatch.setattr(sync_prefect_client, "api_version", something_went_wrong)
        with pytest.raises(RuntimeError) as e:
            sync_prefect_client.raise_for_api_version_mismatch()
        assert "Failed to reach API" in str(e.value)

    def test_raise_for_api_version_mismatch_against_cloud(self, sync_prefect_client: SyncPrefectClient, monkeypatch: Any) -> None:
        monkeypatch.setattr(sync_prefect_client, "server_type", ServerType.CLOUD)
        api_version_mock = Mock()
        monkeypatch.setattr(sync_prefect_client, "api_version", api_version_mock)
        sync_prefect_client.raise_for_api_version_mismatch()
        api_version_mock.assert_not_called()

    @pytest.mark.parametrize("client_version, api_version", [("3.0.0", "2.0.0"), ("2.0.0", "3.0.0")])
    def test_raise_for_api_version_mismatch_with_incompatible_versions(
        self, sync_prefect_client: SyncPrefectClient, monkeypatch: Any, client_version: str, api_version: str
    ) -> None:
        monkeypatch.setattr(sync_prefect_client, "api_version", Mock(return_value=api_version))
        monkeypatch.setattr(sync_prefect_client, "client_version", Mock(return_value=client_version))
        with pytest.raises(RuntimeError) as e:
            sync_prefect_client.raise_for_api_version_mismatch()
        assert f"Found incompatible versions: client: {client_version}, server: {api_version}. " in str(e.value)


class TestPrefectClientWorkerHeartbeat:
    async def test_worker_heartbeat(self, prefect_client: PrefectClient, work_pool: Any) -> None:
        work_pool_name: str = str(work_pool.name)
        await prefect_client.send_worker_heartbeat(work_pool_name=work_pool_name, worker_name="test-worker", heartbeat_interval_seconds=10)
        workers = await prefect_client.read_workers_for_work_pool(work_pool_name)
        assert len(workers) == 1
        assert workers[0].name == "test-worker"
        assert workers[0].heartbeat_interval_seconds == 10

    async def test_worker_heartbeat_sends_metadata_if_passed(self, prefect_client: PrefectClient) -> None:
        with mock.patch("prefect.client.orchestration.base.BaseAsyncClient.request", return_value=httpx.Response(status_code=204)) as mock_post:
            await prefect_client.send_worker_heartbeat(
                work_pool_name="work-pool", worker_name="test-worker", heartbeat_interval_seconds=10, worker_metadata=WorkerMetadata(integrations=[Integration(name="prefect-aws", version="1.0.0")])
            )
            assert mock_post.call_args[1]["json"] == {
                "name": "test-worker",
                "heartbeat_interval_seconds": 10,
                "metadata": {"integrations": [{"name": "prefect-aws", "version": "1.0.0"}]},
            }

    async def test_worker_heartbeat_does_not_send_metadata_if_not_passed(self, prefect_client: PrefectClient) -> None:
        with mock.patch("prefect.client.orchestration.base.BaseAsyncClient.request", return_value=httpx.Response(status_code=204)) as mock_post:
            await prefect_client.send_worker_heartbeat(work_pool_name="work-pool", worker_name="test-worker", heartbeat_interval_seconds=10)
            assert mock_post.call_args[1]["json"] == {"name": "test-worker", "heartbeat_interval_seconds": 10}