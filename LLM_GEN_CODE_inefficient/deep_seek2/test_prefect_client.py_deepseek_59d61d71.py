import json
import os
import ssl
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Generator, List, Optional, Set, Tuple, Union
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
from prefect.client.orchestration import (
    PrefectClient,
    ServerType,
    SyncPrefectClient,
    get_client,
)
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
from prefect.client.schemas.objects import (
    Flow,
    FlowRunNotificationPolicy,
    FlowRunPolicy,
    Integration,
    StateType,
    TaskRun,
    Variable,
    WorkerMetadata,
    WorkQueue,
)
from prefect.client.schemas.responses import (
    DeploymentResponse,
    OrchestrationResult,
    SetStateStatus,
)
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

    def test_get_client_starts_subprocess_server_when_enabled(
        self, enable_ephemeral_server: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        subprocess_server_mock = MagicMock()

        monkeypatch.setattr(
            prefect.server.api.server, "SubprocessASGIServer", subprocess_server_mock
        )

        get_client()
        assert subprocess_server_mock.call_count == 1
        assert subprocess_server_mock.return_value.start.call_count == 1

    def test_get_client_raises_error_when_no_api_url_and_no_ephemeral_mode(
        self, disable_hosted_api_server: bool
    ) -> None:
        with pytest.raises(ValueError, match="API URL"):
            get_client()


class TestClientProxyAwareness:
    @pytest.fixture()
    def remote_https_api(self) -> Generator[httpx.URL, None, None]:
        api_url = "https://127.0.0.1:4242/"
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
        httpx_settings = {"transport": httpx.AsyncHTTPTransport(retries=11)}
        httpx_client = get_client(httpx_settings)._client
        assert isinstance(httpx_client, httpx.AsyncClient)

        transport_for_api = httpx_client._transport_for_url(remote_https_api)
        assert isinstance(transport_for_api, httpx.AsyncHTTPTransport)

        pool = transport_for_api._pool
        assert isinstance(pool, httpcore.AsyncConnectionPool)
        assert pool._retries == 11

    @pytest.fixture
    def https_proxy(self) -> Generator[httpcore.URL, None, None]:
        original = os.environ.get("HTTPS_PROXY")
        try:
            os.environ["HTTPS_PROXY"] = "https://127.0.0.1:6666"
            yield httpcore.URL(os.environ["HTTPS_PROXY"])
        finally:
            if original is None:
                del os.environ["HTTPS_PROXY"]
            else:
                os.environ["HTTPS_PROXY"] = original

    async def test_client_is_aware_of_https_proxy(
        self, remote_https_api: httpx.URL, https_proxy: httpcore.URL
    ) -> None:
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
        api_url = "http://127.0.0.1:4242/"
        with temporary_settings(updates={PREFECT_API_URL: api_url}):
            yield httpx.URL(api_url)

    @pytest.fixture
    def http_proxy(self) -> Generator[httpcore.URL, None, None]:
        original = os.environ.get("HTTP_PROXY")
        try:
            os.environ["HTTP_PROXY"] = "http://127.0.0.1:6666"
            yield httpcore.URL(os.environ["HTTP_PROXY"])
        finally:
            if original is None:
                del os.environ["HTTP_PROXY"]
            else:
                os.environ["HTTP_PROXY"] = original

    async def test_client_is_aware_of_http_proxy(
        self, remote_http_api: httpx.URL, http_proxy: httpcore.URL
    ) -> None:
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
        assert client._started, "Client should be started during function"
        assert not client._closed, "Client should be closed during function"
        await client.api_healthcheck()
        return client

    async def test_get_new_client(self) -> None:
        client = await TestInjectClient.injected_func()
        assert isinstance(client, PrefectClient)
        assert client._closed, "Client should be closed after function returns"

    async def test_get_new_client_with_explicit_none(self) -> None:
        client = await TestInjectClient.injected_func(client=None)
        assert isinstance(client, PrefectClient)
        assert client._closed, "Client should be closed after function returns"

    async def test_use_existing_client(self, prefect_client: PrefectClient) -> None:
        client = await TestInjectClient.injected_func(client=prefect_client)
        assert client is prefect_client, "Client should be the same object"
        assert not client._closed, "Client should not be closed after function returns"

    async def test_use_existing_client_from_flow_run_ctx(self, prefect_client: PrefectClient) -> None:
        with prefect.context.FlowRunContext.model_construct(client=prefect_client):
            client = await TestInjectClient.injected_func()
        assert client is prefect_client, "Client should be the same object"
        assert not client._closed, "Client should not be closed after function returns"

    async def test_use_existing_client_from_task_run_ctx(self, prefect_client: PrefectClient) -> None:
        with prefect.context.FlowRunContext.model_construct(client=prefect_client):
            client = await TestInjectClient.injected_func()
        assert client is prefect_client, "Client should be the same object"
        assert not client._closed, "Client should not be closed after function returns"

    async def test_use_existing_client_from_flow_run_ctx_with_null_kwarg(
        self, prefect_client: PrefectClient
    ) -> None:
        with prefect.context.FlowRunContext.model_construct(client=prefect_client):
            client = await TestInjectClient.injected_func(client=None)
        assert client is prefect_client, "Client should be the same object"
        assert not client._closed, "Client should not be closed after function returns"


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
        startup, shutdown = MagicMock(), MagicMock()
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
        startup, shutdown = MagicMock(), MagicMock()
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
        startup, shutdown = MagicMock(), MagicMock()
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

        one_started = anyio.Event()
        one_exited = anyio.Event()
        two_started = anyio.Event()

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
        startup = MagicMock(side_effect=lambda: print("Startup called!"))
        shutdown = MagicMock(side_effect=lambda: print("Shutdown called!!"))

        app = FastAPI(lifespan=make_lifespan(startup, shutdown))

        one_started = anyio.Event()
        one_exited = anyio.Event()
        two_started = anyio.Event()

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
        startup, shutdown = MagicMock(), MagicMock()
        app = FastAPI(lifespan=make_lifespan(startup, shutdown))

        client = PrefectClient(app)

        with pytest.raises(ValueError):
            async with client:
                raise ValueError()

        startup.assert_called_once()
        shutdown.assert_called_once()

    async def test_client_context_manages_app_lifespan_on_anyio_cancellation(self) -> None:
        startup, shutdown = MagicMock(), MagicMock()
        app = FastAPI(lifespan=make_lifespan(startup, shutdown))

        async def enter_client(task_status: anyio.abc.TaskStatus) -> None:
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
        startup, shutdown = MagicMock(), MagicMock()
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
async def test_client_runs_migrations_for_ephemeral_app_only_once(
    enabled: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    with temporary_settings(updates={PREFECT_API_DATABASE_MIGRATE_ON_START: enabled}):
        monkeypatch.setattr(prefect.server.api.server, "LIFESPAN_RAN_FOR_APP", set())

        app = create_app(ephemeral=True, ignore_cache=True)
        mock = AsyncMock()
        monkeypatch.setattr(
            "prefect.server.database.interface.PrefectDBInterface.create_db", mock
        )
        async with PrefectClient(app):
            if enabled:
                mock.assert_awaited_once_with()

        async with PrefectClient(app):
            if enabled:
                mock.assert_awaited_once_with()

        if not enabled:
            mock.assert_not_awaited()


@pytest.mark.parametrize("enabled", [True, False])
async def test_client_runs_migrations_for_two_different_ephemeral_apps(
    enabled: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    with temporary_settings(updates={PREFECT_API_DATABASE_MIGRATE_ON_START: enabled}):
        monkeypatch.setattr(prefect.server.api.server, "LIFESPAN_RAN_FOR_APP", set())

        app = create_app(ephemeral=True, ignore_cache=True)
        app2 = create_app(ephemeral=True, ignore_cache=True)

        mock = AsyncMock()
        monkeypatch.setattr(
            "prefect.server.database.interface.PrefectDBInterface.create_db", mock
        )
        async with PrefectClient(app):
            if enabled:
                mock.assert_awaited_once_with()

        async with PrefectClient(app2):
            if enabled:
                assert mock.await_count == 2

        if not enabled:
            mock.assert_not_awaited()


async def test_client_does_not_run_migrations_for_hosted_app(
    hosted_api_server: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    with temporary_settings(updates={PREFECT_API_DATABASE_MIGRATE_ON_START: True}):
        mock = AsyncMock()
        monkeypatch.setattr(
            "prefect.server.database.interface.PrefectDBInterface.create_db", mock
        )
        async with PrefectClient(hosted_api_server):
            pass

    mock.assert_not_awaited()


async def test_client_api_url() -> None:
    url = PrefectClient("http://foo.test/bar").api_url
    assert isinstance(url, httpx.URL)
    assert str(url) == "http://foo.test/bar/"
    assert PrefectClient(FastAPI()).api_url is not None


async def test_hello(prefect_client: PrefectClient) -> None:
    response = await prefect_client.hello()
    assert response.json() == "👋"


async def test_healthcheck(prefect_client: PrefectClient) -> None:
    assert await prefect_client.api_healthcheck() is None


async def test_healthcheck_failure(
    prefect_client: PrefectClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        prefect_client._client, "get", AsyncMock(side_effect=ValueError("test"))
    assert exceptions_equal(await prefect_client.api_healthcheck(), ValueError("test"))


async def test_create_then_read_flow(prefect_client: PrefectClient) -> None:
    @flow
    def foo() -> None:
        pass

    flow_id = await prefect_client.create_flow(foo)
    assert isinstance(flow_id, UUID)

    lookup = await prefect_client.read_flow(flow_id)
    assert isinstance(lookup, Flow)
    assert lookup.name == foo.name


async def test_create_then_delete_flow(prefect_client: PrefectClient) -> None:
    @flow
    def foo() -> None:
        pass

    flow_id = await prefect_client.create_flow(foo)
    assert isinstance(flow_id, UUID)

    await prefect_client.delete_flow(flow_id)
    with pytest.raises(prefect.exceptions.PrefectHTTPStatusError, match="404"):
        await prefect_client.read_flow(flow_id)


async def test_create_then_read_deployment(
    prefect_client: PrefectClient, storage_document_id: UUID
) -> None:
    @flow
    def foo() -> None:
        pass

    flow_id = await prefect_client.create_flow(foo)
    schedule = DeploymentScheduleCreate(
        schedule=IntervalSchedule(interval=timedelta(days=1))
    )

    deployment_id = await prefect_client.create_deployment(
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

    lookup = await prefect_client.read_deployment(deployment_id)
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


async def test_update_deployment(
    prefect_client: PrefectClient, storage_document_id: UUID
) -> None:
    @flow
    def foo() -> None:
        pass

    flow_id = await prefect_client.create_flow(foo)

    deployment_id = await prefect_client.create_deployment(
        flow_id=flow_id,
        name="test-deployment",
        version="git-commit-hash",
        parameters={"foo": "bar"},
        tags=["foo", "bar"],
        paused=True,
        storage_document_id=storage_document_id,
        parameter_openapi_schema={},
    )

    deployment = await prefect_client.read_deployment(deployment_id)

    await prefect_client.update_deployment(
        deployment_id=deployment_id,
        deployment=client_schemas.actions.DeploymentUpdate(
            tags=["new", "tags"], concurrency_limit=42
        ),
    )

    updated_deployment = await prefect_client.read_deployment(deployment_id)
    assert updated_deployment.tags == ["new", "tags"]
    assert updated_deployment.global_concurrency_limit.limit == 42
    assert updated_deployment.id == deployment.id
    assert updated_deployment.name == deployment.name
    assert updated_deployment.version == deployment.version
    assert updated_deployment.parameters == deployment.parameters
    assert updated_deployment.paused == deployment.paused
    assert updated_deployment.storage_document_id == deployment.storage_document_id
    assert updated_deployment.parameter_openapi_schema == deployment.parameter_openapi_schema


async def test_update_deployment_to_remove_schedules(
    prefect_client: PrefectClient, storage_document_id: UUID
) -> None:
    @flow
    def foo() -> None:
        pass

    flow_id = await prefect_client.create_flow(foo)
    schedule = DeploymentScheduleCreate(
        schedule=IntervalSchedule(interval=timedelta(days=1))
    )

    deployment_id = await prefect_client.create_deployment(
        flow_id=flow_id,
        name="test-deployment",
        version="git-commit-hash",
        schedules=[schedule],
        parameters={"foo": "bar"},
        tags=["foo", "bar"],
        storage_document_id=storage_document_id,
        parameter_openapi_schema={},
    )

    deployment = await prefect_client.read_deployment(deployment_id)
    assert len(deployment.schedules) == 1

    await prefect_client.update_deployment(
        deployment_id=deployment_id,
        deployment=client_schemas.actions.DeploymentUpdate(schedules=[]),
    )

    updated_deployment = await prefect_client.read_deployment(deployment_id)
    assert len(updated_deployment.schedules) == 0


async def test_read_deployment_by_name(prefect_client: PrefectClient) -> None:
    @flow
    def foo() -> None:
        pass

    flow_id = await prefect_client.create_flow(foo)

    deployment_id = await prefect_client.create_deployment(
        flow_id=flow_id,
        name="test-deployment",
    )

    lookup = await prefect_client.read_deployment_by_name("foo/test-deployment")
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
    prefect_client: PrefectClient,
    deployment_tags: List[str],
    filter_tags: List[str],
    expected_match: bool,
) -> None:
    @flow
    def moo_deng() -> None:
        pass

    flow_id = await prefect_client.create_flow(moo_deng)

    await prefect_client.create_deployment(
        flow_id=flow_id,
        name="moisturized-deployment",
        tags=deployment_tags,
    )
    deployment_responses = await prefect_client.read_deployments(
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

    flow_id = await prefect_client.create_flow(foo)

    deployment_id = await prefect_client.create_deployment(
        flow_id=flow_id,
        name="test-deployment",
    )

    await prefect_client.delete_deployment(deployment_id)
    with pytest.raises(prefect.exceptions.ObjectNotFound):
        await prefect_client.read_deployment(deployment_id)


async def test_read_nonexistent_deployment_by_name(prefect_client: PrefectClient) -> None:
    with pytest.raises((prefect.exceptions.ObjectNotFound, ValueError)):
        await prefect_client.read_deployment_by_name("not-a-real-deployment")


async def test_create_then_read_concurrency_limit(prefect_client: PrefectClient) -> None:
    cl_id = await prefect_client.create_concurrency_limit(
        tag="client-created", concurrency_limit=12345
    )

    lookup = await prefect_client.read_concurrency_limit_by_tag("client-created")
    assert lookup.id == cl_id
    assert lookup.concurrency_limit == 12345


async def test_read_nonexistent_concurrency_limit_by_tag(prefect_client: PrefectClient) -> None:
    with pytest.raises(prefect.exceptions.ObjectNotFound):
        await prefect_client.read_concurrency_limit_by_tag("not-a-real-tag")


async def test_resetting_concurrency_limits(prefect_client: PrefectClient) -> None:
    await prefect_client.create_concurrency_limit(
        tag="an-unimportant-limit", concurrency_limit=100
    )

    await prefect_client.reset_concurrency_limit_by_tag(
        "an-unimportant-limit", slot_override=[uuid4(), uuid4(), uuid4()]
    )
    first_lookup = await prefect_client.read_concurrency_limit_by_tag(
        "an-unimportant-limit"
    )
    assert len(first_lookup.active_slots) == 3

    await prefect_client.reset_concurrency_limit_by_tag("an-unimportant-limit")
    reset_lookup = await prefect_client.read_concurrency_limit_by_tag(
        "an-unimportant-limit"
    )
    assert len(reset_lookup.active_slots) == 0


async def test_deleting_concurrency_limits(prefect_client: PrefectClient) -> None:
    await prefect_client.create_concurrency_limit(
        tag="dead-limit-walking", concurrency_limit=10
    )

    assert await prefect_client.read_concurrency_limit_by_tag("dead-limit-walking")
    await prefect_client.delete_concurrency_limit_by_tag("dead-limit-walking")
    with pytest.raises(prefect.exceptions.ObjectNotFound):
        await prefect_client.read_concurrency_limit_by_tag("dead-limit-walking")


async def test_create_then_read_flow_run(prefect_client: PrefectClient) -> None:
    @flow
    def foo() -> None:
        pass

    flow_run = await prefect_client.create_flow_run(
        foo,
        name="zachs-flow-run",
    )
    assert isinstance(flow_run, client_schemas.FlowRun)

    lookup = await prefect_client.read_flow_run(flow_run.id)
    lookup.estimated_start_time_delta = flow_run.estimated_start_time_delta
    lookup.estimated_run_time = flow_run.estimated_run_time
    assert lookup == flow_run


async def test_create_flow_run_retains_parameters(prefect_client: PrefectClient) -> None:
    @flow
    def foo() -> None:
        pass

    parameters = {"x": 1, "y": [1, 2, 3]}

    flow_run = await prefect_client.create_flow_run(
        foo, name="zachs-flow-run", parameters=parameters
    )
    assert parameters == flow_run.parameters
    assert id(flow_run.parameters) == id(parameters)


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

    flow_run_id = (await prefect_client.create_flow_run(foo)).id
    response = await prefect_client.set_flow_run_state(
        flow_run_id,
        state=Completed(message="Test!"),
    )
    assert isinstance(response, OrchestrationResult)
    assert response.status == SetStateStatus.ACCEPT

    states = await prefect_client.read_flow_run_states(flow_run_id)
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
        await prefect_client.set_flow_run_state(
            uuid4(),
            state=Completed(message="Test!"),
        )


async def test_read_flow_runs_without_filter(prefect_client: PrefectClient) -> None:
    @flow
    def foo() -> None:
        pass

    fr_id_1 = (await prefect_client.create_flow_run(foo)).id
    fr_id_2 = (await prefect_client.create_flow_run(foo)).id

    flow_runs = await prefect_client.read_flow_runs()
    assert len(flow_runs) == 2
    assert all(isinstance(flow_run, client_schemas.FlowRun) for flow_run in flow_runs)
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
    (await prefect