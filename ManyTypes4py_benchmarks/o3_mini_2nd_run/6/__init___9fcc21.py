from __future__ import annotations
import asyncio
import base64
import datetime
import ssl
from collections.abc import Iterable
from contextlib import AsyncExitStack
from logging import Logger
from typing import Any, List, Optional, Union, overload
from uuid import UUID

import certifi
import httpcore
import httpx
import pydantic
from asgi_lifespan import LifespanManager
from packaging import version
from starlette import status
from typing_extensions import ParamSpec, Self, TypeVar

from prefect.client.orchestration._artifacts.client import (
    ArtifactAsyncClient,
    ArtifactClient,
    ArtifactCollectionAsyncClient,
    ArtifactCollectionClient,
)
from prefect.client.orchestration._concurrency_limits.client import ConcurrencyLimitAsyncClient, ConcurrencyLimitClient
from prefect.client.orchestration._logs.client import LogAsyncClient, LogClient
from prefect.client.orchestration._variables.client import VariableAsyncClient, VariableClient
from prefect.client.orchestration._deployments.client import DeploymentAsyncClient, DeploymentClient
from prefect.client.orchestration._automations.client import AutomationAsyncClient, AutomationClient
from prefect.client.orchestration._work_pools.client import WorkPoolAsyncClient, WorkPoolClient
from prefect._experimental.sla.client import SlaAsyncClient, SlaClient
from prefect.client.orchestration._flows.client import FlowAsyncClient, FlowClient
from prefect.client.orchestration._flow_runs.client import FlowRunAsyncClient, FlowRunClient
from prefect.client.orchestration._blocks_documents.client import BlocksDocumentAsyncClient, BlocksDocumentClient
from prefect.client.orchestration._blocks_schemas.client import BlocksSchemaAsyncClient, BlocksSchemaClient
from prefect.client.orchestration._blocks_types.client import BlocksTypeAsyncClient, BlocksTypeClient
import prefect
import prefect.exceptions
import prefect.settings
import prefect.states
from prefect.client.constants import SERVER_API_VERSION
from prefect.client.schemas import FlowRun, OrchestrationResult, TaskRun
from prefect.client.schemas.actions import (
    FlowRunNotificationPolicyCreate,
    FlowRunNotificationPolicyUpdate,
    TaskRunCreate,
    TaskRunUpdate,
    WorkQueueCreate,
    WorkQueueUpdate,
)
from prefect.client.schemas.filters import (
    DeploymentFilter,
    FlowFilter,
    FlowRunFilter,
    FlowRunNotificationPolicyFilter,
    TaskRunFilter,
    WorkQueueFilter,
    WorkQueueFilterName,
)
from prefect.client.schemas.objects import Constant, FlowRunNotificationPolicy, Parameter, TaskRunPolicy, TaskRunResult, WorkQueue, WorkQueueStatusDetail
from prefect.client.schemas.sorting import TaskRunSort
from prefect.logging import get_logger
from prefect.settings import (
    PREFECT_API_AUTH_STRING,
    PREFECT_API_DATABASE_CONNECTION_URL,
    PREFECT_API_ENABLE_HTTP2,
    PREFECT_API_KEY,
    PREFECT_API_REQUEST_TIMEOUT,
    PREFECT_API_SSL_CERT_FILE,
    PREFECT_API_TLS_INSECURE_SKIP_VERIFY,
    PREFECT_API_URL,
    PREFECT_CLIENT_CSRF_SUPPORT_ENABLED,
    PREFECT_CLOUD_API_URL,
    PREFECT_SERVER_ALLOW_EPHEMERAL_MODE,
    PREFECT_TESTING_UNIT_TEST_MODE,
)
from prefect.types._datetime import now
if TYPE_CHECKING:
    from prefect.tasks import Task as TaskObject
from prefect.client.base import ASGIApp, PrefectHttpxAsyncClient, PrefectHttpxSyncClient, ServerType, app_lifespan_context

P = ParamSpec("P")
R = TypeVar("R", infer_variance=True)
T = TypeVar("T")


@overload
def get_client(*, httpx_settings: Optional[dict[str, Any]] = ..., sync_client: bool = False) -> PrefectClient:
    ...


@overload
def get_client(*, httpx_settings: Optional[dict[str, Any]] = ..., sync_client: bool = ...) -> SyncPrefectClient:
    ...


def get_client(httpx_settings: Optional[dict[str, Any]] = None, sync_client: bool = False) -> Union[SyncPrefectClient, PrefectClient]:
    """
    Retrieve a HTTP client for communicating with the Prefect REST API.
    """
    import prefect.context

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if sync_client:
        if (client_ctx := prefect.context.SyncClientContext.get()):
            if client_ctx.client and getattr(client_ctx, "_httpx_settings", None) == httpx_settings:
                return client_ctx.client
    elif (client_ctx := prefect.context.AsyncClientContext.get()):
        if client_ctx.client and getattr(client_ctx, "_httpx_settings", None) == httpx_settings and (loop in (getattr(client_ctx.client, "_loop", None), None)):
            return client_ctx.client
    api: str = PREFECT_API_URL.value()
    server_type: Optional[Any] = None
    if not api and PREFECT_SERVER_ALLOW_EPHEMERAL_MODE:
        from prefect.server.api.server import SubprocessASGIServer

        server = SubprocessASGIServer()
        server.start()
        assert server.server_process is not None, "Server process did not start"
        api = server.api_url
        server_type = ServerType.EPHEMERAL
    elif not api and (not PREFECT_SERVER_ALLOW_EPHEMERAL_MODE):
        raise ValueError("No Prefect API URL provided. Please set PREFECT_API_URL to the address of a running Prefect server.")
    if sync_client:
        return SyncPrefectClient(
            api,
            auth_string=PREFECT_API_AUTH_STRING.value(),
            api_key=PREFECT_API_KEY.value(),
            httpx_settings=httpx_settings,
            server_type=server_type,
        )
    else:
        return PrefectClient(
            api,
            auth_string=PREFECT_API_AUTH_STRING.value(),
            api_key=PREFECT_API_KEY.value(),
            httpx_settings=httpx_settings,
            server_type=server_type,
        )


class PrefectClient(
    ArtifactAsyncClient,
    ArtifactCollectionAsyncClient,
    LogAsyncClient,
    VariableAsyncClient,
    ConcurrencyLimitAsyncClient,
    DeploymentAsyncClient,
    AutomationAsyncClient,
    SlaAsyncClient,
    FlowRunAsyncClient,
    FlowAsyncClient,
    BlocksDocumentAsyncClient,
    BlocksSchemaAsyncClient,
    BlocksTypeAsyncClient,
    WorkPoolAsyncClient,
):
    """
    An asynchronous client for interacting with the Prefect REST API.
    """

    def __init__(
        self,
        api: Union[str, ASGIApp],
        *,
        auth_string: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
        httpx_settings: Optional[dict[str, Any]] = None,
        server_type: Optional[Any] = None,
    ) -> None:
        httpx_settings = httpx_settings.copy() if httpx_settings else {}
        httpx_settings.setdefault("headers", {})
        if PREFECT_API_TLS_INSECURE_SKIP_VERIFY:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            httpx_settings.setdefault("verify", ctx)
        else:
            cert_file: Optional[str] = PREFECT_API_SSL_CERT_FILE.value()
            if not cert_file:
                cert_file = certifi.where()
            ctx = ssl.create_default_context(cafile=cert_file)
            httpx_settings.setdefault("verify", ctx)
        if api_version is None:
            api_version = SERVER_API_VERSION
        httpx_settings["headers"].setdefault("X-PREFECT-API-VERSION", api_version)
        if api_key:
            httpx_settings["headers"].setdefault("Authorization", f"Bearer {api_key}")
        if auth_string:
            token = base64.b64encode(auth_string.encode("utf-8")).decode("utf-8")
            httpx_settings["headers"].setdefault("Authorization", f"Basic {token}")
        self._context_stack: int = 0
        self._exit_stack: AsyncExitStack = AsyncExitStack()
        self._ephemeral_app: Optional[ASGIApp] = None
        self.manage_lifespan: bool = True
        self._ephemeral_lifespan: Any = None
        self._closed: bool = False
        self._started: bool = False
        if isinstance(api, str):
            if httpx_settings.get("app"):
                raise ValueError(
                    "Invalid httpx settings: `app` cannot be set when providing an api url. "
                    "`app` is only for use with ephemeral instances. Provide it as the `api` parameter instead."
                )
            httpx_settings.setdefault("base_url", api)
            httpx_settings.setdefault("limits", httpx.Limits(max_connections=16, max_keepalive_connections=8, keepalive_expiry=25))
            httpx_settings.setdefault("http2", PREFECT_API_ENABLE_HTTP2.value())
            if server_type:
                self.server_type = server_type
            else:
                self.server_type = ServerType.CLOUD if api.startswith(PREFECT_CLOUD_API_URL.value()) else ServerType.SERVER
        else:
            self._ephemeral_app = api
            self.server_type = ServerType.EPHEMERAL
            httpx_settings.setdefault("transport", httpx.ASGITransport(app=self._ephemeral_app, raise_app_exceptions=False))
            httpx_settings.setdefault("base_url", "http://ephemeral-prefect/api")
        httpx_settings.setdefault(
            "timeout",
            httpx.Timeout(
                connect=PREFECT_API_REQUEST_TIMEOUT.value(),
                read=PREFECT_API_REQUEST_TIMEOUT.value(),
                write=PREFECT_API_REQUEST_TIMEOUT.value(),
                pool=PREFECT_API_REQUEST_TIMEOUT.value(),
            ),
        )
        if not PREFECT_TESTING_UNIT_TEST_MODE:
            httpx_settings.setdefault("follow_redirects", True)
        enable_csrf_support: bool = self.server_type != ServerType.CLOUD and PREFECT_CLIENT_CSRF_SUPPORT_ENABLED.value()
        self._client: PrefectHttpxAsyncClient = PrefectHttpxAsyncClient(**httpx_settings, enable_csrf_support=enable_csrf_support)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        if isinstance(api, str) and (not httpx_settings.get("transport")):
            transport_for_url = getattr(self._client, "_transport_for_url", None)
            if callable(transport_for_url):
                server_transport = transport_for_url(httpx.URL(api))
                if isinstance(server_transport, httpx.AsyncHTTPTransport):
                    pool = getattr(server_transport, "_pool", None)
                    if isinstance(pool, httpcore.AsyncConnectionPool):
                        setattr(pool, "_retries", 3)
        self.logger: Logger = get_logger("client")

    @property
    def api_url(self) -> Any:
        """
        Get the base URL for the API.
        """
        return self._client.base_url

    async def api_healthcheck(self) -> Optional[Exception]:
        """
        Attempts to connect to the API and returns the encountered exception if not successful.
        """
        try:
            await self._client.get("/health")
            return None
        except Exception as exc:
            return exc

    async def hello(self) -> httpx.Response:
        """
        Send a GET request to /hello for testing purposes.
        """
        return await self._client.get("/hello")

    async def create_work_queue(
        self,
        name: str,
        description: Optional[str] = None,
        is_paused: Optional[bool] = None,
        concurrency_limit: Optional[int] = None,
        priority: Optional[int] = None,
        work_pool_name: Optional[str] = None,
    ) -> WorkQueue:
        """
        Create a work queue.
        """
        create_model = WorkQueueCreate(name=name, filter=None)
        if description is not None:
            create_model.description = description
        if is_paused is not None:
            create_model.is_paused = is_paused
        if concurrency_limit is not None:
            create_model.concurrency_limit = concurrency_limit
        if priority is not None:
            create_model.priority = priority
        data: dict[str, Any] = create_model.model_dump(mode="json")
        try:
            if work_pool_name is not None:
                response = await self._client.post(f"/work_pools/{work_pool_name}/queues", json=data)
            else:
                response = await self._client.post("/work_queues/", json=data)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == status.HTTP_409_CONFLICT:
                raise prefect.exceptions.ObjectAlreadyExists(http_exc=e) from e
            elif e.response.status_code == status.HTTP_404_NOT_FOUND:
                raise prefect.exceptions.ObjectNotFound(http_exc=e) from e
            else:
                raise
        return WorkQueue.model_validate(response.json())

    async def read_work_queue_by_name(self, name: str, work_pool_name: Optional[str] = None) -> WorkQueue:
        """
        Read a work queue by name.
        """
        try:
            if work_pool_name is not None:
                response = await self._client.get(f"/work_pools/{work_pool_name}/queues/{name}")
            else:
                response = await self._client.get(f"/work_queues/name/{name}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == status.HTTP_404_NOT_FOUND:
                raise prefect.exceptions.ObjectNotFound(http_exc=e) from e
            else:
                raise
        return WorkQueue.model_validate(response.json())

    async def update_work_queue(self, id: Union[str, int], **kwargs: Any) -> None:
        """
        Update properties of a work queue.
        """
        if not kwargs:
            raise ValueError("No fields provided to update.")
        data = WorkQueueUpdate(**kwargs).model_dump(mode="json", exclude_unset=True)
        try:
            await self._client.patch(f"/work_queues/{id}", json=data)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == status.HTTP_404_NOT_FOUND:
                raise prefect.exceptions.ObjectNotFound(http_exc=e) from e
            else:
                raise

    async def get_runs_in_work_queue(
        self, id: Union[str, int], limit: int = 10, scheduled_before: Optional[datetime.datetime] = None
    ) -> List[FlowRun]:
        """
        Read flow runs off a work queue.
        """
        if scheduled_before is None:
            scheduled_before = now("UTC")
        try:
            response = await self._client.post(
                f"/work_queues/{id}/get_runs",
                json={"limit": limit, "scheduled_before": scheduled_before.isoformat()},
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == status.HTTP_404_NOT_FOUND:
                raise prefect.exceptions.ObjectNotFound(http_exc=e) from e
            else:
                raise
        return pydantic.TypeAdapter(list[FlowRun]).validate_python(response.json())

    async def read_work_queue(self, id: Union[str, int]) -> WorkQueue:
        """
        Read a work queue.
        """
        try:
            response = await self._client.get(f"/work_queues/{id}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == status.HTTP_404_NOT_FOUND:
                raise prefect.exceptions.ObjectNotFound(http_exc=e) from e
            else:
                raise
        return WorkQueue.model_validate(response.json())

    async def read_work_queue_status(self, id: Union[str, int]) -> WorkQueueStatusDetail:
        """
        Read a work queue status.
        """
        try:
            response = await self._client.get(f"/work_queues/{id}/status")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == status.HTTP_404_NOT_FOUND:
                raise prefect.exceptions.ObjectNotFound(http_exc=e) from e
            else:
                raise
        return WorkQueueStatusDetail.model_validate(response.json())

    async def match_work_queues(self, prefixes: List[str], work_pool_name: Optional[str] = None) -> List[WorkQueue]:
        """
        Query the Prefect API for work queues with names with a specific prefix.
        """
        page_length: int = 100
        current_page: int = 0
        work_queues: List[WorkQueue] = []
        while True:
            new_queues = await self.read_work_queues(
                work_pool_name=work_pool_name,
                offset=current_page * page_length,
                limit=page_length,
                work_queue_filter=WorkQueueFilter(name=WorkQueueFilterName(startswith_=prefixes)),
            )
            if not new_queues:
                break
            work_queues += new_queues
            current_page += 1
        return work_queues

    async def delete_work_queue_by_id(self, id: Union[str, int]) -> None:
        """
        Delete a work queue by its ID.
        """
        try:
            await self._client.delete(f"/work_queues/{id}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == status.HTTP_404_NOT_FOUND:
                raise prefect.exceptions.ObjectNotFound(http_exc=e) from e
            else:
                raise

    async def set_task_run_name(self, task_run_id: Union[str, int], name: str) -> httpx.Response:
        task_run_data = TaskRunUpdate(name=name)
        return await self._client.patch(f"/task_runs/{task_run_id}", json=task_run_data.model_dump(mode="json", exclude_unset=True))

    async def create_task_run(
        self,
        task: TaskObject,
        flow_run_id: Union[str, int],
        dynamic_key: Any,
        id: Optional[str] = None,
        name: Optional[str] = None,
        extra_tags: Optional[List[str]] = None,
        state: Optional[Any] = None,
        task_inputs: Optional[dict[str, Any]] = None,
    ) -> TaskRun:
        """
        Create a task run.
        """
        tags = set(task.tags).union(extra_tags or [])
        if state is None:
            state = prefect.states.Pending()
        retry_delay = task.retry_delay_seconds
        if isinstance(retry_delay, list):
            retry_delay = [int(rd) for rd in retry_delay]
        elif isinstance(retry_delay, float):
            retry_delay = int(retry_delay)
        task_run_data = TaskRunCreate(
            id=id,
            name=name,
            flow_run_id=flow_run_id,
            task_key=task.task_key,
            dynamic_key=str(dynamic_key),
            tags=list(tags),
            task_version=task.version,
            empirical_policy=TaskRunPolicy(retries=task.retries, retry_delay=retry_delay, retry_jitter_factor=task.retry_jitter_factor),
            state=prefect.states.to_state_create(state),
            task_inputs=task_inputs or {},
        )
        content: str = task_run_data.model_dump_json(exclude={"id"} if id is None else None)
        response = await self._client.post("/task_runs/", content=content)
        return TaskRun.model_validate(response.json())

    async def read_task_run(self, task_run_id: Union[str, int]) -> TaskRun:
        """
        Query the Prefect API for a task run by id.
        """
        try:
            response = await self._client.get(f"/task_runs/{task_run_id}")
            return TaskRun.model_validate(response.json())
        except httpx.HTTPStatusError as e:
            if e.response.status_code == status.HTTP_404_NOT_FOUND:
                raise prefect.exceptions.ObjectNotFound(http_exc=e) from e
            else:
                raise

    async def read_task_runs(
        self,
        *,
        flow_filter: Optional[FlowFilter] = None,
        flow_run_filter: Optional[FlowRunFilter] = None,
        task_run_filter: Optional[TaskRunFilter] = None,
        deployment_filter: Optional[DeploymentFilter] = None,
        sort: Optional[Any] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[TaskRun]:
        """
        Query the Prefect API for task runs. Only task runs matching all criteria will be returned.
        """
        body = {
            "flows": flow_filter.model_dump(mode="json") if flow_filter else None,
            "flow_runs": flow_run_filter.model_dump(mode="json", exclude_unset=True) if flow_run_filter else None,
            "task_runs": task_run_filter.model_dump(mode="json") if task_run_filter else None,
            "deployments": deployment_filter.model_dump(mode="json") if deployment_filter else None,
            "sort": sort,
            "limit": limit,
            "offset": offset,
        }
        response = await self._client.post("/task_runs/filter", json=body)
        return pydantic.TypeAdapter(list[TaskRun]).validate_python(response.json())

    async def delete_task_run(self, task_run_id: Union[str, int]) -> None:
        """
        Delete a task run by id.
        """
        try:
            await self._client.delete(f"/task_runs/{task_run_id}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise prefect.exceptions.ObjectNotFound(http_exc=e) from e
            else:
                raise

    async def set_task_run_state(self, task_run_id: Union[str, int], state: Any, force: bool = False) -> OrchestrationResult:
        """
        Set the state of a task run.
        """
        state_create = prefect.states.to_state_create(state)
        state_create.state_details.task_run_id = task_run_id
        response = await self._client.post(f"/task_runs/{task_run_id}/set_state", json=dict(state=state_create.model_dump(mode="json"), force=force))
        result = OrchestrationResult.model_validate(response.json())
        return result

    async def read_task_run_states(self, task_run_id: Union[str, int]) -> List[prefect.states.State]:
        """
        Query for the states of a task run.
        """
        response = await self._client.get("/task_run_states/", params=dict(task_run_id=str(task_run_id)))
        return pydantic.TypeAdapter(list[prefect.states.State]).validate_python(response.json())

    async def create_flow_run_notification_policy(
        self,
        block_document_id: Union[str, UUID],
        is_active: bool = True,
        tags: Optional[List[str]] = None,
        state_names: Optional[List[str]] = None,
        message_template: Optional[str] = None,
    ) -> UUID:
        """
        Create a notification policy for flow runs.
        """
        if tags is None:
            tags = []
        if state_names is None:
            state_names = []
        policy = FlowRunNotificationPolicyCreate(
            block_document_id=block_document_id, is_active=is_active, tags=tags, state_names=state_names, message_template=message_template
        )
        response = await self._client.post("/flow_run_notification_policies/", json=policy.model_dump(mode="json"))
        policy_id: Optional[str] = response.json().get("id")
        if not policy_id:
            raise httpx.RequestError(f"Malformed response: {response}")
        return UUID(policy_id)

    async def delete_flow_run_notification_policy(self, id: Union[str, UUID]) -> None:
        """
        Delete a flow run notification policy by id.
        """
        try:
            await self._client.delete(f"/flow_run_notification_policies/{id}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == status.HTTP_404_NOT_FOUND:
                raise prefect.exceptions.ObjectNotFound(http_exc=e) from e
            else:
                raise

    async def update_flow_run_notification_policy(
        self,
        id: Union[str, UUID],
        block_document_id: Optional[Union[str, UUID]] = None,
        is_active: Optional[bool] = None,
        tags: Optional[List[str]] = None,
        state_names: Optional[List[str]] = None,
        message_template: Optional[str] = None,
    ) -> None:
        """
        Update a notification policy for flow runs.
        """
        params: dict[str, Any] = {}
        if block_document_id is not None:
            params["block_document_id"] = block_document_id
        if is_active is not None:
            params["is_active"] = is_active
        if tags is not None:
            params["tags"] = tags
        if state_names is not None:
            params["state_names"] = state_names
        if message_template is not None:
            params["message_template"] = message_template
        policy = FlowRunNotificationPolicyUpdate(**params)
        try:
            await self._client.patch(f"/flow_run_notification_policies/{id}", json=policy.model_dump(mode="json", exclude_unset=True))
        except httpx.HTTPStatusError as e:
            if e.response.status_code == status.HTTP_404_NOT_FOUND:
                raise prefect.exceptions.ObjectNotFound(http_exc=e) from e
            else:
                raise

    async def read_flow_run_notification_policies(
        self,
        flow_run_notification_policy_filter: FlowRunNotificationPolicyFilter,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[FlowRunNotificationPolicy]:
        """
        Query the Prefect API for flow run notification policies.
        """
        body = {
            "flow_run_notification_policy_filter": flow_run_notification_policy_filter.model_dump(mode="json") if flow_run_notification_policy_filter else None,
            "limit": limit,
            "offset": offset,
        }
        response = await self._client.post("/flow_run_notification_policies/filter", json=body)
        return pydantic.TypeAdapter(list[FlowRunNotificationPolicy]).validate_python(response.json())

    async def read_work_queues(
        self,
        work_pool_name: Optional[str] = None,
        work_queue_filter: Optional[WorkQueueFilter] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[WorkQueue]:
        """
        Retrieves queues for a work pool.
        """
        json_data = {
            "work_queues": work_queue_filter.model_dump(mode="json", exclude_unset=True) if work_queue_filter else None,
            "limit": limit,
            "offset": offset,
        }
        if work_pool_name:
            try:
                response = await self._client.post(f"/work_pools/{work_pool_name}/queues/filter", json=json_data)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == status.HTTP_404_NOT_FOUND:
                    raise prefect.exceptions.ObjectNotFound(http_exc=e) from e
                else:
                    raise
        else:
            response = await self._client.post("/work_queues/filter", json=json_data)
        return pydantic.TypeAdapter(list[WorkQueue]).validate_python(response.json())

    async def read_worker_metadata(self) -> Any:
        """Reads worker metadata stored in Prefect collection registry."""
        response = await self._client.get("collections/views/aggregate-worker-metadata")
        response.raise_for_status()
        return response.json()

    async def api_version(self) -> Any:
        res = await self._client.get("/admin/version")
        return res.json()

    def client_version(self) -> str:
        return prefect.__version__

    async def raise_for_api_version_mismatch(self) -> None:
        if self.server_type == ServerType.CLOUD:
            return
        try:
            api_version_value = await self.api_version()
        except Exception as e:
            if "Unauthorized" in str(e):
                raise e
            raise RuntimeError(f"Failed to reach API at {self.api_url}") from e
        api_version_parsed = version.parse(api_version_value)
        client_version_parsed = version.parse(self.client_version())
        if api_version_parsed.major != client_version_parsed.major:
            raise RuntimeError(f"Found incompatible versions: client: {client_version_parsed}, server: {api_version_parsed}. Major versions must match.")

    async def __aenter__(self) -> PrefectClient:
        """
        Start the client.
        """
        if self._closed:
            raise RuntimeError("The client cannot be started again after closing. Retrieve a new client with `get_client()` instead.")
        self._context_stack += 1
        if self._started:
            return self
        self._loop = asyncio.get_running_loop()
        await self._exit_stack.__aenter__()
        if self._ephemeral_app and self.manage_lifespan:
            self._ephemeral_lifespan = await self._exit_stack.enter_async_context(app_lifespan_context(self._ephemeral_app))
        if self._ephemeral_app:
            self.logger.debug(f"Using ephemeral application with database at {PREFECT_API_DATABASE_CONNECTION_URL.value()}")
        else:
            self.logger.debug(f"Connecting to API at {self.api_url}")
        await self._exit_stack.enter_async_context(self._client)
        self._started = True
        return self

    async def __aexit__(self, *exc_info: Any) -> Any:
        """
        Shutdown the client.
        """
        self._context_stack -= 1
        if self._context_stack > 0:
            return
        self._closed = True
        return await self._exit_stack.__aexit__(*exc_info)

    def __enter__(self) -> NoReturn:
        raise RuntimeError("The `PrefectClient` must be entered with an async context. Use 'async with PrefectClient(...)' not 'with PrefectClient(...)'")

    def __exit__(self, *_: Any) -> None:
        assert False, "This should never be called but must be defined for __enter__"


class SyncPrefectClient(
    ArtifactClient,
    ArtifactCollectionClient,
    LogClient,
    VariableClient,
    ConcurrencyLimitClient,
    DeploymentClient,
    AutomationClient,
    SlaClient,
    FlowRunClient,
    FlowClient,
    BlocksDocumentClient,
    BlocksSchemaClient,
    BlocksTypeClient,
    WorkPoolClient,
):
    """
    A synchronous client for interacting with the Prefect REST API.
    """

    def __init__(
        self,
        api: Union[str, ASGIApp],
        *,
        auth_string: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
        httpx_settings: Optional[dict[str, Any]] = None,
        server_type: Optional[Any] = None,
    ) -> None:
        httpx_settings = httpx_settings.copy() if httpx_settings else {}
        httpx_settings.setdefault("headers", {})
        if PREFECT_API_TLS_INSECURE_SKIP_VERIFY:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            httpx_settings.setdefault("verify", ctx)
        else:
            cert_file: Optional[str] = PREFECT_API_SSL_CERT_FILE.value()
            if not cert_file:
                cert_file = certifi.where()
            ctx = ssl.create_default_context(cafile=cert_file)
            httpx_settings.setdefault("verify", ctx)
        if api_version is None:
            api_version = SERVER_API_VERSION
        httpx_settings["headers"].setdefault("X-PREFECT-API-VERSION", api_version)
        if api_key:
            httpx_settings["headers"].setdefault("Authorization", f"Bearer {api_key}")
        if auth_string:
            token = base64.b64encode(auth_string.encode("utf-8")).decode("utf-8")
            httpx_settings["headers"].setdefault("Authorization", f"Basic {token}")
        self._context_stack: int = 0
        self._ephemeral_app: Optional[ASGIApp] = None
        self.manage_lifespan: bool = True
        self._closed: bool = False
        self._started: bool = False
        if isinstance(api, str):
            if httpx_settings.get("app"):
                raise ValueError(
                    "Invalid httpx settings: `app` cannot be set when providing an api url. "
                    "`app` is only for use with ephemeral instances. Provide it as the `api` parameter instead."
                )
            httpx_settings.setdefault("base_url", api)
            httpx_settings.setdefault("limits", httpx.Limits(max_connections=16, max_keepalive_connections=8, keepalive_expiry=25))
            httpx_settings.setdefault("http2", PREFECT_API_ENABLE_HTTP2.value())
            if server_type:
                self.server_type = server_type
            else:
                self.server_type = ServerType.CLOUD if api.startswith(PREFECT_CLOUD_API_URL.value()) else ServerType.SERVER
        else:
            self._ephemeral_app = api
            self.server_type = ServerType.EPHEMERAL
        httpx_settings.setdefault(
            "timeout",
            httpx.Timeout(
                connect=PREFECT_API_REQUEST_TIMEOUT.value(),
                read=PREFECT_API_REQUEST_TIMEOUT.value(),
                write=PREFECT_API_REQUEST_TIMEOUT.value(),
                pool=PREFECT_API_REQUEST_TIMEOUT.value(),
            ),
        )
        if not PREFECT_TESTING_UNIT_TEST_MODE:
            httpx_settings.setdefault("follow_redirects", True)
        enable_csrf_support: bool = self.server_type != ServerType.CLOUD and PREFECT_CLIENT_CSRF_SUPPORT_ENABLED.value()
        self._client: PrefectHttpxSyncClient = PrefectHttpxSyncClient(**httpx_settings, enable_csrf_support=enable_csrf_support)
        if isinstance(api, str) and (not httpx_settings.get("transport")):
            transport_for_url = getattr(self._client, "_transport_for_url", None)
            if callable(transport_for_url):
                server_transport = transport_for_url(httpx.URL(api))
                if isinstance(server_transport, httpx.HTTPTransport):
                    pool = getattr(server_transport, "_pool", None)
                    if isinstance(pool, httpcore.ConnectionPool):
                        setattr(pool, "_retries", 3)
        self.logger: Logger = get_logger("client")

    @property
    def api_url(self) -> Any:
        """
        Get the base URL for the API.
        """
        return self._client.base_url

    def __enter__(self) -> SyncPrefectClient:
        """
        Start the client.
        """
        if self._closed:
            raise RuntimeError("The client cannot be started again after closing. Retrieve a new client with `get_client()` instead.")
        self._context_stack += 1
        if self._started:
            return self
        self._client.__enter__()
        self._started = True
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """
        Shutdown the client.
        """
        self._context_stack -= 1
        if self._context_stack > 0:
            return
        self._closed = True
        self._client.__exit__(*exc_info)

    def api_healthcheck(self) -> Optional[Exception]:
        """
        Attempts to connect to the API and returns the encountered exception if not successful.
        """
        try:
            self._client.get("/health")
            return None
        except Exception as exc:
            return exc

    def hello(self) -> httpx.Response:
        """
        Send a GET request to /hello for testing purposes.
        """
        return self._client.get("/hello")

    def api_version(self) -> Any:
        res = self._client.get("/admin/version")
        return res.json()

    def client_version(self) -> str:
        return prefect.__version__

    def raise_for_api_version_mismatch(self) -> None:
        if self.server_type == ServerType.CLOUD:
            return
        try:
            api_version_value = self.api_version()
        except Exception as e:
            if "Unauthorized" in str(e):
                raise e
            raise RuntimeError(f"Failed to reach API at {self.api_url}") from e
        api_version_parsed = version.parse(api_version_value)
        client_version_parsed = version.parse(self.client_version())
        if api_version_parsed.major != client_version_parsed.major:
            raise RuntimeError(f"Found incompatible versions: client: {client_version_parsed}, server: {api_version_parsed}. Major versions must match.")

    def set_task_run_name(self, task_run_id: Union[str, int], name: str) -> httpx.Response:
        task_run_data = TaskRunUpdate(name=name)
        return self._client.patch(f"/task_runs/{task_run_id}", json=task_run_data.model_dump(mode="json", exclude_unset=True))

    def create_task_run(
        self,
        task: TaskObject,
        flow_run_id: Union[str, int],
        dynamic_key: Any,
        id: Optional[str] = None,
        name: Optional[str] = None,
        extra_tags: Optional[List[str]] = None,
        state: Optional[Any] = None,
        task_inputs: Optional[dict[str, Any]] = None,
    ) -> TaskRun:
        """
        Create a task run.
        """
        tags = set(task.tags).union(extra_tags or [])
        if state is None:
            state = prefect.states.Pending()
        retry_delay = task.retry_delay_seconds
        if isinstance(retry_delay, list):
            retry_delay = [int(rd) for rd in retry_delay]
        elif isinstance(retry_delay, float):
            retry_delay = int(retry_delay)
        task_run_data = TaskRunCreate(
            id=id,
            name=name,
            flow_run_id=flow_run_id,
            task_key=task.task_key,
            dynamic_key=dynamic_key,
            tags=list(tags),
            task_version=task.version,
            empirical_policy=TaskRunPolicy(retries=task.retries, retry_delay=retry_delay, retry_jitter_factor=task.retry_jitter_factor),
            state=prefect.states.to_state_create(state),
            task_inputs=task_inputs or {},
        )
        content: str = task_run_data.model_dump_json(exclude={"id"} if id is None else None)
        response = self._client.post("/task_runs/", content=content)
        return TaskRun.model_validate(response.json())

    def read_task_run(self, task_run_id: Union[str, int]) -> TaskRun:
        """
        Query the Prefect API for a task run by id.
        """
        try:
            response = self._client.get(f"/task_runs/{task_run_id}")
            return TaskRun.model_validate(response.json())
        except httpx.HTTPStatusError as e:
            if e.response.status_code == status.HTTP_404_NOT_FOUND:
                raise prefect.exceptions.ObjectNotFound(http_exc=e) from e
            else:
                raise

    def read_task_runs(
        self,
        *,
        flow_filter: Optional[FlowFilter] = None,
        flow_run_filter: Optional[FlowRunFilter] = None,
        task_run_filter: Optional[TaskRunFilter] = None,
        deployment_filter: Optional[DeploymentFilter] = None,
        sort: Optional[Any] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[TaskRun]:
        """
        Query the Prefect API for task runs. Only task runs matching all criteria will be returned.
        """
        body = {
            "flows": flow_filter.model_dump(mode="json") if flow_filter else None,
            "flow_runs": flow_run_filter.model_dump(mode="json", exclude_unset=True) if flow_run_filter else None,
            "task_runs": task_run_filter.model_dump(mode="json") if task_run_filter else None,
            "deployments": deployment_filter.model_dump(mode="json") if deployment_filter else None,
            "sort": sort,
            "limit": limit,
            "offset": offset,
        }
        response = self._client.post("/task_runs/filter", json=body)
        return pydantic.TypeAdapter(list[TaskRun]).validate_python(response.json())

    def set_task_run_state(self, task_run_id: Union[str, int], state: Any, force: bool = False) -> OrchestrationResult:
        """
        Set the state of a task run.
        """
        state_create = prefect.states.to_state_create(state)
        state_create.state_details.task_run_id = task_run_id
        response = self._client.post(f"/task_runs/{task_run_id}/set_state", json=dict(state=state_create.model_dump(mode="json"), force=force))
        result = OrchestrationResult.model_validate(response.json())
        return result

    def read_task_run_states(self, task_run_id: Union[str, int]) -> List[prefect.states.State]:
        """
        Query for the states of a task run.
        """
        response = self._client.get("/task_run_states/", params=dict(task_run_id=str(task_run_id)))
        return pydantic.TypeAdapter(list[prefect.states.State]).validate_python(response.json())