import asyncio
import base64
import datetime
import ssl
from collections.abc import Iterable
from contextlib import AsyncExitStack
from logging import Logger
from typing import TYPE_CHECKING, Any, Literal, NoReturn, Optional, Union, overload
from uuid import UUID
import certifi
import httpcore
import httpx
import pydantic
from asgi_lifespan import LifespanManager
from packaging import version
from starlette import status
from typing_extensions import ParamSpec, Self, TypeVar
from prefect.client.orchestration._artifacts.client import ArtifactClient, ArtifactAsyncClient, ArtifactCollectionClient, ArtifactCollectionAsyncClient
from prefect.client.orchestration._concurrency_limits.client import ConcurrencyLimitAsyncClient, ConcurrencyLimitClient
from prefect.client.orchestration._logs.client import LogClient, LogAsyncClient
from prefect.client.orchestration._variables.client import VariableClient, VariableAsyncClient
from prefect.client.orchestration._deployments.client import DeploymentClient, DeploymentAsyncClient
from prefect.client.orchestration._automations.client import AutomationClient, AutomationAsyncClient
from prefect.client.orchestration._work_pools.client import WorkPoolClient, WorkPoolAsyncClient
from prefect._experimental.sla.client import SlaClient, SlaAsyncClient
from prefect.client.orchestration._flows.client import FlowClient, FlowAsyncClient
from prefect.client.orchestration._flow_runs.client import FlowRunClient, FlowRunAsyncClient
from prefect.client.orchestration._blocks_documents.client import BlocksDocumentClient, BlocksDocumentAsyncClient
from prefect.client.orchestration._blocks_schemas.client import BlocksSchemaClient, BlocksSchemaAsyncClient
from prefect.client.orchestration._blocks_types.client import BlocksTypeClient, BlocksTypeAsyncClient
import prefect
import prefect.exceptions
import prefect.settings
import prefect.states
from prefect.client.constants import SERVER_API_VERSION
from prefect.client.schemas import FlowRun, OrchestrationResult, TaskRun
from prefect.client.schemas.actions import FlowRunNotificationPolicyCreate, FlowRunNotificationPolicyUpdate, TaskRunCreate, TaskRunUpdate, WorkQueueCreate, WorkQueueUpdate
from prefect.client.schemas.filters import DeploymentFilter, FlowFilter, FlowRunFilter, FlowRunNotificationPolicyFilter, TaskRunFilter, WorkQueueFilter, WorkQueueFilterName
from prefect.client.schemas.objects import Constant, FlowRunNotificationPolicy, Parameter, TaskRunPolicy, TaskRunResult, WorkQueue, WorkQueueStatusDetail
from prefect.client.schemas.sorting import TaskRunSort
from prefect.logging import get_logger
from prefect.settings import PREFECT_API_AUTH_STRING, PREFECT_API_DATABASE_CONNECTION_URL, PREFECT_API_ENABLE_HTTP2, PREFECT_API_KEY, PREFECT_API_REQUEST_TIMEOUT, PREFECT_API_SSL_CERT_FILE, PREFECT_API_TLS_INSECURE_SKIP_VERIFY, PREFECT_API_URL, PREFECT_CLIENT_CSRF_SUPPORT_ENABLED, PREFECT_CLOUD_API_URL, PREFECT_SERVER_ALLOW_EPHEMERAL_MODE, PREFECT_TESTING_UNIT_TEST_MODE
from prefect.types._datetime import now
if TYPE_CHECKING:
    from prefect.tasks import Task as TaskObject
from prefect.client.base import ASGIApp, PrefectHttpxAsyncClient, PrefectHttpxSyncClient, ServerType, app_lifespan_context
P = ParamSpec('P')
R = TypeVar('R', infer_variance=True)
T = TypeVar('T')

@overload
def get_client(*, httpx_settings: Optional[dict[str, Any]] = None, sync_client: Literal[False] = False) -> "PrefectClient":
    ...

@overload
def get_client(*, httpx_settings: Optional[dict[str, Any]] = None, sync_client: Literal[True] = ...) -> "SyncPrefectClient":
    ...

@overload
def get_client(*, httpx_settings: Optional[dict[str, Any]] = None, sync_client: bool = ...) -> Union["PrefectClient", "SyncPrefectClient"]:
    ...

def get_client(httpx_settings: Optional[dict[str, Any]] = None, sync_client: bool = False) -> Union["PrefectClient", "SyncPrefectClient"]:
    """
    Retrieve a HTTP client for communicating with the Prefect REST API.

    The client must be context managed; for example:

    