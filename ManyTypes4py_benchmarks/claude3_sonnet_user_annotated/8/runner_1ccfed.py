from __future__ import annotations

import asyncio
import datetime
import inspect
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Coroutine,
    Dict,
    Iterable,
    List,
    Optional,
    TypedDict,
    Union,
    cast,
    Set,
)
from uuid import UUID, uuid4

import anyio
import anyio.abc
from cachetools import LRUCache
from typing_extensions import Self

from prefect._internal.concurrency.api import (
    create_call,
    from_async,
    from_sync,
)
from prefect.client.orchestration import PrefectClient, get_client
from prefect.client.schemas.filters import (
    FlowRunFilter,
    FlowRunFilterId,
    FlowRunFilterState,
    FlowRunFilterStateName,
    FlowRunFilterStateType,
)
from prefect.client.schemas.objects import (
    ConcurrencyLimitConfig,
    State,
    StateType,
)
from prefect.client.schemas.objects import Flow as APIFlow
from prefect.events import DeploymentTriggerTypes, TriggerTypes
from prefect.events.related import tags_as_related_resources
from prefect.events.schemas.events import RelatedResource
from prefect.events.utilities import emit_event
from prefect.exceptions import Abort, ObjectNotFound
from prefect.flows import Flow, FlowStateHook, load_flow_from_flow_run
from prefect.logging.loggers import PrefectLogAdapter, flow_run_logger, get_logger
from prefect.runner.storage import RunnerStorage
from prefect.schedules import Schedule
from prefect.settings import (
    PREFECT_API_URL,
    PREFECT_RUNNER_SERVER_ENABLE,
    get_current_settings,
)
from prefect.states import (
    Crashed,
    Pending,
    exception_to_failed_state,
)
from prefect.types._datetime import DateTime
from prefect.types.entrypoint import EntrypointType
from prefect.utilities.asyncutils import (
    asyncnullcontext,
    is_async_fn,
    sync_compatible,
)
from prefect.utilities.engine import propose_state
from prefect.utilities.processutils import (
    get_sys_executable,
    run_process,
)
from prefect.utilities.services import (
    critical_service_loop,
    start_client_metrics_server,
)
from prefect.utilities.slugify import slugify

if TYPE_CHECKING:
    import concurrent.futures

    from prefect.client.schemas.objects import FlowRun
    from prefect.client.schemas.responses import DeploymentResponse
    from prefect.client.types.flexible_schedule_list import FlexibleScheduleList
    from prefect.deployments.runner import RunnerDeployment


__all__ = ["Runner"]


class ProcessMapEntry(TypedDict):
    flow_run: FlowRun
    pid: int


class Runner:
    def __init__(
        self,
        name: Optional[str] = None,
        query_seconds: Optional[float] = None,
        prefetch_seconds: float = 10,
        heartbeat_seconds: Optional[float] = None,
        limit: Optional[int] = None,
        pause_on_shutdown: bool = True,
        webserver: bool = False,
    ):
        """
        Responsible for managing the execution of remotely initiated flow runs.

        Args:
            name: The name of the runner. If not provided, a random one
                will be generated. If provided, it cannot contain '/' or '%'.
            query_seconds: The number of seconds to wait between querying for
                scheduled flow runs; defaults to `PREFECT_RUNNER_POLL_FREQUENCY`
            prefetch_seconds: The number of seconds to prefetch flow runs for.
            heartbeat_seconds: The number of seconds to wait between emitting
                flow run heartbeats. The runner will not emit heartbeats if the value is None.
                Defaults to `PREFECT_RUNNER_HEARTBEAT_FREQUENCY`.
            limit: The maximum number of flow runs this runner should be running at
            pause_on_shutdown: A boolean for whether or not to automatically pause
                deployment schedules on shutdown; defaults to `True`
            webserver: a boolean flag for whether to start a webserver for this runner

        Examples:
            Set up a Runner to manage the execute of scheduled flow runs for two flows:
                