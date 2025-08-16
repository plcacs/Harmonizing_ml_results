from __future__ import annotations
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any
import httpx
from typing_extensions import TypeVar
from prefect.client.orchestration.base import BaseAsyncClient, BaseClient
from prefect.exceptions import ObjectNotFound
T = TypeVar('T')
R = TypeVar('R', infer_variance=True)
if TYPE_CHECKING:
    from uuid import UUID
    from prefect.client.schemas import FlowRun, OrchestrationResult
    from prefect.client.schemas.filters import DeploymentFilter, FlowFilter, FlowRunFilter, TaskRunFilter, WorkPoolFilter, WorkQueueFilter
    from prefect.client.schemas.objects import FlowRunInput, FlowRunPolicy
    from prefect.client.schemas.sorting import FlowRunSort
    from prefect.flows import Flow as FlowObject
    from prefect.states import State
    from prefect.types import KeyValueLabelsField
