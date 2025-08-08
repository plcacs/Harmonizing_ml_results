from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional, List
from httpx import HTTPStatusError, RequestError
from prefect.client.orchestration.base import BaseAsyncClient, BaseClient
from prefect.exceptions import ObjectNotFound
if TYPE_CHECKING:
    from uuid import UUID
    from prefect.client.schemas.filters import DeploymentFilter, FlowFilter, FlowRunFilter, TaskRunFilter, WorkPoolFilter, WorkQueueFilter
    from prefect.client.schemas.objects import Flow
    from prefect.client.schemas.sorting import FlowSort
    from prefect.flows import Flow as FlowObject

class FlowClient(BaseClient):

    def create_flow(self, flow: FlowObject) -> UUID:
        ...

    def create_flow_from_name(self, flow_name: str) -> UUID:
        ...

    def read_flow(self, flow_id: UUID) -> Flow:
        ...

    def delete_flow(self, flow_id: UUID) -> None:
        ...

    def read_flows(self, *, flow_filter: Optional[FlowFilter] = None, flow_run_filter: Optional[FlowRunFilter] = None, task_run_filter: Optional[TaskRunFilter] = None, deployment_filter: Optional[DeploymentFilter] = None, work_pool_filter: Optional[WorkPoolFilter] = None, work_queue_filter: Optional[WorkQueueFilter] = None, sort: Optional[FlowSort] = None, limit: Optional[int] = None, offset: int = 0) -> List[Flow]:
        ...

    def read_flow_by_name(self, flow_name: str) -> Flow:
        ...

class FlowAsyncClient(BaseAsyncClient):

    async def create_flow(self, flow: FlowObject) -> UUID:
        ...

    async def create_flow_from_name(self, flow_name: str) -> UUID:
        ...

    async def read_flow(self, flow_id: UUID) -> Flow:
        ...

    async def delete_flow(self, flow_id: UUID) -> None:
        ...

    async def read_flows(self, *, flow_filter: Optional[FlowFilter] = None, flow_run_filter: Optional[FlowRunFilter] = None, task_run_filter: Optional[TaskRunFilter] = None, deployment_filter: Optional[DeploymentFilter] = None, work_pool_filter: Optional[WorkPoolFilter] = None, work_queue_filter: Optional[WorkQueueFilter] = None, sort: Optional[FlowSort] = None, limit: Optional[int] = None, offset: int = 0) -> List[Flow]:
        ...

    async def read_flow_by_name(self, flow_name: str) -> Flow:
        ...
