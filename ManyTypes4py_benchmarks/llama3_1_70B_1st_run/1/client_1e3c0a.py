from __future__ import annotations
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Union, Optional
from httpx import HTTPStatusError, RequestError
from prefect.client.orchestration.base import BaseAsyncClient, BaseClient
from prefect.exceptions import ObjectNotFound
if TYPE_CHECKING:
    import datetime
    from uuid import UUID
    from prefect.client.schemas import FlowRun
    from prefect.client.schemas.actions import DeploymentCreate, DeploymentScheduleCreate, DeploymentUpdate
    from prefect.client.schemas.filters import DeploymentFilter, FlowFilter, FlowRunFilter, TaskRunFilter, WorkPoolFilter, WorkQueueFilter
    from prefect.client.schemas.objects import ConcurrencyOptions, DeploymentSchedule
    from prefect.client.schemas.responses import DeploymentResponse, FlowRunResponse
    from prefect.client.schemas.schedules import SCHEDULE_TYPES
    from prefect.client.schemas.sorting import DeploymentSort
    from prefect.states import State
    from prefect.types import KeyValueLabelsField

class DeploymentClient(BaseClient):

    def create_deployment(self, flow_id: UUID, name: str, version: Optional[str] = None, schedules: Optional[list[tuple[str, bool]]] = None, concurrency_limit: Optional[int] = None, concurrency_options: Optional[ConcurrencyOptions] = None, parameters: Optional[dict[str, Any]] = None, description: Optional[str] = None, work_queue_name: Optional[str] = None, work_pool_name: Optional[str] = None, tags: Optional[list[str]] = None, storage_document_id: Optional[UUID] = None, path: Optional[str] = None, entrypoint: Optional[str] = None, infrastructure_document_id: Optional[UUID] = None, parameter_openapi_schema: Optional[dict[str, Any]] = None, paused: Optional[bool] = None, pull_steps: Optional[bool] = None, enforce_parameter_schema: Optional[bool] = None, job_variables: Optional[dict[str, Any]] = None) -> UUID:
        ...

    def set_deployment_paused_state(self, deployment_id: UUID, paused: bool) -> None:
        ...

    def update_deployment(self, deployment_id: UUID, deployment: dict[str, Any]) -> None:
        ...

    def _create_deployment_from_schema(self, schema: DeploymentCreate) -> UUID:
        ...

    def read_deployment(self, deployment_id: UUID) -> DeploymentResponse:
        ...

    def read_deployment_by_name(self, name: str) -> DeploymentResponse:
        ...

    def read_deployments(self, *, flow_filter: Optional[FlowFilter] = None, flow_run_filter: Optional[FlowRunFilter] = None, task_run_filter: Optional[TaskRunFilter] = None, deployment_filter: Optional[DeploymentFilter] = None, work_pool_filter: Optional[WorkPoolFilter] = None, work_queue_filter: Optional[WorkQueueFilter] = None, limit: Optional[int] = None, sort: Optional[DeploymentSort] = None, offset: int = 0) -> list[DeploymentResponse]:
        ...

    def delete_deployment(self, deployment_id: UUID) -> None:
        ...

    def create_deployment_schedules(self, deployment_id: UUID, schedules: list[tuple[str, bool]]) -> list[DeploymentSchedule]:
        ...

    def read_deployment_schedules(self, deployment_id: UUID) -> list[DeploymentSchedule]:
        ...

    def update_deployment_schedule(self, deployment_id: UUID, schedule_id: UUID, active: Optional[bool] = None, schedule: Optional[str] = None) -> None:
        ...

    def delete_deployment_schedule(self, deployment_id: UUID, schedule_id: UUID) -> None:
        ...

    def get_scheduled_flow_runs_for_deployments(self, deployment_ids: list[UUID], scheduled_before: Optional[datetime.datetime] = None, limit: Optional[int] = None) -> list[FlowRunResponse]:
        ...

    def create_flow_run_from_deployment(self, deployment_id: UUID, *, parameters: Optional[dict[str, Any]] = None, context: Optional[dict[str, Any]] = None, state: Optional[State] = None, name: Optional[str] = None, tags: Optional[list[str]] = None, idempotency_key: Optional[str] = None, parent_task_run_id: Optional[UUID] = None, work_queue_name: Optional[str] = None, job_variables: Optional[dict[str, Any]] = None, labels: Optional[KeyValueLabelsField] = None) -> FlowRun:
        ...

class DeploymentAsyncClient(BaseAsyncClient):

    async def create_deployment(self, flow_id: UUID, name: str, version: Optional[str] = None, schedules: Optional[list[tuple[str, bool]]] = None, concurrency_limit: Optional[int] = None, concurrency_options: Optional[ConcurrencyOptions] = None, parameters: Optional[dict[str, Any]] = None, description: Optional[str] = None, work_queue_name: Optional[str] = None, work_pool_name: Optional[str] = None, tags: Optional[list[str]] = None, storage_document_id: Optional[UUID] = None, path: Optional[str] = None, entrypoint: Optional[str] = None, infrastructure_document_id: Optional[UUID] = None, parameter_openapi_schema: Optional[dict[str, Any]] = None, paused: Optional[bool] = None, pull_steps: Optional[bool] = None, enforce_parameter_schema: Optional[bool] = None, job_variables: Optional[dict[str, Any]] = None) -> UUID:
        ...

    async def set_deployment_paused_state(self, deployment_id: UUID, paused: bool) -> None:
        ...

    async def update_deployment(self, deployment_id: UUID, deployment: dict[str, Any]) -> None:
        ...

    async def _create_deployment_from_schema(self, schema: DeploymentCreate) -> UUID:
        ...

    async def read_deployment(self, deployment_id: UUID) -> DeploymentResponse:
        ...

    async def read_deployment_by_name(self, name: str) -> DeploymentResponse:
        ...

    async def read_deployments(self, *, flow_filter: Optional[FlowFilter] = None, flow_run_filter: Optional[FlowRunFilter] = None, task_run_filter: Optional[TaskRunFilter] = None, deployment_filter: Optional[DeploymentFilter] = None, work_pool_filter: Optional[WorkPoolFilter] = None, work_queue_filter: Optional[WorkQueueFilter] = None, limit: Optional[int] = None, sort: Optional[DeploymentSort] = None, offset: int = 0) -> list[DeploymentResponse]:
        ...

    async def delete_deployment(self, deployment_id: UUID) -> None:
        ...

    async def create_deployment_schedules(self, deployment_id: UUID, schedules: list[tuple[str, bool]]) -> list[DeploymentSchedule]:
        ...

    async def read_deployment_schedules(self, deployment_id: UUID) -> list[DeploymentSchedule]:
        ...

    async def update_deployment_schedule(self, deployment_id: UUID, schedule_id: UUID, active: Optional[bool] = None, schedule: Optional[str] = None) -> None:
        ...

    async def delete_deployment_schedule(self, deployment_id: UUID, schedule_id: UUID) -> None:
        ...

    async def get_scheduled_flow_runs_for_deployments(self, deployment_ids: list[UUID], scheduled_before: Optional[datetime.datetime] = None, limit: Optional[int] = None) -> list[FlowRun]:
        ...

    async def create_flow_run_from_deployment(self, deployment_id: UUID, *, parameters: Optional[dict[str, Any]] = None, context: Optional[dict[str, Any]] = None, state: Optional[State] = None, name: Optional[str] = None, tags: Optional[list[str]] = None, idempotency_key: Optional[str] = None, parent_task_run_id: Optional[UUID] = None, work_queue_name: Optional[str] = None, job_variables: Optional[dict[str, Any]] = None, labels: Optional[KeyValueLabelsField] = None) -> FlowRun:
        ...
