from __future__ import annotations
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from datetime import datetime
from httpx import HTTPStatusError, RequestError
from prefect.client.orchestration.base import BaseAsyncClient, BaseClient
from prefect.exceptions import ObjectNotFound
from prefect.client.schemas import FlowRun
from prefect.client.schemas.actions import DeploymentCreate, DeploymentScheduleCreate, DeploymentUpdate
from prefect.client.schemas.filters import DeploymentFilter, FlowFilter, FlowRunFilter, TaskRunFilter, WorkPoolFilter, WorkQueueFilter
from prefect.client.schemas.objects import ConcurrencyOptions, DeploymentSchedule
from prefect.client.schemas.responses import DeploymentResponse, FlowRunResponse
from prefect.client.schemas.schedules import SCHEDULE_TYPES
from prefect.client.schemas.sorting import DeploymentSort
from prefect.states import State
from prefect.types import KeyValueLabelsField
from uuid import UUID

class DeploymentClient(BaseClient):
    def create_deployment(self, flow_id: UUID, name: str, version: Optional[str] = None, schedules: Optional[List[Tuple[Any, bool]]] = None, concurrency_limit: Optional[int] = None, concurrency_options: Optional[ConcurrencyOptions] = None, parameters: Optional[Dict[str, Any]] = None, description: Optional[str] = None, work_queue_name: Optional[str] = None, work_pool_name: Optional[str] = None, tags: Optional[List[str]] = None, storage_document_id: Optional[UUID] = None, path: Optional[str] = None, entrypoint: Optional[str] = None, infrastructure_document_id: Optional[UUID] = None, parameter_openapi_schema: Optional[Dict[str, Any]] = None, paused: Optional[bool] = None, pull_steps: Optional[bool] = None, enforce_parameter_schema: Optional[bool] = None, job_variables: Optional[Dict[str, Any]] = None) -> UUID:
        pass

    def set_deployment_paused_state(self, deployment_id: UUID, paused: bool) -> None:
        pass

    def update_deployment(self, deployment_id: UUID, deployment: DeploymentUpdate) -> None:
        pass

    def _create_deployment_from_schema(self, schema: DeploymentCreate) -> UUID:
        pass

    def read_deployment(self, deployment_id: UUID) -> DeploymentResponse:
        pass

    def read_deployment_by_name(self, name: str) -> DeploymentResponse:
        pass

    def read_deployments(self, *, flow_filter: Optional[FlowFilter] = None, flow_run_filter: Optional[FlowRunFilter] = None, task_run_filter: Optional[TaskRunFilter] = None, deployment_filter: Optional[DeploymentFilter] = None, work_pool_filter: Optional[WorkPoolFilter] = None, work_queue_filter: Optional[WorkQueueFilter] = None, limit: Optional[int] = None, sort: Optional[DeploymentSort] = None, offset: int = 0) -> List[DeploymentResponse]:
        pass

    def delete_deployment(self, deployment_id: UUID) -> None:
        pass

    def create_deployment_schedules(self, deployment_id: UUID, schedules: List[Tuple[Any, bool]]) -> List[DeploymentSchedule]:
        pass

    def read_deployment_schedules(self, deployment_id: UUID) -> List[DeploymentSchedule]:
        pass

    def update_deployment_schedule(self, deployment_id: UUID, schedule_id: UUID, active: Optional[bool] = None, schedule: Optional[Any] = None) -> None:
        pass

    def delete_deployment_schedule(self, deployment_id: UUID, schedule_id: UUID) -> None:
        pass

    def get_scheduled_flow_runs_for_deployments(self, deployment_ids: List[UUID], scheduled_before: Optional[datetime] = None, limit: Optional[int] = None) -> List[FlowRun]:
        pass

    def create_flow_run_from_deployment(self, deployment_id: UUID, *, parameters: Optional[Dict[str, Any]] = None, context: Optional[Dict[str, Any]] = None, state: Optional[State] = None, name: Optional[str] = None, tags: Optional[List[str]] = None, idempotency_key: Optional[str] = None, parent_task_run_id: Optional[UUID] = None, work_queue_name: Optional[str] = None, job_variables: Optional[Dict[str, Any]] = None, labels: Optional[KeyValueLabelsField] = None) -> FlowRun:
        pass

class DeploymentAsyncClient(BaseAsyncClient):
    async def create_deployment(self, flow_id: UUID, name: str, version: Optional[str] = None, schedules: Optional[List[Tuple[Any, bool]]] = None, concurrency_limit: Optional[int] = None, concurrency_options: Optional[ConcurrencyOptions] = None, parameters: Optional[Dict[str, Any]] = None, description: Optional[str] = None, work_queue_name: Optional[str] = None, work_pool_name: Optional[str] = None, tags: Optional[List[str]] = None, storage_document_id: Optional[UUID] = None, path: Optional[str] = None, entrypoint: Optional[str] = None, infrastructure_document_id: Optional[UUID] = None, parameter_openapi_schema: Optional[Dict[str, Any]] = None, paused: Optional[bool] = None, pull_steps: Optional[bool] = None, enforce_parameter_schema: Optional[bool] = None, job_variables: Optional[Dict[str, Any]] = None) -> UUID:
        pass

    async def set_deployment_paused_state(self, deployment_id: UUID, paused: bool) -> None:
        pass

    async def update_deployment(self, deployment_id: UUID, deployment: DeploymentUpdate) -> None:
        pass

    async def _create_deployment_from_schema(self, schema: DeploymentCreate) -> UUID:
        pass

    async def read_deployment(self, deployment_id: UUID) -> DeploymentResponse:
        pass

    async def read_deployment_by_name(self, name: str) -> DeploymentResponse:
        pass

    async def read_deployments(self, *, flow_filter: Optional[FlowFilter] = None, flow_run_filter: Optional[FlowRunFilter] = None, task_run_filter: Optional[TaskRunFilter] = None, deployment_filter: Optional[DeploymentFilter] = None, work_pool_filter: Optional[WorkPoolFilter] = None, work_queue_filter: Optional[WorkQueueFilter] = None, limit: Optional[int] = None, sort: Optional[DeploymentSort] = None, offset: int = 0) -> List[DeploymentResponse]:
        pass

    async def delete_deployment(self, deployment_id: UUID) -> None:
        pass

    async def create_deployment_schedules(self, deployment_id: UUID, schedules: List[Tuple[Any, bool]]) -> List[DeploymentSchedule]:
        pass

    async def read_deployment_schedules(self, deployment_id: UUID) -> List[DeploymentSchedule]:
        pass

    async def update_deployment_schedule(self, deployment_id: UUID, schedule_id: UUID, active: Optional[bool] = None, schedule: Optional[Any] = None) -> None:
        pass

    async def delete_deployment_schedule(self, deployment_id: UUID, schedule_id: UUID) -> None:
        pass

    async def get_scheduled_flow_runs_for_deployments(self, deployment_ids: List[UUID], scheduled_before: Optional[datetime] = None, limit: Optional[int] = None) -> List[FlowRun]:
        pass

    async def create_flow_run_from_deployment(self, deployment_id: UUID, *, parameters: Optional[Dict[str, Any]] = None, context: Optional[Dict[str, Any]] = None, state: Optional[State] = None, name: Optional[str] = None, tags: Optional[List[str]] = None, idempotency_key: Optional[str] = None, parent_task_run_id: Optional[UUID] = None, work_queue_name: Optional[str] = None, job_variables: Optional[Dict[str, Any]] = None, labels: Optional[KeyValueLabelsField] = None) -> FlowRun:
        pass