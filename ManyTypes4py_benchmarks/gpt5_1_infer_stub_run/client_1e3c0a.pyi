from __future__ import annotations

import datetime
from collections.abc import Iterable
from typing import Any, Optional, Union
from uuid import UUID

from prefect.client.orchestration.base import BaseAsyncClient, BaseClient
from prefect.client.schemas.actions import DeploymentCreate, DeploymentScheduleCreate, DeploymentUpdate
from prefect.client.schemas.filters import (
    DeploymentFilter,
    FlowFilter,
    FlowRunFilter,
    TaskRunFilter,
    WorkPoolFilter,
    WorkQueueFilter,
)
from prefect.client.schemas.objects import ConcurrencyOptions, DeploymentSchedule, FlowRun
from prefect.client.schemas.responses import DeploymentResponse, FlowRunResponse
from prefect.client.schemas.schedules import SCHEDULE_TYPES
from prefect.client.schemas.sorting import DeploymentSort
from prefect.states import State
from prefect.types import KeyValueLabelsField


class DeploymentClient(BaseClient):
    def create_deployment(
        self,
        flow_id: UUID,
        name: str,
        version: Optional[str] = ...,
        schedules: Optional[list[DeploymentScheduleCreate]] = ...,
        concurrency_limit: Optional[int] = ...,
        concurrency_options: Optional[ConcurrencyOptions] = ...,
        parameters: Optional[dict[str, Any]] = ...,
        description: Optional[str] = ...,
        work_queue_name: Optional[str] = ...,
        work_pool_name: Optional[str] = ...,
        tags: Optional[Iterable[str]] = ...,
        storage_document_id: Optional[UUID] = ...,
        path: Optional[str] = ...,
        entrypoint: Optional[str] = ...,
        infrastructure_document_id: Optional[UUID] = ...,
        parameter_openapi_schema: Optional[dict[str, Any]] = ...,
        paused: Optional[bool] = ...,
        pull_steps: Optional[list[dict[str, Any]]] = ...,
        enforce_parameter_schema: Optional[bool] = ...,
        job_variables: Optional[dict[str, Any]] = ...,
    ) -> UUID: ...
    def set_deployment_paused_state(self, deployment_id: Union[UUID, str], paused: bool) -> None: ...
    def update_deployment(self, deployment_id: Union[UUID, str], deployment: DeploymentUpdate) -> None: ...
    def _create_deployment_from_schema(self, schema: DeploymentCreate) -> UUID: ...
    def read_deployment(self, deployment_id: Union[UUID, str]) -> DeploymentResponse: ...
    def read_deployment_by_name(self, name: str) -> DeploymentResponse: ...
    def read_deployments(
        self,
        *,
        flow_filter: Optional[FlowFilter] = ...,
        flow_run_filter: Optional[FlowRunFilter] = ...,
        task_run_filter: Optional[TaskRunFilter] = ...,
        deployment_filter: Optional[DeploymentFilter] = ...,
        work_pool_filter: Optional[WorkPoolFilter] = ...,
        work_queue_filter: Optional[WorkQueueFilter] = ...,
        limit: Optional[int] = ...,
        sort: Optional[DeploymentSort] = ...,
        offset: int = ...,
    ) -> list[DeploymentResponse]: ...
    def delete_deployment(self, deployment_id: Union[UUID, str]) -> None: ...
    def create_deployment_schedules(
        self,
        deployment_id: Union[UUID, str],
        schedules: list[tuple[SCHEDULE_TYPES, bool]],
    ) -> list[DeploymentSchedule]: ...
    def read_deployment_schedules(self, deployment_id: Union[UUID, str]) -> list[DeploymentSchedule]: ...
    def update_deployment_schedule(
        self,
        deployment_id: Union[UUID, str],
        schedule_id: Union[UUID, str],
        active: Optional[bool] = ...,
        schedule: Optional[SCHEDULE_TYPES] = ...,
    ) -> None: ...
    def delete_deployment_schedule(self, deployment_id: Union[UUID, str], schedule_id: Union[UUID, str]) -> None: ...
    def get_scheduled_flow_runs_for_deployments(
        self,
        deployment_ids: Iterable[Union[UUID, str]],
        scheduled_before: Optional[datetime.datetime] = ...,
        limit: Optional[int] = ...,
    ) -> list[FlowRunResponse]: ...
    def create_flow_run_from_deployment(
        self,
        deployment_id: Union[UUID, str],
        *,
        parameters: Optional[dict[str, Any]] = ...,
        context: Optional[dict[str, Any]] = ...,
        state: Optional[State] = ...,
        name: Optional[str] = ...,
        tags: Optional[Iterable[str]] = ...,
        idempotency_key: Optional[str] = ...,
        parent_task_run_id: Optional[UUID] = ...,
        work_queue_name: Optional[str] = ...,
        job_variables: Optional[dict[str, Any]] = ...,
        labels: Optional[KeyValueLabelsField] = ...,
    ) -> FlowRun: ...


class DeploymentAsyncClient(BaseAsyncClient):
    async def create_deployment(
        self,
        flow_id: UUID,
        name: str,
        version: Optional[str] = ...,
        schedules: Optional[list[DeploymentScheduleCreate]] = ...,
        concurrency_limit: Optional[int] = ...,
        concurrency_options: Optional[ConcurrencyOptions] = ...,
        parameters: Optional[dict[str, Any]] = ...,
        description: Optional[str] = ...,
        work_queue_name: Optional[str] = ...,
        work_pool_name: Optional[str] = ...,
        tags: Optional[Iterable[str]] = ...,
        storage_document_id: Optional[UUID] = ...,
        path: Optional[str] = ...,
        entrypoint: Optional[str] = ...,
        infrastructure_document_id: Optional[UUID] = ...,
        parameter_openapi_schema: Optional[dict[str, Any]] = ...,
        paused: Optional[bool] = ...,
        pull_steps: Optional[list[dict[str, Any]]] = ...,
        enforce_parameter_schema: Optional[bool] = ...,
        job_variables: Optional[dict[str, Any]] = ...,
    ) -> UUID: ...
    async def set_deployment_paused_state(self, deployment_id: Union[UUID, str], paused: bool) -> None: ...
    async def update_deployment(self, deployment_id: Union[UUID, str], deployment: DeploymentUpdate) -> None: ...
    async def _create_deployment_from_schema(self, schema: DeploymentCreate) -> UUID: ...
    async def read_deployment(self, deployment_id: Union[UUID, str]) -> DeploymentResponse: ...
    async def read_deployment_by_name(self, name: str) -> DeploymentResponse: ...
    async def read_deployments(
        self,
        *,
        flow_filter: Optional[FlowFilter] = ...,
        flow_run_filter: Optional[FlowRunFilter] = ...,
        task_run_filter: Optional[TaskRunFilter] = ...,
        deployment_filter: Optional[DeploymentFilter] = ...,
        work_pool_filter: Optional[WorkPoolFilter] = ...,
        work_queue_filter: Optional[WorkQueueFilter] = ...,
        limit: Optional[int] = ...,
        sort: Optional[DeploymentSort] = ...,
        offset: int = ...,
    ) -> list[DeploymentResponse]: ...
    async def delete_deployment(self, deployment_id: Union[UUID, str]) -> None: ...
    async def create_deployment_schedules(
        self,
        deployment_id: Union[UUID, str],
        schedules: list[tuple[SCHEDULE_TYPES, bool]],
    ) -> list[DeploymentSchedule]: ...
    async def read_deployment_schedules(self, deployment_id: Union[UUID, str]) -> list[DeploymentSchedule]: ...
    async def update_deployment_schedule(
        self,
        deployment_id: Union[UUID, str],
        schedule_id: Union[UUID, str],
        active: Optional[bool] = ...,
        schedule: Optional[SCHEDULE_TYPES] = ...,
    ) -> None: ...
    async def delete_deployment_schedule(self, deployment_id: Union[UUID, str], schedule_id: Union[UUID, str]) -> None: ...
    async def get_scheduled_flow_runs_for_deployments(
        self,
        deployment_ids: Iterable[Union[UUID, str]],
        scheduled_before: Optional[datetime.datetime] = ...,
        limit: Optional[int] = ...,
    ) -> list[FlowRun]: ...
    async def create_flow_run_from_deployment(
        self,
        deployment_id: Union[UUID, str],
        *,
        parameters: Optional[dict[str, Any]] = ...,
        context: Optional[dict[str, Any]] = ...,
        state: Optional[State] = ...,
        name: Optional[str] = ...,
        tags: Optional[Iterable[str]] = ...,
        idempotency_key: Optional[str] = ...,
        parent_task_run_id: Optional[UUID] = ...,
        work_queue_name: Optional[str] = ...,
        job_variables: Optional[dict[str, Any]] = ...,
        labels: Optional[KeyValueLabelsField] = ...,
    ) -> FlowRun: ...