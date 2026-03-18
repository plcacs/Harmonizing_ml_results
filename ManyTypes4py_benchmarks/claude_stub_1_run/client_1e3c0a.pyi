```pyi
from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Union
from uuid import UUID

from httpx import HTTPStatusError, RequestError
from prefect.client.orchestration.base import BaseAsyncClient, BaseClient
from prefect.exceptions import ObjectNotFound

if TYPE_CHECKING:
    import datetime
    from prefect.client.schemas import FlowRun
    from prefect.client.schemas.actions import (
        DeploymentCreate,
        DeploymentScheduleCreate,
        DeploymentUpdate,
    )
    from prefect.client.schemas.filters import (
        DeploymentFilter,
        FlowFilter,
        FlowRunFilter,
        TaskRunFilter,
        WorkPoolFilter,
        WorkQueueFilter,
    )
    from prefect.client.schemas.objects import ConcurrencyOptions, DeploymentSchedule
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
        version: str | None = None,
        schedules: list[tuple[SCHEDULE_TYPES, bool]] | None = None,
        concurrency_limit: int | None = None,
        concurrency_options: ConcurrencyOptions | None = None,
        parameters: dict[str, Any] | None = None,
        description: str | None = None,
        work_queue_name: str | None = None,
        work_pool_name: str | None = None,
        tags: Iterable[str] | None = None,
        storage_document_id: UUID | None = None,
        path: str | None = None,
        entrypoint: str | None = None,
        infrastructure_document_id: UUID | None = None,
        parameter_openapi_schema: dict[str, Any] | None = None,
        paused: bool | None = None,
        pull_steps: list[dict[str, Any]] | None = None,
        enforce_parameter_schema: bool | None = None,
        job_variables: dict[str, Any] | None = None,
    ) -> UUID: ...
    def set_deployment_paused_state(self, deployment_id: UUID, paused: bool) -> None: ...
    def update_deployment(self, deployment_id: UUID, deployment: DeploymentUpdate) -> None: ...
    def _create_deployment_from_schema(self, schema: DeploymentCreate) -> UUID: ...
    def read_deployment(self, deployment_id: UUID | str) -> DeploymentResponse: ...
    def read_deployment_by_name(self, name: str) -> DeploymentResponse: ...
    def read_deployments(
        self,
        *,
        flow_filter: FlowFilter | None = None,
        flow_run_filter: FlowRunFilter | None = None,
        task_run_filter: TaskRunFilter | None = None,
        deployment_filter: DeploymentFilter | None = None,
        work_pool_filter: WorkPoolFilter | None = None,
        work_queue_filter: WorkQueueFilter | None = None,
        limit: int | None = None,
        sort: DeploymentSort | None = None,
        offset: int = 0,
    ) -> list[DeploymentResponse]: ...
    def delete_deployment(self, deployment_id: UUID) -> None: ...
    def create_deployment_schedules(
        self,
        deployment_id: UUID,
        schedules: list[tuple[SCHEDULE_TYPES, bool]],
    ) -> list[DeploymentSchedule]: ...
    def read_deployment_schedules(self, deployment_id: UUID) -> list[DeploymentSchedule]: ...
    def update_deployment_schedule(
        self,
        deployment_id: UUID,
        schedule_id: UUID,
        active: bool | None = None,
        schedule: SCHEDULE_TYPES | None = None,
    ) -> None: ...
    def delete_deployment_schedule(self, deployment_id: UUID, schedule_id: UUID) -> None: ...
    def get_scheduled_flow_runs_for_deployments(
        self,
        deployment_ids: Iterable[UUID],
        scheduled_before: datetime.datetime | None = None,
        limit: int | None = None,
    ) -> list[FlowRunResponse]: ...
    def create_flow_run_from_deployment(
        self,
        deployment_id: UUID,
        *,
        parameters: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        state: State | None = None,
        name: str | None = None,
        tags: Iterable[str] | None = None,
        idempotency_key: str | None = None,
        parent_task_run_id: UUID | None = None,
        work_queue_name: str | None = None,
        job_variables: dict[str, Any] | None = None,
        labels: KeyValueLabelsField | None = None,
    ) -> FlowRun: ...

class DeploymentAsyncClient(BaseAsyncClient):
    async def create_deployment(
        self,
        flow_id: UUID,
        name: str,
        version: str | None = None,
        schedules: list[tuple[SCHEDULE_TYPES, bool]] | None = None,
        concurrency_limit: int | None = None,
        concurrency_options: ConcurrencyOptions | None = None,
        parameters: dict[str, Any] | None = None,
        description: str | None = None,
        work_queue_name: str | None = None,
        work_pool_name: str | None = None,
        tags: Iterable[str] | None = None,
        storage_document_id: UUID | None = None,
        path: str | None = None,
        entrypoint: str | None = None,
        infrastructure_document_id: UUID | None = None,
        parameter_openapi_schema: dict[str, Any] | None = None,
        paused: bool | None = None,
        pull_steps: list[dict[str, Any]] | None = None,
        enforce_parameter_schema: bool | None = None,
        job_variables: dict[str, Any] | None = None,
    ) -> UUID: ...
    async def set_deployment_paused_state(self, deployment_id: UUID, paused: bool) -> None: ...
    async def update_deployment(self, deployment_id: UUID, deployment: DeploymentUpdate) -> None: ...
    async def _create_deployment_from_schema(self, schema: DeploymentCreate) -> UUID: ...
    async def read_deployment(self, deployment_id: UUID | str) -> DeploymentResponse: ...
    async def read_deployment_by_name(self, name: str) -> DeploymentResponse: ...
    async def read_deployments(
        self,
        *,
        flow_filter: FlowFilter | None = None,
        flow_run_filter: FlowRunFilter | None = None,
        task_run_filter: TaskRunFilter | None = None,
        deployment_filter: DeploymentFilter | None = None,
        work_pool_filter: WorkPoolFilter | None = None,
        work_queue_filter: WorkQueueFilter | None = None,
        limit: int | None = None,
        sort: DeploymentSort | None = None,
        offset: int = 0,
    ) -> list[DeploymentResponse]: ...
    async def delete_deployment(self, deployment_id: UUID) -> None: ...
    async def create_deployment_schedules(
        self,
        deployment_id: UUID,
        schedules: list[tuple[SCHEDULE_TYPES, bool]],
    ) -> list[DeploymentSchedule]: ...
    async def read_deployment_schedules(self, deployment_id: UUID) -> list[DeploymentSchedule]: ...
    async def update_deployment_schedule(
        self,
        deployment_id: UUID,
        schedule_id: UUID,
        active: bool | None = None,
        schedule: SCHEDULE_TYPES | None = None,
    ) -> None: ...
    async def delete_deployment_schedule(self, deployment_id: UUID, schedule_id: UUID) -> None: ...
    async def get_scheduled_flow_runs_for_deployments(
        self,
        deployment_ids: Iterable[UUID],
        scheduled_before: datetime.datetime | None = None,
        limit: int | None = None,
    ) -> list[FlowRunResponse]: ...
    async def create_flow_run_from_deployment(
        self,
        deployment_id: UUID,
        *,
        parameters: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        state: State | None = None,
        name: str | None = None,
        tags: Iterable[str] | None = None,
        idempotency_key: str | None = None,
        parent_task_run_id: UUID | None = None,
        work_queue_name: str | None = None,
        job_variables: dict[str, Any] | None = None,
        labels: KeyValueLabelsField | None = None,
    ) -> FlowRun: ...
```