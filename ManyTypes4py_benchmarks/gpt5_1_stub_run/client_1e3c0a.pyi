from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from uuid import UUID
import datetime
from prefect.client.orchestration.base import BaseAsyncClient, BaseClient
from prefect.client.schemas.objects import DeploymentSchedule, FlowRun
from prefect.client.schemas.responses import DeploymentResponse, FlowRunResponse
from prefect.states import State

class DeploymentClient(BaseClient):
    def create_deployment(
        self,
        flow_id: Union[str, UUID],
        name: str,
        version: Optional[str] = ...,
        schedules: Optional[Iterable[Any]] = ...,
        concurrency_limit: Optional[int] = ...,
        concurrency_options: Optional[Any] = ...,
        parameters: Optional[Dict[str, Any]] = ...,
        description: Optional[str] = ...,
        work_queue_name: Optional[str] = ...,
        work_pool_name: Optional[str] = ...,
        tags: Optional[Iterable[str]] = ...,
        storage_document_id: Optional[Union[str, UUID]] = ...,
        path: Optional[str] = ...,
        entrypoint: Optional[str] = ...,
        infrastructure_document_id: Optional[Union[str, UUID]] = ...,
        parameter_openapi_schema: Optional[Dict[str, Any]] = ...,
        paused: Optional[bool] = ...,
        pull_steps: Optional[Iterable[Any]] = ...,
        enforce_parameter_schema: Optional[bool] = ...,
        job_variables: Optional[Dict[str, Any]] = ...,
    ) -> UUID: ...
    def set_deployment_paused_state(self, deployment_id: Union[str, UUID], paused: bool) -> None: ...
    def update_deployment(self, deployment_id: Union[str, UUID], deployment: Any) -> None: ...
    def _create_deployment_from_schema(self, schema: Any) -> UUID: ...
    def read_deployment(self, deployment_id: Union[str, UUID]) -> DeploymentResponse: ...
    def read_deployment_by_name(self, name: str) -> DeploymentResponse: ...
    def read_deployments(
        self,
        *,
        flow_filter: Optional[Any] = ...,
        flow_run_filter: Optional[Any] = ...,
        task_run_filter: Optional[Any] = ...,
        deployment_filter: Optional[Any] = ...,
        work_pool_filter: Optional[Any] = ...,
        work_queue_filter: Optional[Any] = ...,
        limit: Optional[int] = ...,
        sort: Optional[Any] = ...,
        offset: int = ...,
    ) -> List[DeploymentResponse]: ...
    def delete_deployment(self, deployment_id: Union[str, UUID]) -> None: ...
    def create_deployment_schedules(
        self,
        deployment_id: Union[str, UUID],
        schedules: Iterable[Tuple[Any, bool]],
    ) -> List[DeploymentSchedule]: ...
    def read_deployment_schedules(self, deployment_id: Union[str, UUID]) -> List[DeploymentSchedule]: ...
    def update_deployment_schedule(
        self,
        deployment_id: Union[str, UUID],
        schedule_id: Union[str, UUID],
        active: Optional[bool] = ...,
        schedule: Optional[Any] = ...,
    ) -> None: ...
    def delete_deployment_schedule(self, deployment_id: Union[str, UUID], schedule_id: Union[str, UUID]) -> None: ...
    def get_scheduled_flow_runs_for_deployments(
        self,
        deployment_ids: Iterable[Union[str, UUID]],
        scheduled_before: Optional[Union[datetime.datetime, str]] = ...,
        limit: Optional[int] = ...,
    ) -> List[FlowRunResponse]: ...
    def create_flow_run_from_deployment(
        self,
        deployment_id: Union[str, UUID],
        *,
        parameters: Optional[Dict[str, Any]] = ...,
        context: Optional[Dict[str, Any]] = ...,
        state: Optional[State] = ...,
        name: Optional[str] = ...,
        tags: Optional[Iterable[str]] = ...,
        idempotency_key: Optional[str] = ...,
        parent_task_run_id: Optional[Union[str, UUID]] = ...,
        work_queue_name: Optional[str] = ...,
        job_variables: Optional[Dict[str, Any]] = ...,
        labels: Optional[Dict[str, Any]] = ...,
    ) -> FlowRun: ...

class DeploymentAsyncClient(BaseAsyncClient):
    async def create_deployment(
        self,
        flow_id: Union[str, UUID],
        name: str,
        version: Optional[str] = ...,
        schedules: Optional[Iterable[Any]] = ...,
        concurrency_limit: Optional[int] = ...,
        concurrency_options: Optional[Any] = ...,
        parameters: Optional[Dict[str, Any]] = ...,
        description: Optional[str] = ...,
        work_queue_name: Optional[str] = ...,
        work_pool_name: Optional[str] = ...,
        tags: Optional[Iterable[str]] = ...,
        storage_document_id: Optional[Union[str, UUID]] = ...,
        path: Optional[str] = ...,
        entrypoint: Optional[str] = ...,
        infrastructure_document_id: Optional[Union[str, UUID]] = ...,
        parameter_openapi_schema: Optional[Dict[str, Any]] = ...,
        paused: Optional[bool] = ...,
        pull_steps: Optional[Iterable[Any]] = ...,
        enforce_parameter_schema: Optional[bool] = ...,
        job_variables: Optional[Dict[str, Any]] = ...,
    ) -> UUID: ...
    async def set_deployment_paused_state(self, deployment_id: Union[str, UUID], paused: bool) -> None: ...
    async def update_deployment(self, deployment_id: Union[str, UUID], deployment: Any) -> None: ...
    async def _create_deployment_from_schema(self, schema: Any) -> UUID: ...
    async def read_deployment(self, deployment_id: Union[str, UUID]) -> DeploymentResponse: ...
    async def read_deployment_by_name(self, name: str) -> DeploymentResponse: ...
    async def read_deployments(
        self,
        *,
        flow_filter: Optional[Any] = ...,
        flow_run_filter: Optional[Any] = ...,
        task_run_filter: Optional[Any] = ...,
        deployment_filter: Optional[Any] = ...,
        work_pool_filter: Optional[Any] = ...,
        work_queue_filter: Optional[Any] = ...,
        limit: Optional[int] = ...,
        sort: Optional[Any] = ...,
        offset: int = ...,
    ) -> List[DeploymentResponse]: ...
    async def delete_deployment(self, deployment_id: Union[str, UUID]) -> None: ...
    async def create_deployment_schedules(
        self,
        deployment_id: Union[str, UUID],
        schedules: Iterable[Tuple[Any, bool]],
    ) -> List[DeploymentSchedule]: ...
    async def read_deployment_schedules(self, deployment_id: Union[str, UUID]) -> List[DeploymentSchedule]: ...
    async def update_deployment_schedule(
        self,
        deployment_id: Union[str, UUID],
        schedule_id: Union[str, UUID],
        active: Optional[bool] = ...,
        schedule: Optional[Any] = ...,
    ) -> None: ...
    async def delete_deployment_schedule(self, deployment_id: Union[str, UUID], schedule_id: Union[str, UUID]) -> None: ...
    async def get_scheduled_flow_runs_for_deployments(
        self,
        deployment_ids: Iterable[Union[str, UUID]],
        scheduled_before: Optional[Union[datetime.datetime, str]] = ...,
        limit: Optional[int] = ...,
    ) -> List[FlowRun]: ...
    async def create_flow_run_from_deployment(
        self,
        deployment_id: Union[str, UUID],
        *,
        parameters: Optional[Dict[str, Any]] = ...,
        context: Optional[Dict[str, Any]] = ...,
        state: Optional[State] = ...,
        name: Optional[str] = ...,
        tags: Optional[Iterable[str]] = ...,
        idempotency_key: Optional[str] = ...,
        parent_task_run_id: Optional[Union[str, UUID]] = ...,
        work_queue_name: Optional[str] = ...,
        job_variables: Optional[Dict[str, Any]] = ...,
        labels: Optional[Dict[str, Any]] = ...,
    ) -> FlowRun: ...