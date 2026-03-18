from typing import Any, Optional
from uuid import UUID
from prefect.client.orchestration.base import BaseAsyncClient, BaseClient

class DeploymentClient(BaseClient):
    def create_deployment(
        self,
        flow_id: Any,
        name: Any,
        version: Optional[Any] = ...,
        schedules: Optional[Any] = ...,
        concurrency_limit: Optional[Any] = ...,
        concurrency_options: Optional[Any] = ...,
        parameters: Optional[Any] = ...,
        description: Optional[Any] = ...,
        work_queue_name: Optional[Any] = ...,
        work_pool_name: Optional[Any] = ...,
        tags: Optional[Any] = ...,
        storage_document_id: Optional[Any] = ...,
        path: Optional[Any] = ...,
        entrypoint: Optional[Any] = ...,
        infrastructure_document_id: Optional[Any] = ...,
        parameter_openapi_schema: Optional[Any] = ...,
        paused: Optional[Any] = ...,
        pull_steps: Optional[Any] = ...,
        enforce_parameter_schema: Optional[Any] = ...,
        job_variables: Optional[Any] = ...,
    ) -> UUID: ...
    def set_deployment_paused_state(self, deployment_id: Any, paused: Any) -> None: ...
    def update_deployment(self, deployment_id: Any, deployment: Any) -> None: ...
    def _create_deployment_from_schema(self, schema: Any) -> UUID: ...
    def read_deployment(self, deployment_id: Any) -> Any: ...
    def read_deployment_by_name(self, name: Any) -> Any: ...
    def read_deployments(
        self,
        *,
        flow_filter: Optional[Any] = ...,
        flow_run_filter: Optional[Any] = ...,
        task_run_filter: Optional[Any] = ...,
        deployment_filter: Optional[Any] = ...,
        work_pool_filter: Optional[Any] = ...,
        work_queue_filter: Optional[Any] = ...,
        limit: Optional[Any] = ...,
        sort: Optional[Any] = ...,
        offset: Any = ...,
    ) -> list[Any]: ...
    def delete_deployment(self, deployment_id: Any) -> None: ...
    def create_deployment_schedules(self, deployment_id: Any, schedules: Any) -> list[Any]: ...
    def read_deployment_schedules(self, deployment_id: Any) -> list[Any]: ...
    def update_deployment_schedule(
        self,
        deployment_id: Any,
        schedule_id: Any,
        active: Optional[Any] = ...,
        schedule: Optional[Any] = ...,
    ) -> None: ...
    def delete_deployment_schedule(self, deployment_id: Any, schedule_id: Any) -> None: ...
    def get_scheduled_flow_runs_for_deployments(
        self,
        deployment_ids: Any,
        scheduled_before: Optional[Any] = ...,
        limit: Optional[Any] = ...,
    ) -> list[Any]: ...
    def create_flow_run_from_deployment(
        self,
        deployment_id: Any,
        *,
        parameters: Optional[Any] = ...,
        context: Optional[Any] = ...,
        state: Optional[Any] = ...,
        name: Optional[Any] = ...,
        tags: Optional[Any] = ...,
        idempotency_key: Optional[Any] = ...,
        parent_task_run_id: Optional[Any] = ...,
        work_queue_name: Optional[Any] = ...,
        job_variables: Optional[Any] = ...,
        labels: Optional[Any] = ...,
    ) -> Any: ...

class DeploymentAsyncClient(BaseAsyncClient):
    async def create_deployment(
        self,
        flow_id: Any,
        name: Any,
        version: Optional[Any] = ...,
        schedules: Optional[Any] = ...,
        concurrency_limit: Optional[Any] = ...,
        concurrency_options: Optional[Any] = ...,
        parameters: Optional[Any] = ...,
        description: Optional[Any] = ...,
        work_queue_name: Optional[Any] = ...,
        work_pool_name: Optional[Any] = ...,
        tags: Optional[Any] = ...,
        storage_document_id: Optional[Any] = ...,
        path: Optional[Any] = ...,
        entrypoint: Optional[Any] = ...,
        infrastructure_document_id: Optional[Any] = ...,
        parameter_openapi_schema: Optional[Any] = ...,
        paused: Optional[Any] = ...,
        pull_steps: Optional[Any] = ...,
        enforce_parameter_schema: Optional[Any] = ...,
        job_variables: Optional[Any] = ...,
    ) -> UUID: ...
    async def set_deployment_paused_state(self, deployment_id: Any, paused: Any) -> None: ...
    async def update_deployment(self, deployment_id: Any, deployment: Any) -> None: ...
    async def _create_deployment_from_schema(self, schema: Any) -> UUID: ...
    async def read_deployment(self, deployment_id: Any) -> Any: ...
    async def read_deployment_by_name(self, name: Any) -> Any: ...
    async def read_deployments(
        self,
        *,
        flow_filter: Optional[Any] = ...,
        flow_run_filter: Optional[Any] = ...,
        task_run_filter: Optional[Any] = ...,
        deployment_filter: Optional[Any] = ...,
        work_pool_filter: Optional[Any] = ...,
        work_queue_filter: Optional[Any] = ...,
        limit: Optional[Any] = ...,
        sort: Optional[Any] = ...,
        offset: Any = ...,
    ) -> list[Any]: ...
    async def delete_deployment(self, deployment_id: Any) -> None: ...
    async def create_deployment_schedules(self, deployment_id: Any, schedules: Any) -> list[Any]: ...
    async def read_deployment_schedules(self, deployment_id: Any) -> list[Any]: ...
    async def update_deployment_schedule(
        self,
        deployment_id: Any,
        schedule_id: Any,
        active: Optional[Any] = ...,
        schedule: Optional[Any] = ...,
    ) -> None: ...
    async def delete_deployment_schedule(self, deployment_id: Any, schedule_id: Any) -> None: ...
    async def get_scheduled_flow_runs_for_deployments(
        self,
        deployment_ids: Any,
        scheduled_before: Optional[Any] = ...,
        limit: Optional[Any] = ...,
    ) -> list[Any]: ...
    async def create_flow_run_from_deployment(
        self,
        deployment_id: Any,
        *,
        parameters: Optional[Any] = ...,
        context: Optional[Any] = ...,
        state: Optional[Any] = ...,
        name: Optional[Any] = ...,
        tags: Optional[Any] = ...,
        idempotency_key: Optional[Any] = ...,
        parent_task_run_id: Optional[Any] = ...,
        work_queue_name: Optional[Any] = ...,
        job_variables: Optional[Any] = ...,
        labels: Optional[Any] = ...,
    ) -> Any: ...