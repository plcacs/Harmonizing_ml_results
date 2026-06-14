from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID

from prefect.client.orchestration.base import BaseAsyncClient, BaseClient
from prefect.client.schemas.actions import WorkPoolCreate, WorkPoolUpdate
from prefect.client.schemas.filters import WorkerFilter, WorkPoolFilter
from prefect.client.schemas.objects import Worker, WorkerMetadata, WorkPool
from prefect.client.schemas.responses import WorkerFlowRunResponse


class WorkPoolClient(BaseClient):
    def send_worker_heartbeat(
        self,
        work_pool_name: str,
        worker_name: str,
        heartbeat_interval_seconds: Optional[float] = None,
        get_worker_id: bool = False,
        worker_metadata: Optional[WorkerMetadata] = None,
    ) -> Optional[UUID]: ...

    def read_workers_for_work_pool(
        self,
        work_pool_name: str,
        worker_filter: Optional[WorkerFilter] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> list[Worker]: ...

    def read_work_pool(self, work_pool_name: str) -> WorkPool: ...

    def read_work_pools(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
        work_pool_filter: Optional[WorkPoolFilter] = None,
    ) -> list[WorkPool]: ...

    def create_work_pool(
        self,
        work_pool: WorkPoolCreate,
        overwrite: bool = False,
    ) -> WorkPool: ...

    def update_work_pool(
        self,
        work_pool_name: str,
        work_pool: WorkPoolUpdate,
    ) -> None: ...

    def delete_work_pool(self, work_pool_name: str) -> None: ...

    def get_scheduled_flow_runs_for_work_pool(
        self,
        work_pool_name: str,
        work_queue_names: Optional[list[str]] = None,
        scheduled_before: Optional[datetime] = None,
    ) -> list[WorkerFlowRunResponse]: ...


class WorkPoolAsyncClient(BaseAsyncClient):
    async def send_worker_heartbeat(
        self,
        work_pool_name: str,
        worker_name: str,
        heartbeat_interval_seconds: Optional[float] = None,
        get_worker_id: bool = False,
        worker_metadata: Optional[WorkerMetadata] = None,
    ) -> Optional[UUID]: ...

    async def read_workers_for_work_pool(
        self,
        work_pool_name: str,
        worker_filter: Optional[WorkerFilter] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> list[Worker]: ...

    async def read_work_pool(self, work_pool_name: str) -> WorkPool: ...

    async def read_work_pools(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
        work_pool_filter: Optional[WorkPoolFilter] = None,
    ) -> list[WorkPool]: ...

    async def create_work_pool(
        self,
        work_pool: WorkPoolCreate,
        overwrite: bool = False,
    ) -> WorkPool: ...

    async def update_work_pool(
        self,
        work_pool_name: str,
        work_pool: WorkPoolUpdate,
    ) -> None: ...

    async def delete_work_pool(self, work_pool_name: str) -> None: ...

    async def get_scheduled_flow_runs_for_work_pool(
        self,
        work_pool_name: str,
        work_queue_names: Optional[list[str]] = None,
        scheduled_before: Optional[datetime] = None,
    ) -> list[WorkerFlowRunResponse]: ...