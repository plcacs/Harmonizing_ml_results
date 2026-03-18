```python
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional, Union, List
from datetime import datetime
from httpx import HTTPStatusError
from prefect.client.base import ServerType
from prefect.client.orchestration.base import BaseAsyncClient, BaseClient
from prefect.exceptions import ObjectAlreadyExists, ObjectNotFound

if TYPE_CHECKING:
    from uuid import UUID
    from prefect.client.schemas.actions import WorkPoolCreate, WorkPoolUpdate
    from prefect.client.schemas.filters import WorkerFilter, WorkPoolFilter
    from prefect.client.schemas.objects import Worker, WorkerMetadata, WorkPool
    from prefect.client.schemas.responses import WorkerFlowRunResponse

class WorkPoolClient(BaseClient):
    def send_worker_heartbeat(
        self,
        work_pool_name: str,
        worker_name: str,
        heartbeat_interval_seconds: Optional[int] = ...,
        get_worker_id: bool = ...,
        worker_metadata: Optional[Any] = ...
    ) -> Optional[UUID]: ...
    
    def read_workers_for_work_pool(
        self,
        work_pool_name: str,
        worker_filter: Optional[WorkerFilter] = ...,
        offset: Optional[int] = ...,
        limit: Optional[int] = ...
    ) -> List[Worker]: ...
    
    def read_work_pool(
        self,
        work_pool_name: str
    ) -> WorkPool: ...
    
    def read_work_pools(
        self,
        limit: Optional[int] = ...,
        offset: int = ...,
        work_pool_filter: Optional[WorkPoolFilter] = ...
    ) -> List[WorkPool]: ...
    
    def create_work_pool(
        self,
        work_pool: WorkPoolCreate,
        overwrite: bool = ...
    ) -> WorkPool: ...
    
    def update_work_pool(
        self,
        work_pool_name: str,
        work_pool: WorkPoolUpdate
    ) -> None: ...
    
    def delete_work_pool(
        self,
        work_pool_name: str
    ) -> None: ...
    
    def get_scheduled_flow_runs_for_work_pool(
        self,
        work_pool_name: str,
        work_queue_names: Optional[List[str]] = ...,
        scheduled_before: Optional[Union[str, datetime]] = ...
    ) -> List[WorkerFlowRunResponse]: ...

class WorkPoolAsyncClient(BaseAsyncClient):
    async def send_worker_heartbeat(
        self,
        work_pool_name: str,
        worker_name: str,
        heartbeat_interval_seconds: Optional[int] = ...,
        get_worker_id: bool = ...,
        worker_metadata: Optional[Any] = ...
    ) -> Optional[UUID]: ...
    
    async def read_workers_for_work_pool(
        self,
        work_pool_name: str,
        worker_filter: Optional[WorkerFilter] = ...,
        offset: Optional[int] = ...,
        limit: Optional[int] = ...
    ) -> List[Worker]: ...
    
    async def read_work_pool(
        self,
        work_pool_name: str
    ) -> WorkPool: ...
    
    async def read_work_pools(
        self,
        limit: Optional[int] = ...,
        offset: int = ...,
        work_pool_filter: Optional[WorkPoolFilter] = ...
    ) -> List[WorkPool]: ...
    
    async def create_work_pool(
        self,
        work_pool: WorkPoolCreate,
        overwrite: bool = ...
    ) -> WorkPool: ...
    
    async def update_work_pool(
        self,
        work_pool_name: str,
        work_pool: WorkPoolUpdate
    ) -> None: ...
    
    async def delete_work_pool(
        self,
        work_pool_name: str
    ) -> None: ...
    
    async def get_scheduled_flow_runs_for_work_pool(
        self,
        work_pool_name: str,
        work_queue_names: Optional[List[str]] = ...,
        scheduled_before: Optional[Union[str, datetime]] = ...
    ) -> List[WorkerFlowRunResponse]: ...
```