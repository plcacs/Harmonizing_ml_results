from __future__ import annotations
from typing import TYPE_CHECKING, Any, List
from httpx import HTTPStatusError, RequestError
from prefect.client.orchestration.base import BaseAsyncClient, BaseClient
from prefect.exceptions import ObjectNotFound
if TYPE_CHECKING:
    from uuid import UUID
    from httpx import Response
    from prefect.client.schemas.actions import GlobalConcurrencyLimitCreate, GlobalConcurrencyLimitUpdate
    from prefect.client.schemas.objects import ConcurrencyLimit
    from prefect.client.schemas.responses import GlobalConcurrencyLimitResponse

class ConcurrencyLimitClient(BaseClient):

    def create_concurrency_limit(self, tag: str, concurrency_limit: int) -> UUID:
    
    def read_concurrency_limit_by_tag(self, tag: str) -> ConcurrencyLimit:
    
    def read_concurrency_limits(self, limit: int, offset: int) -> List[ConcurrencyLimit]:
    
    def reset_concurrency_limit_by_tag(self, tag: str, slot_override: List[str] = None) -> None:
    
    def delete_concurrency_limit_by_tag(self, tag: str) -> None:
    
    def increment_v1_concurrency_slots(self, names: List[str], task_run_id: UUID) -> Any:
    
    def decrement_v1_concurrency_slots(self, names: List[str], task_run_id: UUID, occupancy_seconds: float) -> Any:
    
    def increment_concurrency_slots(self, names: List[str], slots: int, mode: str, create_if_missing: bool = None) -> Any:
    
    def release_concurrency_slots(self, names: List[str], slots: int, occupancy_seconds: float) -> Any:
    
    def create_global_concurrency_limit(self, concurrency_limit: GlobalConcurrencyLimitCreate) -> UUID:
    
    def update_global_concurrency_limit(self, name: str, concurrency_limit: GlobalConcurrencyLimitUpdate) -> Any:
    
    def delete_global_concurrency_limit_by_name(self, name: str) -> Any:
    
    def read_global_concurrency_limit_by_name(self, name: str) -> GlobalConcurrencyLimitResponse:
    
    def upsert_global_concurrency_limit_by_name(self, name: str, limit: int) -> None:
    
    def read_global_concurrency_limits(self, limit: int = 10, offset: int = 0) -> List[GlobalConcurrencyLimitResponse]:

class ConcurrencyLimitAsyncClient(BaseAsyncClient):

    async def create_concurrency_limit(self, tag: str, concurrency_limit: int) -> UUID:
    
    async def read_concurrency_limit_by_tag(self, tag: str) -> ConcurrencyLimit:
    
    async def read_concurrency_limits(self, limit: int, offset: int) -> List[ConcurrencyLimit]:
    
    async def reset_concurrency_limit_by_tag(self, tag: str, slot_override: List[str] = None) -> None:
    
    async def delete_concurrency_limit_by_tag(self, tag: str) -> None:
    
    async def increment_v1_concurrency_slots(self, names: List[str], task_run_id: UUID) -> Any:
    
    async def decrement_v1_concurrency_slots(self, names: List[str], task_run_id: UUID, occupancy_seconds: float) -> Any:
    
    async def increment_concurrency_slots(self, names: List[str], slots: int, mode: str, create_if_missing: bool = None) -> Any:
    
    async def release_concurrency_slots(self, names: List[str], slots: int, occupancy_seconds: float) -> Any:
    
    async def create_global_concurrency_limit(self, concurrency_limit: GlobalConcurrencyLimitCreate) -> UUID:
    
    async def update_global_concurrency_limit(self, name: str, concurrency_limit: GlobalConcurrencyLimitUpdate) -> Any:
    
    async def delete_global_concurrency_limit_by_name(self, name: str) -> Any:
    
    async def read_global_concurrency_limit_by_name(self, name: str) -> GlobalConcurrencyLimitResponse:
    
    async def upsert_global_concurrency_limit_by_name(self, name: str, limit: int) -> None:
    
    async def read_global_concurrency_limits(self, limit: int = 10, offset: int = 0) -> List[GlobalConcurrencyLimitResponse]:
