from datetime import datetime, timedelta
from typing import Any, Dict, List, MutableMapping, Optional, Set, Union
from uuid import UUID, uuid4
from cachetools import TTLCache
from sqlalchemy.ext.asyncio import AsyncSession
from prefect.server import models, schemas
from prefect.server.database.orm_models import ORMDeployment, ORMFlow, ORMFlowRun, ORMFlowRunState, ORMTaskRun, ORMTaskRunState, ORMWorkPool, ORMWorkQueue
from prefect.server.events.schemas.events import Event
from prefect.server.models import deployments
from prefect.server.schemas.statuses import DeploymentStatus
from prefect.settings import PREFECT_API_EVENTS_RELATED_RESOURCE_CACHE_TTL
from prefect.types._datetime import DateTime, now
from prefect.utilities.text import truncated_to

ResourceData: Dict[str, Dict[str, Any]] = {}
RelatedResourceList: List[Dict[str, str]] = []
TRUNCATE_STATE_MESSAGES_AT: int = 100000
_flow_run_resource_data_cache: TTLCache = TTLCache(maxsize=1000, ttl=PREFECT_API_EVENTS_RELATED_RESOURCE_CACHE_TTL.value().total_seconds())

async def flow_run_state_change_event(session: AsyncSession, occurred: DateTime, flow_run: ORMFlowRun, initial_state_id: UUID, initial_state: ORMFlowRunState, validated_state_id: UUID, validated_state: ORMFlowRunState) -> Event:
    ...

async def _flow_run_related_resources_from_orm(session: AsyncSession, flow_run: ORMFlowRun) -> RelatedResourceList:
    ...

def _as_resource_data(flow_run: ORMFlowRun, flow: ORMFlow, deployment: ORMDeployment, work_queue: ORMWorkQueue, work_pool: ORMWorkPool, task_run: Optional[ORMTaskRun] = None) -> ResourceData:
    ...

def _resource_data_as_related_resources(resource_data: ResourceData, excluded_kinds: Optional[List[str]] = None) -> RelatedResourceList:
    ...

def _provenance_as_related_resources(created_by: Optional[Union[ORMDeployment, ORMFlowRun]]) -> RelatedResourceList:
    ...

def _state_type(state: Optional[ORMFlowRunState]) -> Optional[str]:
    ...

def state_payload(state: Optional[ORMFlowRunState]) -> Optional[Dict[str, Any]]:
    ...

def _timing_is_tight(occurred: DateTime, initial_state: ORMFlowRunState) -> bool:
    ...

async def deployment_status_event(session: AsyncSession, deployment_id: UUID, status: DeploymentStatus, occurred: DateTime) -> Event:
    ...

async def work_queue_status_event(session: AsyncSession, work_queue: ORMWorkQueue, occurred: DateTime) -> Event:
    ...

async def work_pool_status_event(event_id: UUID, occurred: DateTime, pre_update_work_pool: ORMWorkPool, work_pool: ORMWorkPool) -> Event:
    ...

def _get_recent_preceding_work_pool_event_id(work_pool: ORMWorkPool) -> Optional[UUID]:
    ...
