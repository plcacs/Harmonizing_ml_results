from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Union
import datetime

class SetStateStatus(BaseModel):
    ACCEPT: str
    REJECT: str
    ABORT: str
    WAIT: str

class StateAcceptDetails(BaseModel):
    type: str

class StateRejectDetails(BaseModel):
    type: str
    reason: Optional[str]

class StateAbortDetails(BaseModel):
    type: str
    reason: Optional[str]

class StateWaitDetails(BaseModel):
    type: str
    delay_seconds: Any
    reason: Optional[str]

class HistoryResponseState(BaseModel):
    state_type: Any
    state_name: Any
    count_runs: Any
    sum_estimated_run_time: Any
    sum_estimated_lateness: Any

class HistoryResponse(BaseModel):
    interval_start: Any
    interval_end: Any
    states: Any

class OrchestrationResult(BaseModel):
    pass

class WorkerFlowRunResponse(BaseModel):
    model_config: Dict[str, Any]

class FlowRunResponse(BaseModel):
    name: str
    flow_id: Any
    state_id: Optional[Any]
    deployment_id: Optional[Any]
    deployment_version: Optional[str]
    work_queue_id: Optional[Any]
    work_queue_name: Optional[str]
    flow_version: Optional[str]
    parameters: Dict[str, Any]
    idempotency_key: Optional[str]
    context: Dict[str, Any]
    empirical_policy: Any
    tags: List[str]
    parent_task_run_id: Optional[Any]
    state_type: Optional[Any]
    state_name: Optional[Any]
    run_count: int
    expected_start_time: Optional[Any]
    next_scheduled_start_time: Optional[Any]
    start_time: Optional[Any]
    end_time: Optional[Any]
    total_run_time: datetime.timedelta
    estimated_run_time: datetime.timedelta
    estimated_start_time_delta: datetime.timedelta
    auto_scheduled: bool
    infrastructure_document_id: Optional[Any]
    infrastructure_pid: Optional[Any]
    created_by: Optional[Any]
    work_pool_id: Optional[Any]
    work_pool_name: Optional[str]
    state: Optional[Any]
    job_variables: Optional[Any]

class DeploymentResponse(BaseModel):
    name: Any
    version: Optional[str]
    description: Optional[str]
    flow_id: Any
    paused: bool
    schedules: List[Any]
    concurrency_limit: Optional[Any]
    global_concurrency_limit: Optional[Any]
    concurrency_options: Optional[Any]
    job_variables: Dict[str, Any]
    parameters: Dict[str, Any]
    tags: List[str]
    work_queue_name: Optional[str]
    last_polled: Optional[Any]
    parameter_openapi_schema: Optional[Any]
    path: Optional[Any]
    pull_steps: Optional[Any]
    entrypoint: Optional[Any]
    storage_document_id: Optional[Any]
    infrastructure_document_id: Optional[Any]
    created_by: Optional[Any]
    updated_by: Optional[Any]
    work_pool_name: Optional[str]
    status: Any
    enforce_parameter_schema: bool

class WorkQueueResponse(BaseModel):
    work_pool_name: Optional[str]
    status: Optional[Any]

class WorkQueueWithStatus(BaseModel):
    pass

class WorkerResponse(BaseModel):
    status: Any

class GlobalConcurrencyLimitResponse(BaseModel):
    active: bool
    name: Any
    limit: Any
    active_slots: Any
    slot_decay_per_second: float

class FlowPaginationResponse(BaseModel):
    pass

class FlowRunPaginationResponse(BaseModel):
    pass

class DeploymentPaginationResponse(BaseModel):
    pass

class SchemaValuePropertyError(BaseModel):
    pass

class SchemaValueIndexError(BaseModel):
    pass

class SchemaValuesValidationResponse(BaseModel):
    pass
