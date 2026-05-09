import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union, overload
from typing_extensions import Self
from prefect.context import TaskRunContext
from prefect.filesystems import WritableFileSystem
from prefect.locking.protocol import LockManager
from prefect.transactions import IsolationLevel

@dataclass
class CachePolicy:
    key_storage: Optional[WritableFileSystem]
    isolation_level: Optional[IsolationLevel]
    lock_manager: Optional[LockManager]

    @classmethod
    def from_cache_key_fn(cls, cache_key_fn: Callable[[TaskRunContext, Any, Any], str]) -> 'CacheKeyFnPolicy': ...

    def configure(
        self,
        key_storage: Optional[WritableFileSystem] = ...,
        lock_manager: Optional[LockManager] = ...,
        isolation_level: Optional[IsolationLevel] = ...,
    ) -> Self: ...

    def compute_key(
        self,
        task_ctx: Optional[TaskRunContext],
        inputs: Optional[Dict[str, Any]],
        flow_parameters: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> Optional[str]: ...

    def __sub__(self, other: str) -> Self: ...

    def __add__(self, other: Union['CachePolicy', '_None']) -> 'CompoundCachePolicy': ...

@dataclass
class CacheKeyFnPolicy(CachePolicy):
    cache_key_fn: Optional[Callable[[TaskRunContext, Any, Any], str]]

    def compute_key(
        self,
        task_ctx: Optional[TaskRunContext],
        inputs: Optional[Dict[str, Any]],
        flow_parameters: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> Optional[str]: ...

@dataclass
class CompoundCachePolicy(CachePolicy):
    policies: List[CachePolicy]

    def __post_init__(self) -> None: ...

    def compute_key(
        self,
        task_ctx: Optional[TaskRunContext],
        inputs: Optional[Dict[str, Any]],
        flow_parameters: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> Optional[str]: ...

    def __add__(self, other: Union[CachePolicy, 'CompoundCachePolicy']) -> 'CompoundCachePolicy': ...

    def __sub__(self, other: str) -> 'CompoundCachePolicy': ...

@dataclass
class _None(CachePolicy):
    def compute_key(
        self,
        task_ctx: Optional[TaskRunContext],
        inputs: Optional[Dict[str, Any]],
        flow_parameters: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> None: ...

    def __add__(self, other: CachePolicy) -> CachePolicy: ...

@dataclass
class TaskSource(CachePolicy):
    def compute_key(
        self,
        task_ctx: Optional[TaskRunContext],
        inputs: Optional[Dict[str, Any]],
        flow_parameters: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> Optional[str]: ...

@dataclass
class FlowParameters(CachePolicy):
    def compute_key(
        self,
        task_ctx: Optional[TaskRunContext],
        inputs: Optional[Dict[str, Any]],
        flow_parameters: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> Optional[str]: ...

@dataclass
class RunId(CachePolicy):
    def compute_key(
        self,
        task_ctx: Optional[TaskRunContext],
        inputs: Optional[Dict[str, Any]],
        flow_parameters: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> Optional[str]: ...

@dataclass
class Inputs(CachePolicy):
    exclude: List[str]

    def compute_key(
        self,
        task_ctx: Optional[TaskRunContext],
        inputs: Optional[Dict[str, Any]],
        flow_parameters: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> Optional[str]: ...

    def __sub__(self, other: str) -> 'Inputs': ...

INPUTS: Inputs = ...
NONE: _None = ...
NO_CACHE: _None = ...
TASK_SOURCE: TaskSource = ...
FLOW_PARAMETERS: FlowParameters = ...
RUN_ID: RunId = ...
DEFAULT: CompoundCachePolicy = ...