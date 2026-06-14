from typing import Any, Callable, Dict, List, Optional
from typing_extensions import Self
from prefect.context import TaskRunContext
from prefect.filesystems import WritableFileSystem
from prefect.locking.protocol import LockManager
from prefect.transactions import IsolationLevel

class CachePolicy:
    key_storage: Optional[WritableFileSystem]
    isolation_level: Optional[IsolationLevel]
    lock_manager: Optional[LockManager]
    @classmethod
    def from_cache_key_fn(cls, cache_key_fn: Callable[[Optional[TaskRunContext], Dict[str, Any]], Optional[str]]) -> "CacheKeyFnPolicy": ...
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
    def __add__(self, other: "CachePolicy") -> "CachePolicy": ...

class CacheKeyFnPolicy(CachePolicy):
    cache_key_fn: Optional[Callable[[Optional[TaskRunContext], Dict[str, Any]], Optional[str]]]
    def compute_key(
        self,
        task_ctx: Optional[TaskRunContext],
        inputs: Optional[Dict[str, Any]],
        flow_parameters: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> Optional[str]: ...

class CompoundCachePolicy(CachePolicy):
    policies: List[CachePolicy]
    def compute_key(
        self,
        task_ctx: Optional[TaskRunContext],
        inputs: Optional[Dict[str, Any]],
        flow_parameters: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> Optional[str]: ...
    def __add__(self, other: "CachePolicy") -> "CompoundCachePolicy": ...
    def __sub__(self, other: str) -> "CompoundCachePolicy": ...

class _None(CachePolicy):
    def compute_key(
        self,
        task_ctx: Optional[TaskRunContext],
        inputs: Optional[Dict[str, Any]],
        flow_parameters: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> Optional[str]: ...
    def __add__(self, other: "CachePolicy") -> "CachePolicy": ...

class TaskSource(CachePolicy):
    def compute_key(
        self,
        task_ctx: Optional[TaskRunContext],
        inputs: Optional[Dict[str, Any]],
        flow_parameters: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> Optional[str]: ...

class FlowParameters(CachePolicy):
    def compute_key(
        self,
        task_ctx: Optional[TaskRunContext],
        inputs: Optional[Dict[str, Any]],
        flow_parameters: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> Optional[str]: ...

class RunId(CachePolicy):
    def compute_key(
        self,
        task_ctx: Optional[TaskRunContext],
        inputs: Optional[Dict[str, Any]],
        flow_parameters: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> Optional[str]: ...

class Inputs(CachePolicy):
    exclude: List[str]
    def compute_key(
        self,
        task_ctx: Optional[TaskRunContext],
        inputs: Optional[Dict[str, Any]],
        flow_parameters: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> Optional[str]: ...
    def __sub__(self, other: str) -> "Inputs": ...

INPUTS: Inputs
NONE: _None
NO_CACHE: _None
TASK_SOURCE: TaskSource
FLOW_PARAMETERS: FlowParameters
RUN_ID: RunId
DEFAULT: CompoundCachePolicy