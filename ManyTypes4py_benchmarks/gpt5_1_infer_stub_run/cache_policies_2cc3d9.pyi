from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Union
from typing_extensions import Self
from prefect.context import TaskRunContext
from prefect.filesystems import WritableFileSystem
from prefect.locking.protocol import LockManager
from prefect.transactions import IsolationLevel

@dataclass
class CachePolicy:
    """
    Base class for all cache policies.
    """
    key_storage: Optional[Union[Path, WritableFileSystem]] = ...
    isolation_level: Optional[IsolationLevel] = ...
    lock_manager: Optional[LockManager] = ...

    @classmethod
    def from_cache_key_fn(cls, cache_key_fn: Callable[..., Optional[str]]) -> "CacheKeyFnPolicy": ...
    def configure(
        self,
        key_storage: Optional[Union[Path, WritableFileSystem]] = ...,
        lock_manager: Optional[LockManager] = ...,
        isolation_level: Optional[IsolationLevel] = ...,
    ) -> Self: ...
    def compute_key(
        self,
        task_ctx: Optional[TaskRunContext],
        inputs: Optional[dict[str, Any]],
        flow_parameters: Optional[dict[str, Any]],
        **kwargs: Any,
    ) -> Optional[str]: ...
    def __sub__(self, other: str) -> "CachePolicy": ...
    def __add__(self, other: "CachePolicy") -> "CachePolicy": ...

@dataclass
class CacheKeyFnPolicy(CachePolicy):
    """
    This policy accepts a custom function with signature f(task_run_context, task_parameters, flow_parameters) -> str
    and uses it to compute a task run cache key.
    """
    cache_key_fn: Optional[Callable[..., Optional[str]]] = ...

    def compute_key(
        self,
        task_ctx: Optional[TaskRunContext],
        inputs: Optional[dict[str, Any]],
        flow_parameters: Optional[dict[str, Any]],
        **kwargs: Any,
    ) -> Optional[str]: ...

@dataclass
class CompoundCachePolicy(CachePolicy):
    """
    This policy is constructed from two or more other cache policies and works by computing the keys
    for each policy individually, and then hashing a sorted tuple of all computed keys.

    Any keys that return `None` will be ignored.
    """
    policies: list[CachePolicy] = ...

    def __post_init__(self) -> None: ...
    def compute_key(
        self,
        task_ctx: Optional[TaskRunContext],
        inputs: Optional[dict[str, Any]],
        flow_parameters: Optional[dict[str, Any]],
        **kwargs: Any,
    ) -> Optional[str]: ...
    def __add__(self, other: CachePolicy) -> "CompoundCachePolicy": ...
    def __sub__(self, other: str) -> "CompoundCachePolicy": ...

@dataclass
class _None(CachePolicy):
    """
    Policy that always returns `None` for the computed cache key.
    This policy prevents persistence and avoids caching entirely.
    """
    def compute_key(
        self,
        task_ctx: Optional[TaskRunContext],
        inputs: Optional[dict[str, Any]],
        flow_parameters: Optional[dict[str, Any]],
        **kwargs: Any,
    ) -> Optional[str]: ...
    def __add__(self, other: CachePolicy) -> CachePolicy: ...

@dataclass
class TaskSource(CachePolicy):
    """
    Policy for computing a cache key based on the source code of the task.

    This policy only considers raw lines of code in the task, and not the source code of nested tasks.
    """
    def compute_key(
        self,
        task_ctx: Optional[TaskRunContext],
        inputs: Optional[dict[str, Any]],
        flow_parameters: Optional[dict[str, Any]],
        **kwargs: Any,
    ) -> Optional[str]: ...

@dataclass
class FlowParameters(CachePolicy):
    """
    Policy that computes the cache key based on a hash of the flow parameters.
    """
    def compute_key(
        self,
        task_ctx: Optional[TaskRunContext],
        inputs: Optional[dict[str, Any]],
        flow_parameters: Optional[dict[str, Any]],
        **kwargs: Any,
    ) -> Optional[str]: ...

@dataclass
class RunId(CachePolicy):
    """
    Returns either the prevailing flow run ID, or if not found, the prevailing task
    run ID.
    """
    def compute_key(
        self,
        task_ctx: Optional[TaskRunContext],
        inputs: Optional[dict[str, Any]],
        flow_parameters: Optional[dict[str, Any]],
        **kwargs: Any,
    ) -> Optional[str]: ...

@dataclass
class Inputs(CachePolicy):
    """
    Policy that computes a cache key based on a hash of the runtime inputs provided to the task..
    """
    exclude: list[str] = ...

    def compute_key(
        self,
        task_ctx: Optional[TaskRunContext],
        inputs: Optional[dict[str, Any]],
        flow_parameters: Optional[dict[str, Any]],
        **kwargs: Any,
    ) -> Optional[str]: ...
    def __sub__(self, other: str) -> "Inputs": ...

INPUTS: Inputs = ...
NONE: _None = ...
NO_CACHE: _None = ...
TASK_SOURCE: TaskSource = ...
FLOW_PARAMETERS: FlowParameters = ...
RUN_ID: RunId = ...
DEFAULT: CompoundCachePolicy = ...