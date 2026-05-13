from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from prefect.filesystems import WritableFileSystem
    from prefect.locking.protocol import LockManager
    from prefect.transactions import IsolationLevel
    from prefect.context import TaskRunContext

class CachePolicy(ABC):
    key_storage: Optional['WritableFileSystem']
    isolation_level: Optional['IsolationLevel']
    lock_manager: Optional['LockManager']

    @classmethod
    def from_cache_key_fn(cls, cache_key_fn: Callable[..., str]) -> 'CacheKeyFnPolicy': ...

    def configure(
        self,
        key_storage: Optional['WritableFileSystem'] = None,
        lock_manager: Optional['LockManager'] = None,
        isolation_level: Optional['IsolationLevel'] = None,
    ) -> 'CachePolicy': ...

    @abstractmethod
    def compute_key(
        self,
        task_ctx: Optional['TaskRunContext'],
        inputs: Optional[Dict[str, Any]],
        flow_parameters: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> Optional[str]: ...

    def __sub__(self, other: str) -> 'CachePolicy': ...

    def __add__(self, other: 'CachePolicy') -> 'CachePolicy': ...

class CacheKeyFnPolicy(CachePolicy):
    cache_key_fn: Optional[Callable[..., str]]

    def compute_key(
        self,
        task_ctx: Optional['TaskRunContext'],
        inputs: Optional[Dict[str, Any]],
        flow_parameters: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> Optional[str]: ...

class CompoundCachePolicy(CachePolicy):
    policies: List['CachePolicy']

    def __post_init__(self) -> None: ...

    def compute_key(
        self,
        task_ctx: Optional['TaskRunContext'],
        inputs: Optional[Dict[str, Any]],
        flow_parameters: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> Optional[str]: ...

    def __add__(self, other: 'CachePolicy') -> 'CachePolicy': ...

    def __sub__(self, other: str) -> 'CachePolicy': ...

class _None(CachePolicy):
    def compute_key(
        self,
        task_ctx: Optional['TaskRunContext'],
        inputs: Optional[Dict[str, Any]],
        flow_parameters: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> None: ...

    def __add__(self, other: 'CachePolicy') -> 'CachePolicy': ...

class TaskSource(CachePolicy):
    def compute_key(
        self,
        task_ctx: Optional['TaskRunContext'],
        inputs: Optional[Dict[str, Any]],
        flow_parameters: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> Optional[str]: ...

class FlowParameters(CachePolicy):
    def compute_key(
        self,
        task_ctx: Optional['TaskRunContext'],
        inputs: Optional[Dict[str, Any]],
        flow_parameters: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> Optional[str]: ...

class RunId(CachePolicy):
    def compute_key(
        self,
        task_ctx: Optional['TaskRunContext'],
        inputs: Optional[Dict[str, Any]],
        flow_parameters: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> Optional[str]: ...

class Inputs(CachePolicy):
    exclude: List[str]

    def compute_key(
        self,
        task_ctx: Optional['TaskRunContext'],
        inputs: Optional[Dict[str, Any]],
        flow_parameters: Optional[Dict[str, Any]],
        **kwargs: Any,
    ) -> Optional[str]: ...

    def __sub__(self, other: str) -> 'Inputs': ...

INPUTS: Inputs
NONE: _None
NO_CACHE: _None
TASK_SOURCE: TaskSource
FLOW_PARAMETERS: FlowParameters
RUN_ID: RunId
DEFAULT: CachePolicy