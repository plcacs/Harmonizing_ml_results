import inspect
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Union
from typing_extensions import Self
from prefect.context import TaskRunContext
from prefect.exceptions import HashError
from prefect.utilities.hashing import HashError

if TYPE_CHECKING:
    from prefect.filesystems import WritableFileSystem
    from prefect.locking.protocol import LockManager
    from prefect.transactions import IsolationLevel

@dataclass
class CachePolicy:
    key_storage: Any = ...
    isolation_level: Optional[IsolationLevel] = ...
    lock_manager: Optional[LockManager] = ...

    @classmethod
    def from_cache_key_fn(cls, cache_key_fn: Callable) -> 'CachePolicy':
        ...

    def configure(self, key_storage: Any = None, lock_manager: Any = None, isolation_level: Any = None) -> Self:
        ...

    def compute_key(self, task_ctx: Optional[TaskRunContext], inputs: Dict[str, Any], flow_parameters: Dict[str, Any], **kwargs: Any) -> Optional[str]:
        ...

    def __sub__(self, other: str) -> Self:
        ...

    def __add__(self, other: Union['CachePolicy', '_None']) -> Union['CachePolicy', '_None']:
        ...

@dataclass
class CacheKeyFnPolicy(CachePolicy):
    cache_key_fn: Optional[Callable] = ...

    def compute_key(self, task_ctx: Optional[TaskRunContext], inputs: Dict[str, Any], flow_parameters: Dict[str, Any], **kwargs: Any) -> Optional[str]:
        ...

@dataclass
class CompoundCachePolicy(CachePolicy):
    policies: List[CachePolicy] = field(default_factory=list)

    def __post_init__(self) -> None:
        ...

    def compute_key(self, task_ctx: Optional[TaskRunContext], inputs: Dict[str, Any], flow_parameters: Dict[str, Any], **kwargs: Any) -> str:
        ...

    def __add__(self, other: Union['CachePolicy', '_None']) -> 'CompoundCachePolicy':
        ...

    def __sub__(self, other: str) -> Self:
        ...

@dataclass
class _None(CachePolicy):
    def compute_key(self, task_ctx: Optional[TaskRunContext], inputs: Dict[str, Any], flow_parameters: Dict[str, Any], **kwargs: Any) -> None:
        ...

    def __add__(self, other: Union['CachePolicy', '_None']) -> Union['CachePolicy', '_None']:
        ...

@dataclass
class TaskSource(CachePolicy):
    def compute_key(self, task_ctx: Optional[TaskRunContext], inputs: Dict[str, Any], flow_parameters: Dict[str, Any], **kwargs: Any) -> Optional[str]:
        ...

@dataclass
class FlowParameters(CachePolicy):
    def compute_key(self, task_ctx: Optional[TaskRunContext], inputs: Dict[str, Any], flow_parameters: Dict[str, Any], **kwargs: Any) -> Optional[str]:
        ...

@dataclass
class RunId(CachePolicy):
    def compute_key(self, task_ctx: Optional[TaskRunContext], inputs: Dict[str, Any], flow_parameters: Dict[str, Any], **kwargs: Any) -> str:
        ...

@dataclass
class Inputs(CachePolicy):
    exclude: List[str] = field(default_factory=list)

    def compute_key(self, task_ctx: Optional[TaskRunContext], inputs: Dict[str, Any], flow_parameters: Dict[str, Any], **kwargs: Any) -> Optional[str]:
        ...

    def __sub__(self, other: str) -> Self:
        ...

INPUTS: Inputs = ...
NONE: _None = ...
NO_CACHE: _None = ...
TASK_SOURCE: TaskSource = ...
FLOW_PARAMETERS: FlowParameters = ...
RUN_ID: RunId = ...
DEFAULT: CompoundCachePolicy = ...