```python
import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from typing_extensions import Self

if TYPE_CHECKING:
    from prefect.context import TaskRunContext
    from prefect.filesystems import WritableFileSystem
    from prefect.locking.protocol import LockManager
    from prefect.transactions import IsolationLevel

@dataclass
class CachePolicy:
    key_storage: Any = ...
    isolation_level: Any = ...
    lock_manager: Any = ...

    @classmethod
    def from_cache_key_fn(cls, cache_key_fn: Callable[..., Any]) -> "CachePolicy": ...
    
    def configure(
        self,
        key_storage: Optional[Any] = ...,
        lock_manager: Optional[Any] = ...,
        isolation_level: Optional[Any] = ...
    ) -> Self: ...
    
    def compute_key(
        self,
        task_ctx: Optional["TaskRunContext"],
        inputs: Optional[Dict[str, Any]],
        flow_parameters: Optional[Dict[str, Any]],
        **kwargs: Any
    ) -> Optional[str]: ...
    
    def __sub__(self, other: Any) -> "CachePolicy": ...
    
    def __add__(self, other: "CachePolicy") -> "CachePolicy": ...

@dataclass
class CacheKeyFnPolicy(CachePolicy):
    cache_key_fn: Optional[Callable[..., Any]] = ...
    
    def compute_key(
        self,
        task_ctx: Optional["TaskRunContext"],
        inputs: Optional[Dict[str, Any]],
        flow_parameters: Optional[Dict[str, Any]],
        **kwargs: Any
    ) -> Optional[str]: ...

@dataclass
class CompoundCachePolicy(CachePolicy):
    policies: List[CachePolicy] = ...
    
    def __post_init__(self) -> None: ...
    
    def compute_key(
        self,
        task_ctx: Optional["TaskRunContext"],
        inputs: Optional[Dict[str, Any]],
        flow_parameters: Optional[Dict[str, Any]],
        **kwargs: Any
    ) -> Optional[str]: ...
    
    def __add__(self, other: "CachePolicy") -> "CompoundCachePolicy": ...
    
    def __sub__(self, other: str) -> "CompoundCachePolicy": ...

@dataclass
class _None(CachePolicy):
    def compute_key(
        self,
        task_ctx: Optional["TaskRunContext"],
        inputs: Optional[Dict[str, Any]],
        flow_parameters: Optional[Dict[str, Any]],
        **kwargs: Any
    ) -> None: ...
    
    def __add__(self, other: "CachePolicy") -> "CachePolicy": ...

@dataclass
class TaskSource(CachePolicy):
    def compute_key(
        self,
        task_ctx: Optional["TaskRunContext"],
        inputs: Optional[Dict[str, Any]],
        flow_parameters: Optional[Dict[str, Any]],
        **kwargs: Any
    ) -> Optional[str]: ...

@dataclass
class FlowParameters(CachePolicy):
    def compute_key(
        self,
        task_ctx: Optional["TaskRunContext"],
        inputs: Optional[Dict[str, Any]],
        flow_parameters: Optional[Dict[str, Any]],
        **kwargs: Any
    ) -> Optional[str]: ...

@dataclass
class RunId(CachePolicy):
    def compute_key(
        self,
        task_ctx: Optional["TaskRunContext"],
        inputs: Optional[Dict[str, Any]],
        flow_parameters: Optional[Dict[str, Any]],
        **kwargs: Any
    ) -> Optional[str]: ...

@dataclass
class Inputs(CachePolicy):
    exclude: List[str] = ...
    
    def compute_key(
        self,
        task_ctx: Optional["TaskRunContext"],
        inputs: Optional[Dict[str, Any]],
        flow_parameters: Optional[Dict[str, Any]],
        **kwargs: Any
    ) -> Optional[str]: ...
    
    def __sub__(self, other: str) -> "Inputs": ...

INPUTS: Inputs = ...
NONE: _None = ...
NO_CACHE: _None = ...
TASK_SOURCE: TaskSource = ...
FLOW_PARAMETERS: FlowParameters = ...
RUN_ID: RunId = ...
DEFAULT: CompoundCachePolicy = ...
```