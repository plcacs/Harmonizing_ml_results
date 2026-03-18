```python
import inspect
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from typing_extensions import Self

if TYPE_CHECKING:
    from prefect.context import TaskRunContext
    from prefect.filesystems import WritableFileSystem
    from prefect.locking.protocol import LockManager
    from prefect.transactions import IsolationLevel

class CachePolicy:
    key_storage: Any = ...
    isolation_level: Any = ...
    lock_manager: Any = ...
    
    @classmethod
    def from_cache_key_fn(cls, cache_key_fn: Callable) -> "CachePolicy": ...
    
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

class CacheKeyFnPolicy(CachePolicy):
    cache_key_fn: Optional[Callable] = ...
    
    def compute_key(
        self,
        task_ctx: Optional["TaskRunContext"],
        inputs: Optional[Dict[str, Any]],
        flow_parameters: Optional[Dict[str, Any]],
        **kwargs: Any
    ) -> Optional[str]: ...

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
    def __sub__(self, other: Any) -> "CompoundCachePolicy": ...

class _None(CachePolicy):
    def compute_key(
        self,
        task_ctx: Optional["TaskRunContext"],
        inputs: Optional[Dict[str, Any]],
        flow_parameters: Optional[Dict[str, Any]],
        **kwargs: Any
    ) -> None: ...
    
    def __add__(self, other: "CachePolicy") -> "CachePolicy": ...

class TaskSource(CachePolicy):
    def compute_key(
        self,
        task_ctx: Optional["TaskRunContext"],
        inputs: Optional[Dict[str, Any]],
        flow_parameters: Optional[Dict[str, Any]],
        **kwargs: Any
    ) -> Optional[str]: ...

class FlowParameters(CachePolicy):
    def compute_key(
        self,
        task_ctx: Optional["TaskRunContext"],
        inputs: Optional[Dict[str, Any]],
        flow_parameters: Optional[Dict[str, Any]],
        **kwargs: Any
    ) -> Optional[str]: ...

class RunId(CachePolicy):
    def compute_key(
        self,
        task_ctx: Optional["TaskRunContext"],
        inputs: Optional[Dict[str, Any]],
        flow_parameters: Optional[Dict[str, Any]],
        **kwargs: Any
    ) -> Optional[str]: ...

class Inputs(CachePolicy):
    exclude: List[str] = ...
    
    def compute_key(
        self,
        task_ctx: Optional["TaskRunContext"],
        inputs: Optional[Dict[str, Any]],
        flow_parameters: Optional[Dict[str, Any]],
        **kwargs: Any
    ) -> Optional[str]: ...
    
    def __sub__(self, other: Any) -> "Inputs": ...

INPUTS: Inputs = ...
NONE: _None = ...
NO_CACHE: _None = ...
TASK_SOURCE: TaskSource = ...
FLOW_PARAMETERS: FlowParameters = ...
RUN_ID: RunId = ...
DEFAULT: CompoundCachePolicy = ...
```