from typing import Any, Callable, Dict, Literal, Optional, Union

class CachePolicy:
    key_storage: Any = None
    isolation_level: Any = None
    lock_manager: Any = None

    @classmethod
    def from_cache_key_fn(cls, cache_key_fn: Callable) -> 'CacheKeyFnPolicy':
        ...

    def configure(self, key_storage: Any = None, lock_manager: Any = None, isolation_level: Any = None) -> 'CachePolicy':
        ...

    def compute_key(self, task_ctx: TaskRunContext, inputs: Dict[str, Any], flow_parameters: Dict[str, Any], **kwargs: Any) -> Optional[str]:
        ...

    def __sub__(self, other: str) -> 'CachePolicy':
        ...

    def __add__(self, other: Union['_None', 'CachePolicy']) -> Union['_None', 'CompoundCachePolicy']:
        ...

class CacheKeyFnPolicy(CachePolicy):
    cache_key_fn: Callable = None

    def compute_key(self, task_ctx: TaskRunContext, inputs: Dict[str, Any], flow_parameters: Dict[str, Any], **kwargs: Any) -> Optional[str]:
        ...

class CompoundCachePolicy(CachePolicy):
    policies: List[CachePolicy] = []

    def __post_init__(self) -> None:
        ...

    def compute_key(self, task_ctx: TaskRunContext, inputs: Dict[str, Any], flow_parameters: Dict[str, Any], **kwargs: Any) -> Optional[str]:
        ...

    def __add__(self, other: Union['CompoundCachePolicy', 'CachePolicy']) -> 'CompoundCachePolicy':
        ...

    def __sub__(self, other: str) -> 'CompoundCachePolicy':
        ...

class _None(CachePolicy):
    def compute_key(self, task_ctx: TaskRunContext, inputs: Dict[str, Any], flow_parameters: Dict[str, Any], **kwargs: Any) -> None:
        ...

    def __add__(self, other: Union['_None', 'CachePolicy']) -> Union['_None', 'CachePolicy']:
        ...

class TaskSource(CachePolicy):
    def compute_key(self, task_ctx: TaskRunContext, inputs: Dict[str, Any], flow_parameters: Dict[str, Any], **kwargs: Any) -> Optional[str]:
        ...

class FlowParameters(CachePolicy):
    def compute_key(self, task_ctx: TaskRunContext, inputs: Dict[str, Any], flow_parameters: Dict[str, Any], **kwargs: Any) -> Optional[str]:
        ...

class RunId(CachePolicy):
    def compute_key(self, task_ctx: TaskRunContext, inputs: Dict[str, Any], flow_parameters: Dict[str, Any], **kwargs: Any) -> Optional[str]:
        ...

class Inputs(CachePolicy):
    exclude: List[str] = []

    def compute_key(self, task_ctx: TaskRunContext, inputs: Dict[str, Any], flow_parameters: Dict[str, Any], **kwargs: Any) -> Optional[str]:
        ...

    def __sub__(self, other: str) -> 'Inputs':
        ...

INPUTS: Inputs = Inputs()
NONE: _None = _None()
NO_CACHE: _None = _None()
TASK_SOURCE: TaskSource = TaskSource()
FLOW_PARAMETERS: FlowParameters = FlowParameters()
RUN_ID: RunId = RunId()
DEFAULT: CompoundCachePolicy = INPUTS + TASK_SOURCE + RUN_ID
