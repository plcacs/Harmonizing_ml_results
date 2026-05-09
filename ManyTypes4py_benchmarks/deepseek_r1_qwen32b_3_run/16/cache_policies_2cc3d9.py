import inspect
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Literal, Optional, Union
from typing_extensions import Self
from prefect.context import TaskRunContext
from prefect.exceptions import HashError
from prefect.utilities.hashing import hash_objects

if TYPE_CHECKING:
    from prefect.filesystems import WritableFileSystem
    from prefect.locking.protocol import LockManager
    from prefect.transactions import IsolationLevel

@dataclass
class CachePolicy:
    key_storage = None
    isolation_level = None
    lock_manager = None

    @classmethod
    def from_cache_key_fn(cls, cache_key_fn: Callable[[TaskRunContext, Dict[str, Any]], str]) -> 'CacheKeyFnPolicy':
        return CacheKeyFnPolicy(cache_key_fn=cache_key_fn)

    def configure(self, key_storage=None, lock_manager=None, isolation_level=None) -> Self:
        new = deepcopy(self)
        if key_storage is not None:
            new.key_storage = key_storage
        if lock_manager is not None:
            new.lock_manager = lock_manager
        if isolation_level is not None:
            new.isolation_level = isolation_level
        return new

    def compute_key(self, task_ctx: TaskRunContext, inputs: Dict[str, Any], flow_parameters: Dict[str, Any], **kwargs) -> Optional[str]:
        raise NotImplementedError

    def __sub__(self, other: str) -> Self:
        if not isinstance(other, str):
            raise TypeError('Can only subtract strings from key policies.')
        return self

    def __add__(self, other: 'CachePolicy') -> Self:
        if isinstance(other, _None):
            return self
        if other.key_storage is not None and self.key_storage is not None and (other.key_storage != self.key_storage):
            raise ValueError('Cannot add CachePolicies with different storage locations.')
        if other.isolation_level is not None and self.isolation_level is not None and (other.isolation_level != self.isolation_level):
            raise ValueError('Cannot add CachePolicies with different isolation levels.')
        if other.lock_manager is not None and self.lock_manager is not None and (other.lock_manager != self.lock_manager):
            raise ValueError('Cannot add CachePolicies with different lock implementations.')
        return CompoundCachePolicy(policies=[self, other], key_storage=self.key_storage or other.key_storage, isolation_level=self.isolation_level or other.isolation_level, lock_manager=self.lock_manager or other.lock_manager)

@dataclass
class CacheKeyFnPolicy(CachePolicy):
    cache_key_fn: Optional[Callable[[TaskRunContext, Dict[str, Any]], str]] = None

    def compute_key(self, task_ctx: TaskRunContext, inputs: Dict[str, Any], flow_parameters: Dict[str, Any], **kwargs) -> Optional[str]:
        if self.cache_key_fn:
            return self.cache_key_fn(task_ctx, inputs)

@dataclass
class CompoundCachePolicy(CachePolicy):
    policies: list['CachePolicy'] = field(default_factory=list)

    def __post_init__(self):
        self.policies = [policy for p in self.policies for policy in (p.policies if isinstance(p, CompoundCachePolicy) else [p])]
        inputs_policies = [p for p in self.policies if isinstance(p, Inputs)]
        self.policies = [p for p in self.policies if not isinstance(p, Inputs)]
        if inputs_policies:
            all_excludes = set()
            for inputs_policy in inputs_policies:
                all_excludes.update(inputs_policy.exclude)
            self.policies.append(Inputs(exclude=sorted(all_excludes)))

    def compute_key(self, task_ctx: TaskRunContext, inputs: Dict[str, Any], flow_parameters: Dict[str, Any], **kwargs) -> Optional[str]:
        keys = []
        for policy in self.policies:
            policy_key = policy.compute_key(task_ctx=task_ctx, inputs=inputs, flow_parameters=flow_parameters, **kwargs)
            if policy_key is not None:
                keys.append(policy_key)
        if not keys:
            return None
        return hash_objects(*keys, raise_on_failure=True)

    def __add__(self, other: 'CachePolicy') -> Self:
        super().__add__(other)
        if isinstance(other, CompoundCachePolicy):
            policies = [*self.policies, *other.policies]
        else:
            policies = [*self.policies, other]
        return CompoundCachePolicy(policies=policies, key_storage=self.key_storage or other.key_storage, isolation_level=self.isolation_level or other.isolation_level, lock_manager=self.lock_manager or other.lock_manager)

    def __sub__(self, other: str) -> Self:
        if not isinstance(other, str):
            raise TypeError('Can only subtract strings from key policies.')
        inputs_policies = [p for p in self.policies if isinstance(p, Inputs)]
        if inputs_policies:
            new = Inputs(exclude=[other])
            return CompoundCachePolicy(policies=[*self.policies, new])
        else:
            return self

@dataclass
class _None(CachePolicy):
    def compute_key(self, task_ctx: TaskRunContext, inputs: Dict[str, Any], flow_parameters: Dict[str, Any], **kwargs) -> None:
        return None

    def __add__(self, other: 'CachePolicy') -> 'CachePolicy':
        return other

@dataclass
class TaskSource(CachePolicy):
    def compute_key(self, task_ctx: TaskRunContext, inputs: Dict[str, Any], flow_parameters: Dict[str, Any], **kwargs) -> Optional[str]:
        if not task_ctx:
            return None
        try:
            lines = inspect.getsource(task_ctx.task)
        except TypeError:
            lines = inspect.getsource(task_ctx.task.fn.__class__)
        except OSError as exc:
            if 'source code' in str(exc):
                lines = task_ctx.task.fn.__code__.co_code
            else:
                raise
        return hash_objects(lines, raise_on_failure=True)

@dataclass
class FlowParameters(CachePolicy):
    def compute_key(self, task_ctx: TaskRunContext, inputs: Dict[str, Any], flow_parameters: Dict[str, Any], **kwargs) -> Optional[str]:
        if not flow_parameters:
            return None
        return hash_objects(flow_parameters, raise_on_failure=True)

@dataclass
class RunId(CachePolicy):
    def compute_key(self, task_ctx: TaskRunContext, inputs: Dict[str, Any], flow_parameters: Dict[str, Any], **kwargs) -> Optional[str]:
        if not task_ctx:
            return None
        run_id = task_ctx.task_run.flow_run_id
        if run_id is None:
            run_id = task_ctx.task_run.id
        return str(run_id)

@dataclass
class Inputs(CachePolicy):
    exclude: list[str] = field(default_factory=list)

    def compute_key(self, task_ctx: TaskRunContext, inputs: Dict[str, Any], flow_parameters: Dict[str, Any], **kwargs) -> Optional[str]:
        hashed_inputs = {}
        inputs = inputs or {}
        exclude = self.exclude or []
        if not inputs:
            return None
        for key, val in inputs.items():
            if key not in exclude:
                hashed_inputs[key] = val
        try:
            return hash_objects(hashed_inputs, raise_on_failure=True)
        except HashError as exc:
            msg = f'{exc}\n\nThis often occurs when task inputs contain objects that cannot be cached like locks, file handles, or other system resources.\n\nTo resolve this, you can:\n  1. Exclude these arguments by defining a custom `cache_key_fn`\n  2. Disable caching by passing `cache_policy=NO_CACHE`\n'
            raise ValueError(msg) from exc

    def __sub__(self, other: str) -> 'Inputs':
        if not isinstance(other, str):
            raise TypeError('Can only subtract strings from key policies.')
        return Inputs(exclude=self.exclude + [other])

INPUTS = Inputs()
NONE = _None()
NO_CACHE = _None()
TASK_SOURCE = TaskSource()
FLOW_PARAMETERS = FlowParameters()
RUN_ID = RunId()
DEFAULT = INPUTS + TASK_SOURCE + RUN_ID