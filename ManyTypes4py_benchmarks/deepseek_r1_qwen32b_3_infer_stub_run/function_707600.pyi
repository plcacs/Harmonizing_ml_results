from typing import (
    Any,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
    Type,
    Dict,
    Set,
    FrozenSet,
    TypeVar,
    Callable,
    overload,
    TYPE_CHECKING,
)

from jedi.inference.base_value import ValueSet, TreeValue, NO_VALUES
from jedi.inference.signature import TreeSignature
from jedi.inference.filters import FunctionExecutionFilter, AnonymousFunctionExecutionFilter
from jedi.inference.names import ValueName, AbstractNameDefinition, AnonymousParamName, ParamName, NameWrapper
from jedi.inference.context import ValueContext, TreeContextMixin
from jedi.inference.value import iterable
from jedi.inference.lazy_value import LazyKnownValues, LazyKnownValue, LazyTreeValue
from jedi.inference.gradual.generics import TupleGenericManager

class LambdaName(AbstractNameDefinition):
    string_name: str
    api_type: str

    def __init__(self, lambda_value: TreeValue) -> None:
        ...

    @property
    def start_pos(self) -> Any:
        ...

    def infer(self) -> ValueSet[TreeValue]:
        ...

class FunctionAndClassBase(TreeValue):
    def get_qualified_names(self) -> Optional[Tuple[str, ...]]:
        ...

class FunctionMixin:
    api_type: str

    def get_filters(self, origin_scope: Any = None) -> Iterable[Any]:
        ...

    def py__get__(self, instance: Any, class_value: Any) -> ValueSet[Any]:
        ...

    def get_param_names(self) -> List[AnonymousParamName]:
        ...

    @property
    def name(self) -> Union[LambdaName, ValueName]:
        ...

    def is_function(self) -> bool:
        ...

    def py__name__(self) -> str:
        ...

    def get_type_hint(self, add_class_info: bool = True) -> str:
        ...

    def py__call__(self, arguments: Any) -> ValueSet[Any]:
        ...

    def _as_context(self, arguments: Optional[Any] = None) -> BaseFunctionExecutionContext:
        ...

    def get_signatures(self) -> List[TreeSignature]:
        ...

class FunctionValue(FunctionMixin, FunctionAndClassBase):
    @classmethod
    def from_context(cls, context: Any, tree_node: Any) -> Union['FunctionValue', 'OverloadedFunctionValue']:
        ...

    def py__class__(self) -> ValueSet[TreeValue]:
        ...

    def get_default_param_context(self) -> Any:
        ...

    def get_signature_functions(self) -> List[Any]:
        ...

class FunctionNameInClass(NameWrapper):
    def __init__(self, class_context: Any, name: Any) -> None:
        ...

    def get_defining_qualified_value(self) -> ValueSet[Any]:
        ...

class MethodValue(FunctionValue):
    def __init__(self, inference_state: Any, class_context: Any, *args: Any, **kwargs: Any) -> None:
        ...

    def get_default_param_context(self) -> Any:
        ...

    def get_qualified_names(self) -> Optional[Tuple[str, ...]]:
        ...

    @property
    def name(self) -> FunctionNameInClass:
        ...

class BaseFunctionExecutionContext(ValueContext, TreeContextMixin):
    def infer_annotations(self) -> ValueSet[Any]:
        ...

    @recursion.execution_recursion_decorator()
    def get_return_values(self, check_yields: bool = False) -> ValueSet[Any]:
        ...

    @recursion.execution_recursion_decorator(default=iter(()))
    def get_yield_lazy_values(self, is_async: bool = False) -> Iterable[Any]:
        ...

    def merge_yield_values(self, is_async: bool = False) -> ValueSet[Any]:
        ...

    def is_generator(self) -> bool:
        ...

    def infer(self) -> ValueSet[Any]:
        ...

class FunctionExecutionContext(BaseFunctionExecutionContext):
    def __init__(self, function_value: FunctionValue, arguments: Any) -> None:
        ...

    def get_filters(self, until_position: Any = None, origin_scope: Any = None) -> Generator[FunctionExecutionFilter, None, None]:
        ...

    def infer_annotations(self) -> ValueSet[Any]:
        ...

    def get_param_names(self) -> List[ParamName]:
        ...

class AnonymousFunctionExecution(BaseFunctionExecutionContext):
    def infer_annotations(self) -> ValueSet[Any]:
        ...

    def get_filters(self, until_position: Any = None, origin_scope: Any = None) -> Generator[AnonymousFunctionExecutionFilter, None, None]:
        ...

    def get_param_names(self) -> List[AnonymousParamName]:
        ...

class OverloadedFunctionValue(FunctionMixin, ValueWrapper):
    def __init__(self, function: Any, overloaded_functions: List[Any]) -> None:
        ...

    def py__call__(self, arguments: Any) -> ValueSet[Any]:
        ...

    def get_signature_functions(self) -> List[Any]:
        ...

    def get_type_hint(self, add_class_info: bool = True) -> str:
        ...