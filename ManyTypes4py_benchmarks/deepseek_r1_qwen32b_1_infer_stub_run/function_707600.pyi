from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union
from parso.python.tree import Node
from jedi import debug
from jedi.inference import flow_analysis
from jedi.inference.base_value import ContextualizedNode, ValueSet
from jedi.inference.signature import TreeSignature
from jedi.inference.value import iterable
from jedi.inference.context import ValueContext


class LambdaName:
    string_name: str
    api_type: str

    def __init__(self, lambda_value: Any) -> None:
        ...

    @property
    def start_pos(self) -> Any:
        ...

    def infer(self) -> ValueSet:
        ...


class FunctionAndClassBase:
    def get_qualified_names(self) -> Optional[Tuple[str]]:
        ...


class FunctionMixin:
    api_type: str

    def get_filters(self, origin_scope: Any = None) -> Any:
        ...

    def py__get__(self, instance: Any, class_value: Any) -> ValueSet:
        ...

    def get_param_names(self) -> List[Any]:
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

    def py__call__(self, arguments: Any) -> ValueSet:
        ...

    def _as_context(self, arguments: Any = None) -> Union[AnonymousFunctionExecution, FunctionExecutionContext]:
        ...

    def get_signatures(self) -> List[TreeSignature]:
        ...


class FunctionValue(FunctionMixin, FunctionAndClassBase):
    @classmethod
    def from_context(cls, context: Any, tree_node: Node) -> Union[FunctionValue, OverloadedFunctionValue]:
        ...

    def py__class__(self) -> Any:
        ...

    def get_default_param_context(self) -> Any:
        ...

    def get_signature_functions(self) -> List[Any]:
        ...


class FunctionNameInClass:
    def __init__(self, class_context: Any, name: Any) -> None:
        ...

    def get_defining_qualified_value(self) -> Any:
        ...


class MethodValue(FunctionValue):
    def __init__(self, inference_state: Any, class_context: Any, *args: Any, **kwargs: Any) -> None:
        ...

    def get_default_param_context(self) -> Any:
        ...

    def get_qualified_names(self) -> Optional[Tuple[str]]:
        ...

    @property
    def name(self) -> FunctionNameInClass:
        ...


class BaseFunctionExecutionContext(ValueContext):
    def infer_annotations(self) -> ValueSet:
        ...

    def get_return_values(self, check_yields: bool = False) -> ValueSet:
        ...

    def _get_yield_lazy_value(self, yield_expr: Node) -> Any:
        ...

    def get_yield_lazy_values(self, is_async: bool = False) -> Any:
        ...

    def merge_yield_values(self, is_async: bool = False) -> ValueSet:
        ...

    def is_generator(self) -> bool:
        ...

    def infer(self) -> ValueSet:
        ...


class FunctionExecutionContext(BaseFunctionExecutionContext):
    def __init__(self, function_value: FunctionValue, arguments: Any) -> None:
        ...

    def get_filters(self, until_position: Any = None, origin_scope: Any = None) -> Any:
        ...

    def infer_annotations(self) -> ValueSet:
        ...

    def get_param_names(self) -> List[Any]:
        ...


class AnonymousFunctionExecution(BaseFunctionExecutionContext):
    def infer_annotations(self) -> ValueSet:
        ...

    def get_filters(self, until_position: Any = None, origin_scope: Any = None) -> Any:
        ...

    def get_param_names(self) -> List[Any]:
        ...


class OverloadedFunctionValue(FunctionMixin, ValueWrapper):
    def __init__(self, function: FunctionValue, overloaded_functions: List[FunctionValue]) -> None:
        ...

    def py__call__(self, arguments: Any) -> ValueSet:
        ...

    def get_signature_functions(self) -> List[FunctionValue]:
        ...

    def get_type_hint(self, add_class_info: bool = True) -> str:
        ...


def _find_overload_functions(context: Any, tree_node: Node) -> Iterator[Node]:
    ...