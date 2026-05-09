from jedi.inference.base_value import ValueSet, TreeValue, LazyKnownValue
from jedi.inference.context import Context
from jedi.inference.signature import TreeSignature
from jedi.inference.value import BoundMethod
from jedi.inference.names import ValueName, AbstractNameDefinition, AnonymousParamName, ParamName, NameWrapper
from jedi.inference.filters import FunctionExecutionFilter, AnonymousFunctionExecutionFilter
from typing import Optional, List, Generator, Tuple, Union, Any

class LambdaName(AbstractNameDefinition):
    string_name: str
    api_type: str
    
    def __init__(self, lambda_value: TreeValue) -> None:
        ...
    
    @property
    def start_pos(self) -> Any:
        ...
    
    def infer(self) -> ValueSet:
        ...

class FunctionAndClassBase(TreeValue):
    def get_qualified_names(self) -> Optional[Tuple[str, ...]]:
        ...

class FunctionMixin:
    api_type: str
    
    def get_filters(self, origin_scope: Any = None) -> Generator[Union[FunctionExecutionFilter, AnonymousFunctionExecutionFilter], None, None]:
        ...
    
    def py__get__(self, instance: Any, class_value: Any) -> ValueSet:
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
    
    def py__call__(self, arguments: Any) -> ValueSet:
        ...
    
    def _as_context(self, arguments: Any = None) -> Any:
        ...
    
    def get_signatures(self) -> List[TreeSignature]:
        ...

class FunctionValue(FunctionMixin, FunctionAndClassBase):
    @classmethod
    def from_context(cls, context: Any, tree_node: Any) -> Union['OverloadedFunctionValue', 'FunctionValue']:
        ...
    
    def py__class__(self) -> Any:
        ...
    
    def get_default_param_context(self) -> Any:
        ...
    
    def get_signature_functions(self) -> List[Any]:
        ...

class FunctionNameInClass(NameWrapper):
    def __init__(self, class_context: Any, name: NameWrapper) -> None:
        ...
    
    def get_defining_qualified_value(self) -> Any:
        ...

class MethodValue(FunctionValue):
    def __init__(self, inference_state: Any, class_context: Context, *args: Any, **kwargs: Any) -> None:
        ...
    
    def get_default_param_context(self) -> Context:
        ...
    
    def get_qualified_names(self) -> Optional[Tuple[str, ...]]:
        ...
    
    @property
    def name(self) -> FunctionNameInClass:
        ...

class BaseFunctionExecutionContext:
    def infer_annotations(self) -> ValueSet:
        ...
    
    def get_return_values(self, check_yields: bool = False) -> ValueSet:
        ...
    
    def _get_yield_lazy_value(self, yield_expr: Any) -> Generator[LazyKnownValue, None, None]:
        ...
    
    def get_yield_lazy_values(self, is_async: bool = False) -> Generator[LazyKnownValue, None, None]:
        ...
    
    def merge_yield_values(self, is_async: bool = False) -> ValueSet:
        ...
    
    def is_generator(self) -> bool:
        ...
    
    def infer(self) -> ValueSet:
        ...
    
    @property
    def is_coroutine(self) -> bool:
        ...

class FunctionExecutionContext(BaseFunctionExecutionContext):
    def __init__(self, function_value: FunctionValue, arguments: Any) -> None:
        ...
    
    def get_filters(self, until_position: Any = None, origin_scope: Any = None) -> Generator[FunctionExecutionFilter, None, None]:
        ...
    
    def infer_annotations(self) -> ValueSet:
        ...
    
    def get_param_names(self) -> List[ParamName]:
        ...

class AnonymousFunctionExecution(BaseFunctionExecutionContext):
    def infer_annotations(self) -> ValueSet:
        ...
    
    def get_filters(self, until_position: Any = None, origin_scope: Any = None) -> Generator[AnonymousFunctionExecutionFilter, None, None]:
        ...
    
    def get_param_names(self) -> List[AnonymousParamName]:
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

def _find_overload_functions(context: Any, tree_node: Any) -> Generator[Any, None, None]:
    ...