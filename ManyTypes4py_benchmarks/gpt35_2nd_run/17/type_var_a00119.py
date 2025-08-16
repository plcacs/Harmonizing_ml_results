from typing import Dict, Any, Iterable, Set, Union

class TypeVarClass(ValueWrapper):

    def py__call__(self, arguments: Any) -> ValueSet:
        ...

    def _find_string_name(self, lazy_value: Any) -> Union[str, None]:
        ...

class TypeVar(BaseTypingValue):

    def __init__(self, parent_context: Any, tree_name: str, var_name: str, unpacked_args: Iterable[Tuple[Union[str, None], Any]]):
        ...

    def py__name__(self) -> str:
        ...

    def get_filters(self, *args: Any, **kwargs: Any) -> Iterable:
        ...

    def _get_classes(self) -> ValueSet:
        ...

    def is_same_class(self, other: Any) -> bool:
        ...

    def constraints(self) -> ValueSet:
        ...

    def define_generics(self, type_var_dict: Dict[str, Any]) -> ValueSet:
        ...

    def execute_annotation(self) -> Any:
        ...

    def infer_type_vars(self, value_set: ValueSet) -> Dict[str, ValueSet]:
        ...

    def __repr__(self) -> str:
        ...

class TypeWrapper(ValueWrapper):

    def __init__(self, wrapped_value: Any, original_value: Any):
        ...

    def execute_annotation(self) -> ValueSet:
        ...
