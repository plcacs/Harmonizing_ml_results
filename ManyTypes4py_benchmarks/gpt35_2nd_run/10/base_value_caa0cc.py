from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

class Value:
    inference_state: Any
    parent_context: Optional[Any]

    def py__getitem__(self, index_value_set: 'ValueSet', contextualized_node: Any) -> 'ValueSet':
        ...

    def py__simple_getitem__(self, index: Union[float, int, str, slice, bytes]) -> Any:
        ...

    def py__iter__(self, contextualized_node: Any = None) -> Iterable:
        ...

    def py__next__(self, contextualized_node: Any = None) -> Iterable:
        ...

    def get_signatures(self) -> List:
        ...

    def get_type_hint(self, add_class_info: bool = True) -> Optional[str]:
        ...

    def infer_type_vars(self, value_set: 'ValueSet') -> Dict:
        ...

class ValueSet:
    _set: Set

    def __init__(self, iterable: Iterable) -> None:
        ...

    @classmethod
    def from_sets(cls, sets: Iterable) -> 'ValueSet':
        ...

    def __or__(self, other: 'ValueSet') -> 'ValueSet':
        ...

    def __and__(self, other: 'ValueSet') -> 'ValueSet':
        ...

    def filter(self, filter_func: Any) -> 'ValueSet':
        ...

    def py__class__(self) -> 'ValueSet':
        ...

    def iterate(self, contextualized_node: Any = None, is_async: bool = False) -> Iterable:
        ...

    def execute(self, arguments: Any) -> 'ValueSet':
        ...

    def execute_with_values(self, *args: Any, **kwargs: Any) -> 'ValueSet':
        ...

    def goto(self, *args: Any, **kwargs: Any) -> List:
        ...

    def py__getattribute__(self, *args: Any, **kwargs: Any) -> 'ValueSet':
        ...

    def get_item(self, *args: Any, **kwargs: Any) -> 'ValueSet':
        ...

    def try_merge(self, function_name: str) -> 'ValueSet':
        ...

    def gather_annotation_classes(self) -> 'ValueSet':
        ...

    def get_signatures(self) -> List:
        ...

    def get_type_hint(self, add_class_info: bool = True) -> Optional[str]:
        ...

    def infer_type_vars(self, value_set: 'ValueSet') -> Dict:
        ...

NO_VALUES: ValueSet
