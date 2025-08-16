from typing import Any, Dict, List, Union

class ExampleAnnotation(BaseAnnotation):
    pass

class Color(AutoEnum):
    RED = AutoEnum.auto()
    BLUE = AutoEnum.auto()

def test_flatdict_conversion(d: Dict[Any, Any], expected: Dict[Tuple[Any, ...], Any]):
    ...

def negative_even_numbers(x: Union[int, Any]) -> Union[int, Any]:
    ...

def all_negative_numbers(x: Union[int, Any]) -> Union[int, Any]:
    ...

def visit_even_numbers(x: Union[int, Any]) -> Union[int, Any]:
    ...

def add_to_visited_list(x: Any):
    ...

def remove_nested_keys(keys: List[str], obj: Any) -> Any:
    ...

def isiterable(obj: Any) -> bool:
    ...

def get_from_dict(dct: Dict[Any, Any], keys: Union[str, List[str]], default: Any = None) -> Any:
    ...

def set_in_dict(dct: Dict[Any, Any], keys: Union[str, List[str]], value: Any):
    ...

def deep_merge(dct: Dict[Any, Any], merge: Dict[Any, Any]) -> Dict[Any, Any]:
    ...

def deep_merge_dicts(*dicts: Dict[Any, Any]) -> Dict[Any, Any]:
    ...
