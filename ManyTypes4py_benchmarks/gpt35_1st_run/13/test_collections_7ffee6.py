from typing import Any, Dict, List, Union

class ExampleAnnotation(BaseAnnotation):
    pass

class Color(AutoEnum):
    RED: str = AutoEnum.auto()
    BLUE: str = AutoEnum.auto()

def test_flatdict_conversion(d: Dict[int, Any], expected: Dict[Union[int, Tuple], Any]):
    ...

def visit_even_numbers(x: Any) -> Any:
    ...

def add_to_visited_list(x: Any) -> None:
    ...

def remove_nested_keys(keys: List[str], obj: Any) -> Any:
    ...

def isiterable(obj: Any) -> bool:
    ...

def get_from_dict(dct: Dict, keys: Union[str, List[str]], default: Any = None) -> Any:
    ...

def set_in_dict(dct: Dict, keys: Union[str, List[str]], value: Any) -> None:
    ...

def deep_merge(dct: Dict, merge: Dict) -> Dict:
    ...

def deep_merge_dicts(*dicts: Dict) -> Dict:
    ...
