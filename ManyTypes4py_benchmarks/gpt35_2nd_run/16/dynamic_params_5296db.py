from jedi.inference.value import ValueSet, NO_VALUES
from jedi.inference.base_value import Value
from typing import List, Optional, Iterator, Tuple

def _avoid_recursions(func: Callable[[Value, int], ValueSet]) -> Callable[[Value, int], ValueSet]:
    ...

def dynamic_param_lookup(function_value: Value, param_index: int) -> ValueSet:
    ...

def _search_function_arguments(module_context: Value, funcdef: Value, string_name: str) -> List[Value]:
    ...

def _get_lambda_name(node: Value) -> Optional[str]:
    ...

def _get_potential_nodes(module_value: Value, func_string_name: str) -> Iterator[Tuple[Value, Value]]:
    ...

def _check_name_for_execution(inference_state: Value, context: Value, compare_node: Value, name: Value, trailer: Value) -> Iterator[Value]:
    ...
