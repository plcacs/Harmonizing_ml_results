from typing import List, Tuple

def _add_argument_issue(error_name: str, lazy_value: LazyTreeValue, message: str) -> None:
    ...

class ExecutedParamName(ParamName):

    def __init__(self, function_value, arguments, param_node, lazy_value, is_default=False) -> None:
        ...

    def infer(self) -> LazyTreeValue:
        ...

    def matches_signature(self) -> bool:
        ...

    def __repr__(self) -> str:
        ...

def get_executed_param_names_and_issues(function_value, arguments) -> Tuple[List[ExecutedParamName], List[LazyTreeValue]]:
    ...

def get_executed_param_names(function_value, arguments) -> List[ExecutedParamName]:
    ...

def _error_argument_count(funcdef, actual_count: int) -> str:
    ...
