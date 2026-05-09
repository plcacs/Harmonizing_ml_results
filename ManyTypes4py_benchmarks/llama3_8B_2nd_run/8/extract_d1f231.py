from textwrap import dedent
from parso import split_lines
from jedi import debug
from jedi.api.exceptions import RefactoringError
from jedi.api.refactoring import Refactoring, EXPRESSION_PARTS
from jedi.common import indent_block
from jedi.parser_utils import function_is_classmethod, function_is_staticmethod

def extract_variable(
    inference_state: object, 
    path: str, 
    module_node: object, 
    name: str, 
    pos: tuple, 
    until_pos: tuple
) -> Refactoring:
    ...

def extract_function(
    inference_state: object, 
    path: str, 
    module_context: object, 
    name: str, 
    pos: tuple, 
    until_pos: tuple
) -> Refactoring:
    ...
