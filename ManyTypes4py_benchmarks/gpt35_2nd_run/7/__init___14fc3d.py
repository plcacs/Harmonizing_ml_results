from pathlib import Path
from jedi.api import classes, interpreter, helpers, refactoring
from jedi.api.completion import Completion, search_in_module
from jedi.api.environment import InterpreterEnvironment
from jedi.api.errors import parso_to_jedi_errors
from jedi.api.project import get_default_project, Project
from jedi.inference import InferenceState
from jedi.inference.arguments import try_iter_content
from jedi.inference.base_value import ValueSet
from jedi.inference.gradual.conversion import convert_values
from jedi.inference.gradual.utils import load_proper_stub_module
from jedi.inference.helpers import infer_call_of_leaf
from jedi.inference.references import find_references
from jedi.inference.syntax_tree import tree_name_to_values
from jedi.inference.sys_path import transform_path_to_dotted
from jedi.inference.utils import to_list
from jedi.inference.value import ModuleValue
from jedi.inference.value.iterable import unpack_tuple_to_dict
from jedi.parser_utils import get_executable_nodes
from jedi import cache, debug, settings
from jedi.file_io import KnownContentFileIO
from jedi.api.helpers import validate_line_column
from jedi.api.keywords import KeywordName
from jedi.api.refactoring.extract import extract_function, extract_variable
from jedi.api.project import get_default_project, Project
from jedi.api.refactoring import refactoring
from jedi.api import refactoring
from jedi.api import helpers
from jedi.api import interpreter
from jedi.api import classes

class Script:
    def __init__(self, code: str = None, *, path: Path = None, environment: InterpreterEnvironment = None, project: Project = None) -> None:
        ...

    def complete(self, line: int = None, column: int = None, *, fuzzy: bool = False) -> list[Completion]:
        ...

    def infer(self, line: int = None, column: int = None, *, only_stubs: bool = False, prefer_stubs: bool = False) -> list[classes.Name]:
        ...

    def goto(self, line: int = None, column: int = None, *, follow_imports: bool = False, follow_builtin_imports: bool = False, only_stubs: bool = False, prefer_stubs: bool = False) -> list[classes.Name]:
        ...

    def search(self, string: str, *, all_scopes: bool = False) -> list[classes.Name]:
        ...

    def complete_search(self, string: str, **kwargs) -> list[Completion]:
        ...

    def help(self, line: int = None, column: int = None) -> list[classes.Name]:
        ...

    def get_references(self, line: int = None, column: int = None, **kwargs) -> list[classes.Name]:
        ...

    def get_signatures(self, line: int = None, column: int = None) -> list[classes.Signature]:
        ...

    def get_context(self, line: int = None, column: int = None) -> classes.Name:
        ...

    def get_names(self, **kwargs) -> list[classes.Name]:
        ...

    def get_syntax_errors(self) -> list[classes.SyntaxError]:
        ...

    def rename(self, line: int = None, column: int = None, *, new_name: str) -> refactoring.Refactoring:
        ...

    def extract_variable(self, line: int, column: int, *, new_name: str, until_line: int = None, until_column: int = None) -> refactoring.Refactoring:
        ...

    def extract_function(self, line: int, column: int, *, new_name: str, until_line: int = None, until_column: int = None) -> refactoring.Refactoring:
        ...

    def inline(self, line: int = None, column: int = None) -> refactoring.Refactoring:
        ...

class Interpreter(Script):
    def __init__(self, code: str, namespaces: list[dict], *, project: Project = None, **kwds) -> None:
        ...

def preload_module(*modules: str) -> None:
    ...

def set_debug_function(func_cb=debug.print_to_stdout, warnings: bool = True, notices: bool = True, speed: bool = True) -> None:
    ...
