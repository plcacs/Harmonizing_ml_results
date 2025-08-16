from __future__ import with_statement
from typing import List, Dict, Union, Tuple, Any, Set, FrozenSet, Generator
import os
import sys
import logging
import operator
import collections
import io
from itertools import chain
from .pgen2 import driver, tokenize, token
from .fixer_util import find_root
from . import pytree, pygram
from . import btm_utils as bu
from . import btm_matcher as bm

def get_all_fix_names(fixer_pkg: str, remove_prefix: bool = True) -> List[str]:
    ...

class _EveryNode(Exception):
    pass

def _get_head_types(pat: Union[pytree.NodePattern, pytree.LeafPattern]) -> Set[int]:
    ...

def _get_headnode_dict(fixer_list: List[Any]) -> Dict[int, List[Any]]:
    ...

def get_fixers_from_package(pkg_name: str) -> List[str]:
    ...

def _identity(obj: Any) -> Any:
    ...

def _detect_future_features(source: str) -> FrozenSet[str]:
    ...

class FixerError(Exception):
    ...

class RefactoringTool:
    _default_options: Dict[str, bool] = {'print_function': False, 'write_unchanged_files': False}
    CLASS_PREFIX: str = 'Fix'
    FILE_PREFIX: str = 'fix_'

    def __init__(self, fixer_names: List[str], options: Dict[str, Any] = None, explicit: List[str] = None) -> None:
        ...

    def get_fixers(self) -> Tuple[List[Any], List[Any]]:
        ...

    def log_error(self, msg: str, *args: Any, **kwds: Any) -> None:
        ...

    def log_message(self, msg: str, *args: Any) -> None:
        ...

    def log_debug(self, msg: str, *args: Any) -> None:
        ...

    def print_output(self, old_text: str, new_text: str, filename: str, equal: bool) -> None:
        ...

    def refactor(self, items: List[str], write: bool = False, doctests_only: bool = False) -> None:
        ...

    def refactor_dir(self, dir_name: str, write: bool = False, doctests_only: bool = False) -> None:
        ...

    def _read_python_source(self, filename: str) -> Tuple[Union[str, None], Union[str, None]]:
        ...

    def refactor_file(self, filename: str, write: bool = False, doctests_only: bool = False) -> None:
        ...

    def refactor_string(self, data: str, name: str) -> Any:
        ...

    def refactor_stdin(self, doctests_only: bool = False) -> None:
        ...

    def refactor_tree(self, tree: Any, name: str) -> bool:
        ...

    def traverse_by(self, fixers: Dict[int, List[Any]], traversal: Generator) -> None:
        ...

    def processed_file(self, new_text: str, filename: str, old_text: str = None, write: bool = False, encoding: str = None) -> None:
        ...

    def write_file(self, new_text: str, filename: str, old_text: str, encoding: str = None) -> None:
        ...

    def refactor_docstring(self, input: str, filename: str) -> str:
        ...

    def refactor_doctest(self, block: List[str], lineno: int, indent: str, filename: str) -> List[str]:
        ...

    def summarize(self) -> None:
        ...

    def parse_block(self, block: List[str], lineno: int, indent: str) -> Any:
        ...

    def wrap_toks(self, block: List[str], lineno: int, indent: str) -> Generator:
        ...

    def gen_lines(self, block: List[str], indent: str) -> Generator:
        ...

class MultiprocessingUnsupported(Exception):
    pass

class MultiprocessRefactoringTool(RefactoringTool):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ...

    def refactor(self, items: List[str], write: bool = False, doctests_only: bool = False, num_processes: int = 1) -> None:
        ...

    def _child(self) -> None:
        ...

    def refactor_file(self, *args: Any, **kwargs: Any) -> None:
        ...
