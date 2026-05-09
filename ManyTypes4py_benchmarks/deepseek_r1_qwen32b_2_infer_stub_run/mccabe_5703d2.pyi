""" Meager code path measurement tool. """

from __future__ import annotations
from collections import defaultdict
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)
import ast
import optparse
import sys
from ast import AST

__version__ = '0.7.0'

class ASTVisitor:
    def __init__(self) -> None:
        ...
    
    def default(self, node: Any, *args: Any) -> None:
        ...
    
    def dispatch(self, node: Any, *args: Any) -> None:
        ...
    
    def preorder(self, tree: Any, visitor: Any, *args: Any) -> None:
        ...

class PathNode:
    def __init__(self, name: str, look: str = 'circle') -> None:
        ...
    
    def to_dot(self) -> None:
        ...
    
    def dot_id(self) -> int:
        ...

class PathGraph:
    def __init__(self, name: str, entity: str, lineno: int, column: int = 0) -> None:
        ...
    
    def connect(self, n1: PathNode, n2: PathNode) -> None:
        ...
    
    def to_dot(self) -> None:
        ...
    
    def complexity(self) -> int:
        ...

class PathGraphingAstVisitor(ASTVisitor):
    def __init__(self) -> None:
        ...
    
    def reset(self) -> None:
        ...
    
    def dispatch_list(self, node_list: Iterable[Any]) -> None:
        ...
    
    def visitFunctionDef(self, node: Any) -> None:
        ...
    visitAsyncFunctionDef = visitFunctionDef
    
    def visitClassDef(self, node: Any) -> None:
        ...
    
    def appendPathNode(self, name: str) -> Optional[PathNode]:
        ...
    
    def visitSimpleStatement(self, node: Any) -> None:
        ...
    
    def default(self, node: Any, *args: Any) -> None:
        ...
    
    def visitLoop(self, node: Any) -> None:
        ...
    visitAsyncFor = visitFor = visitWhile = visitLoop
    
    def visitIf(self, node: Any) -> None:
        ...
    
    def _subgraph(self, node: Any, name: str, extra_blocks: Iterable[Any] = ()) -> None:
        ...
    
    def _subgraph_parse(self, node: Any, pathnode: PathNode, extra_blocks: Iterable[Any]) -> None:
        ...
    
    def visitTryExcept(self, node: Any) -> None:
        ...
    visitTry = visitTryExcept
    
    def visitWith(self, node: Any) -> None:
        ...
    visitAsyncWith = visitWith

class McCabeChecker:
    name: str
    version: str
    _code: str
    _error_tmpl: str
    max_complexity: int
    
    def __init__(self, tree: AST, filename: str) -> None:
        ...
    
    @classmethod
    def add_options(cls, parser: optparse.OptionParser) -> None:
        ...
    
    @classmethod
    def parse_options(cls, options: Any) -> None:
        ...
    
    def run(self) -> Iterator[Tuple[int, int, str, Type[MccabeChecker]]]:
        ...

def get_code_complexity(code: str, threshold: int = 7, filename: str = 'stdin') -> int:
    ...

def get_module_complexity(module_path: str, threshold: int = 7) -> int:
    ...

def _read(filename: str) -> str:
    ...

def main(argv: Optional[List[str]] = None) -> None:
    ...