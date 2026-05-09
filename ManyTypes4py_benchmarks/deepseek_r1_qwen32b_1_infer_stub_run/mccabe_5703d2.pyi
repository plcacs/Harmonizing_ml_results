from __future__ import annotations
import ast
import optparse
import sys
from typing import (
    Any,
    Callable,
    ClassVar,
    DefaultDict,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

class ASTVisitor:
    _cache: Dict[type, Callable]
    node: ast.AST

    def __init__(self) -> None:
        ...

    def default(self, node: ast.AST, *args: Any) -> None:
        ...

    def dispatch(self, node: ast.AST, *args: Any) -> Any:
        ...

    def preorder(self, tree: ast.AST, visitor: Any, *args: Any) -> None:
        ...

class PathNode:
    name: str
    look: str

    def __init__(self, name: str, look: str = 'circle') -> None:
        ...

    def to_dot(self) -> None:
        ...

    def dot_id(self) -> int:
        ...

class PathGraph:
    name: str
    entity: str
    lineno: int
    column: int
    nodes: DefaultDict[PathNode, List[PathNode]]

    def __init__(self, name: str, entity: str, lineno: int, column: int = 0) -> None:
        ...

    def connect(self, n1: PathNode, n2: PathNode) -> None:
        ...

    def to_dot(self) -> None:
        ...

    def complexity(self) -> int:
        ...

class PathGraphingAstVisitor(ASTVisitor):
    classname: str
    graphs: Dict[str, PathGraph]
    graph: Optional[PathGraph]
    tail: Optional[PathNode]

    def __init__(self) -> None:
        ...

    def reset(self) -> None:
        ...

    def dispatch_list(self, node_list: Iterable[ast.AST]) -> None:
        ...

    def visitFunctionDef(self, node: ast.FunctionDef) -> None:
        ...

    def visitClassDef(self, node: ast.ClassDef) -> None:
        ...

    def appendPathNode(self, name: str) -> Optional[PathNode]:
        ...

    def visitSimpleStatement(self, node: ast.stmt) -> None:
        ...

    def default(self, node: ast.AST, *args: Any) -> None:
        ...

    def visitLoop(self, node: ast.stmt) -> None:
        ...

    def visitIf(self, node: ast.If) -> None:
        ...

    def _subgraph(self, node: ast.stmt, name: str, extra_blocks: Iterable[ast.stmt] = ()) -> None:
        ...

    def _subgraph_parse(self, node: ast.stmt, pathnode: PathNode, extra_blocks: Iterable[ast.stmt]) -> None:
        ...

    def visitTryExcept(self, node: ast.Try) -> None:
        ...

    def visitWith(self, node: ast.With) -> None:
        ...

class McCabeChecker:
    name: ClassVar[str] = ...
    version: ClassVar[str] = ...
    _code: ClassVar[str] = ...
    _error_tmpl: ClassVar[str] = ...
    max_complexity: int

    def __init__(self, tree: ast.AST, filename: str) -> None:
        ...

    @classmethod
    def add_options(cls, parser: optparse.OptionParser) -> None:
        ...

    @classmethod
    def parse_options(cls, options: optparse.Values) -> None:
        ...

    def run(self) -> Iterable[Tuple[int, int, str, type]]:
        ...

def get_code_complexity(code: str, threshold: int = 7, filename: str = 'stdin') -> int:
    ...

def get_module_complexity(module_path: str, threshold: int = 7) -> int:
    ...

def _read(filename: str) -> str:
    ...

def main(argv: Optional[List[str]] = None) -> None:
    ...