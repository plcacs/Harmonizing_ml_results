""" Meager code path measurement tool.
    Ned Batchelder
    http://nedbatchelder.com/blog/200803/python_code_complexity_microtool.html
    MIT License.
"""
from __future__ import with_statement
import optparse
import sys
import tokenize
from collections import defaultdict
from typing import Any, Callable, DefaultDict, Dict, List, Optional, Tuple, Type, Union
import ast
from ast import iter_child_nodes

__version__ = '0.7.0'


class ASTVisitor:
    """Performs a depth-first walk of the AST."""

    def __init__(self) -> None:
        self.node: Optional[ast.AST] = None
        self._cache: Dict[Type[ast.AST], Callable[[ast.AST, Any], Any]] = {}

    def default(self, node: ast.AST, *args: Any) -> None:
        for child in iter_child_nodes(node):
            self.dispatch(child, *args)

    def dispatch(self, node: ast.AST, *args: Any) -> Any:
        self.node = node
        klass = node.__class__
        meth = self._cache.get(klass)
        if meth is None:
            className = klass.__name__
            meth = getattr(self.visitor, f'visit{className}', self.default)
            self._cache[klass] = meth
        return meth(node, *args)

    def preorder(self, tree: ast.AST, visitor: Any, *args: Any) -> None:
        """Do preorder walk of tree using visitor"""
        self.visitor = visitor
        visitor.visit = self.dispatch
        self.dispatch(tree, *args)


class PathNode:
    def __init__(self, name: str, look: str = 'circle') -> None:
        self.name: str = name
        self.look: str = look

    def to_dot(self) -> None:
        print(f'node [shape={self.look},label="{self.name}"] {self.dot_id()};')

    def dot_id(self) -> int:
        return id(self)


class PathGraph:
    def __init__(self, name: str, entity: str, lineno: int, column: int = 0) -> None:
        self.name: str = name
        self.entity: str = entity
        self.lineno: int = lineno
        self.column: int = column
        self.nodes: DefaultDict[PathNode, List[PathNode]] = defaultdict(list)

    def connect(self, n1: PathNode, n2: PathNode) -> None:
        self.nodes[n1].append(n2)
        self.nodes[n2] = []

    def to_dot(self) -> None:
        print('subgraph {')
        for node in self.nodes:
            node.to_dot()
        for node, nexts in self.nodes.items():
            for next_node in nexts:
                print(f'{node.dot_id()} -- {next_node.dot_id()};')
        print('}')

    def complexity(self) -> int:
        """ Return the McCabe complexity for the graph.
            V-E+2
        """
        num_edges = sum(len(n) for n in self.nodes.values())
        num_nodes = len(self.nodes)
        return num_edges - num_nodes + 2


class PathGraphingAstVisitor(ASTVisitor):
    """ A visitor for a parsed Abstract Syntax Tree which finds executable
        statements.
    """

    def __init__(self) -> None:
        super().__init__()
        self.classname: str = ''
        self.graphs: Dict[str, PathGraph] = {}
        self.reset()

    def reset(self) -> None:
        self.graph: Optional[PathGraph] = None
        self.tail: Optional[PathNode] = None

    def dispatch_list(self, node_list: List[ast.AST]) -> None:
        for node in node_list:
            self.dispatch(node)

    def visitFunctionDef(self, node: ast.FunctionDef) -> None:
        if self.classname:
            entity = f'{self.classname}{node.name}'
        else:
            entity = node.name
        name = f'{node.lineno}:{node.col_offset}: {repr(entity)}'
        if self.graph is not None:
            pathnode = self.appendPathNode(name)
            self.tail = pathnode
            self.dispatch_list(node.body)
            bottom = PathNode('', look='point')
            self.graph.connect(self.tail, bottom)
            self.graph.connect(pathnode, bottom)
            self.tail = bottom
        else:
            self.graph = PathGraph(name, entity, node.lineno, node.col_offset)
            pathnode = PathNode(name)
            self.tail = pathnode
            self.dispatch_list(node.body)
            self.graphs[f'{self.classname}{node.name}'] = self.graph
            self.reset()

    visitAsyncFunctionDef = visitFunctionDef

    def visitClassDef(self, node: ast.ClassDef) -> None:
        old_classname = self.classname
        self.classname += f'{node.name}.'
        self.dispatch_list(node.body)
        self.classname = old_classname

    def appendPathNode(self, name: str) -> Optional[PathNode]:
        if not self.tail:
            return None
        pathnode = PathNode(name)
        if self.tail:
            self.graph.connect(self.tail, pathnode)
        self.tail = pathnode
        return pathnode

    def visitSimpleStatement(self, node: ast.stmt) -> None:
        lineno = node.lineno if node.lineno is not None else 0
        name = f'Stmt {lineno}'
        self.appendPathNode(name)

    def default(self, node: ast.AST, *args: Any) -> None:
        if isinstance(node, ast.stmt):
            self.visitSimpleStatement(node)
        else:
            super().default(node, *args)

    def visitLoop(self, node: Union[ast.For, ast.AsyncFor, ast.While]) -> None:
        name = f'Loop {node.lineno}'
        self._subgraph(node, name)

    visitAsyncFor = visitFor = visitWhile = visitLoop

    def visitIf(self, node: ast.If) -> None:
        name = f'If {node.lineno}'
        self._subgraph(node, name)

    def _subgraph(
        self,
        node: Union[ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try],
        name: str,
        extra_blocks: Tuple[ast.stmt, ...] = (),
    ) -> None:
        """create the subgraphs representing any `if` and `for` statements"""
        if self.graph is None:
            self.graph = PathGraph(name, name, node.lineno, node.col_offset)
            pathnode = PathNode(name)
            self._subgraph_parse(node, pathnode, extra_blocks)
            key = f'{self.classname}{name}'
            self.graphs[key] = self.graph
            self.reset()
        else:
            pathnode = self.appendPathNode(name)
            if pathnode:
                self._subgraph_parse(node, pathnode, extra_blocks)

    def _subgraph_parse(
        self,
        node: Union[ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try],
        pathnode: PathNode,
        extra_blocks: Tuple[ast.stmt, ...],
    ) -> None:
        """parse the body and any `else` block of `if` and `for` statements"""
        loose_ends: List[PathNode] = []
        self.tail = pathnode
        self.dispatch_list(node.body)
        loose_ends.append(self.tail)
        for extra in extra_blocks:
            self.tail = pathnode
            self.dispatch(extra)
            loose_ends.append(self.tail)
        if hasattr(node, 'orelse') and node.orelse:
            self.tail = pathnode
            self.dispatch_list(node.orelse)
            loose_ends.append(self.tail)
        else:
            loose_ends.append(pathnode)
        if pathnode:
            bottom = PathNode('', look='point')
            for le in loose_ends:
                if le:
                    self.graph.connect(le, bottom)
            self.tail = bottom

    def visitTryExcept(self, node: ast.Try) -> None:
        name = f'TryExcept {node.lineno}'
        self._subgraph(node, name, extra_blocks=tuple(node.handlers))

    visitTry = visitTryExcept

    def visitWith(self, node: ast.With) -> None:
        name = f'With {node.lineno}'
        self.appendPathNode(name)
        self.dispatch_list(node.body)

    visitAsyncWith = visitWith


class McCabeChecker:
    """McCabe cyclomatic complexity checker."""
    name: str = 'mccabe'
    version: str = __version__
    _code: str = 'C901'
    _error_tmpl: str = 'C901 %r is too complex (%d)'
    max_complexity: int = -1

    def __init__(self, tree: ast.AST, filename: str) -> None:
        self.tree: ast.AST = tree
        self.filename: str = filename

    @classmethod
    def add_options(cls, parser: optparse.OptionParser) -> None:
        flag = '--max-complexity'
        kwargs: Dict[str, Any] = {
            'default': -1,
            'action': 'store',
            'type': 'int',
            'help': 'McCabe complexity threshold',
            'parse_from_config': True
        }
        config_opts = getattr(parser, 'config_options', None)
        if isinstance(config_opts, list):
            kwargs.pop('parse_from_config')
            parser.add_option(flag, **kwargs)
            parser.config_options.append('max-complexity')  # type: ignore
        else:
            parser.add_option(flag, **kwargs)

    @classmethod
    def parse_options(cls, options: optparse.Values) -> None:
        cls.max_complexity = int(options.max_complexity)

    def run(self) -> List[Tuple[int, int, str, Type['McCabeChecker']]]:
        if self.max_complexity < 0:
            return []
        visitor = PathGraphingAstVisitor()
        visitor.preorder(self.tree, visitor)
        results: List[Tuple[int, int, str, Type['McCabeChecker']]] = []
        for graph in visitor.graphs.values():
            if graph.complexity() > self.max_complexity:
                text = self._error_tmpl % (graph.entity, graph.complexity())
                results.append((graph.lineno, graph.column, text, type(self)))
        return results


def get_code_complexity(code: str, threshold: int = 7, filename: str = 'stdin') -> int:
    try:
        tree: ast.AST = compile(code, filename, 'exec', ast.PyCF_ONLY_AST)
    except SyntaxError:
        e: SyntaxError = sys.exc_info()[1]  # type: ignore
        sys.stderr.write(f'Unable to parse {filename}: {e}\n')
        return 0
    complx: List[str] = []
    McCabeChecker.max_complexity = threshold
    checker = McCabeChecker(tree, filename)
    for lineno, offset, text, check in checker.run():
        complx.append(f'{filename}:{lineno}:1: {text}')
    if not complx:
        return 0
    print('\n'.join(complx))
    return len(complx)


def get_module_complexity(module_path: str, threshold: int = 7) -> int:
    """Returns the complexity of a module"""
    code: str = _read(module_path)
    return get_code_complexity(code, threshold, filename=module_path)


def _read(filename: str) -> str:
    if (2, 5) < sys.version_info < (3, 0):
        with open(filename, 'rU') as f:
            return f.read()
    elif (3, 0) <= sys.version_info < (4, 0):
        'Read the source code.'
        try:
            with open(filename, 'rb') as f:
                encoding, _ = tokenize.detect_encoding(f.readline)
        except (LookupError, SyntaxError, UnicodeError):
            with open(filename, encoding='latin-1') as f:
                return f.read()
        with open(filename, 'r', encoding=encoding) as f:
            return f.read()
    else:
        raise RuntimeError("Unsupported Python version")


def main(argv: Optional[List[str]] = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    opar = optparse.OptionParser()
    opar.add_option('-d', '--dot', dest='dot', help='output a graphviz dot file', action='store_true')
    opar.add_option('-m', '--min', dest='threshold', help='minimum complexity for output', type='int', default=1)
    options, args = opar.parse_args(argv)
    if not args:
        sys.stderr.write('No input file provided.\n')
        sys.exit(1)
    code: str = _read(args[0])
    tree: ast.AST = compile(code, args[0], 'exec', ast.PyCF_ONLY_AST)
    visitor = PathGraphingAstVisitor()
    visitor.preorder(tree, visitor)
    if options.dot:
        print('graph {')
        for graph in visitor.graphs.values():
            if not options.threshold or graph.complexity() >= options.threshold:
                graph.to_dot()
        print('}')
    else:
        for graph in visitor.graphs.values():
            if graph.complexity() >= options.threshold:
                print(graph.name, graph.complexity())


if __name__ == '__main__':
    main(sys.argv[1:])
