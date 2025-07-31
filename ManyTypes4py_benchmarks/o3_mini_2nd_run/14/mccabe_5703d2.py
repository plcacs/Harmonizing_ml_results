#!/usr/bin/env python
""" Meager code path measurement tool.
    Ned Batchelder
    http://nedbatchelder.com/blog/200803/python_code_complexity_microtool.html
    MIT License.
"""
from __future__ import with_statement
import optparse
import sys
import tokenize
import ast
from ast import iter_child_nodes
from collections import defaultdict
from typing import Any, Dict, DefaultDict, Generator, List, Optional, Tuple, Type

__version__ = '0.7.0'


class ASTVisitor(object):
    """Performs a depth-first walk of the AST."""

    def __init__(self) -> None:
        self.node: Optional[ast.AST] = None
        self._cache: Dict[type, Any] = {}

    def default(self, node: ast.AST, *args: Any) -> None:
        for child in iter_child_nodes(node):
            self.dispatch(child, *args)

    def dispatch(self, node: ast.AST, *args: Any) -> Any:
        self.node = node
        klass = node.__class__
        meth = self._cache.get(klass)
        if meth is None:
            className = klass.__name__
            meth = getattr(self.visitor, 'visit' + className, self.default)
            self._cache[klass] = meth
        return meth(node, *args)

    def preorder(self, tree: ast.AST, visitor: Any, *args: Any) -> None:
        """Do preorder walk of tree using visitor"""
        self.visitor = visitor
        visitor.visit = self.dispatch
        self.dispatch(tree, *args)


class PathNode(object):

    def __init__(self, name: str, look: str = 'circle') -> None:
        self.name: str = name
        self.look: str = look

    def to_dot(self) -> None:
        print('node [shape=%s,label="%s"] %d;' % (self.look, self.name, self.dot_id()))

    def dot_id(self) -> int:
        return id(self)


class PathGraph(object):

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
            for nxt in nexts:
                print('%s -- %s;' % (node.dot_id(), nxt.dot_id()))
        print('}')

    def complexity(self) -> int:
        """ Return the McCabe complexity for the graph.
            V-E+2
        """
        num_edges: int = sum([len(n) for n in self.nodes.values()])
        num_nodes: int = len(self.nodes)
        return num_edges - num_nodes + 2


class PathGraphingAstVisitor(ASTVisitor):
    """ A visitor for a parsed Abstract Syntax Tree which finds executable
        statements.
    """

    def __init__(self) -> None:
        super(PathGraphingAstVisitor, self).__init__()
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
            entity: str = '%s%s' % (self.classname, node.name)
        else:
            entity = node.name
        name: str = '%d:%d: %r' % (node.lineno, node.col_offset, entity)
        if self.graph is not None:
            pathnode: Optional[PathNode] = self.appendPathNode(name)
            self.tail = pathnode
            self.dispatch_list(node.body)
            bottom: PathNode = PathNode('', look='point')
            if self.tail is not None:
                self.graph.connect(self.tail, bottom)
            if pathnode is not None:
                self.graph.connect(pathnode, bottom)
            self.tail = bottom
        else:
            self.graph = PathGraph(name, entity, node.lineno, node.col_offset)
            pathnode = PathNode(name)
            self.tail = pathnode
            self.dispatch_list(node.body)
            self.graphs['%s%s' % (self.classname, node.name)] = self.graph
            self.reset()
    visitAsyncFunctionDef = visitFunctionDef

    def visitClassDef(self, node: ast.ClassDef) -> None:
        old_classname: str = self.classname
        self.classname += node.name + '.'
        self.dispatch_list(node.body)
        self.classname = old_classname

    def appendPathNode(self, name: str) -> Optional[PathNode]:
        if not self.tail:
            return None
        pathnode: PathNode = PathNode(name)
        if self.graph is not None:
            self.graph.connect(self.tail, pathnode)
        self.tail = pathnode
        return pathnode

    def visitSimpleStatement(self, node: ast.AST) -> None:
        lineno: int = node.lineno if getattr(node, "lineno", None) is not None else 0
        name: str = 'Stmt %d' % lineno
        self.appendPathNode(name)

    def default(self, node: ast.AST, *args: Any) -> None:
        if isinstance(node, ast.stmt):
            self.visitSimpleStatement(node)
        else:
            super(PathGraphingAstVisitor, self).default(node, *args)

    def visitLoop(self, node: ast.AST) -> None:
        name: str = 'Loop %d' % getattr(node, "lineno", 0)
        self._subgraph(node, name)
    visitAsyncFor = visitFor = visitWhile = visitLoop

    def visitIf(self, node: ast.If) -> None:
        name: str = 'If %d' % node.lineno
        self._subgraph(node, name)

    def _subgraph(self, node: ast.AST, name: str, extra_blocks: Tuple[Any, ...] = ()) -> None:
        """create the subgraphs representing any `if` and `for` statements"""
        if self.graph is None:
            self.graph = PathGraph(name, name, getattr(node, "lineno", 0), getattr(node, "col_offset", 0))
            pathnode: PathNode = PathNode(name)
            self._subgraph_parse(node, pathnode, extra_blocks)
            self.graphs['%s%s' % (self.classname, name)] = self.graph
            self.reset()
        else:
            pathnode = self.appendPathNode(name)
            if pathnode is not None:
                self._subgraph_parse(node, pathnode, extra_blocks)

    def _subgraph_parse(self, node: ast.AST, pathnode: PathNode, extra_blocks: Tuple[Any, ...]) -> None:
        """parse the body and any `else` block of `if` and `for` statements"""
        loose_ends: List[PathNode] = []
        self.tail = pathnode
        if hasattr(node, 'body'):
            self.dispatch_list(node.body)
        if self.tail is not None:
            loose_ends.append(self.tail)
        for extra in extra_blocks:
            self.tail = pathnode
            if hasattr(extra, 'body'):
                self.dispatch_list(extra.body)
            if self.tail is not None:
                loose_ends.append(self.tail)
        if hasattr(node, 'orelse') and node.orelse:
            self.tail = pathnode
            self.dispatch_list(node.orelse)
            if self.tail is not None:
                loose_ends.append(self.tail)
        else:
            loose_ends.append(pathnode)
        bottom: PathNode = PathNode('', look='point')
        if self.graph is not None:
            for le in loose_ends:
                self.graph.connect(le, bottom)
        self.tail = bottom

    def visitTryExcept(self, node: ast.Try) -> None:
        name: str = 'TryExcept %d' % node.lineno
        self._subgraph(node, name, extra_blocks=tuple(getattr(node, 'handlers', ())))
    visitTry = visitTryExcept

    def visitWith(self, node: ast.With) -> None:
        name: str = 'With %d' % node.lineno
        self.appendPathNode(name)
        if hasattr(node, 'body'):
            self.dispatch_list(node.body)
    visitAsyncWith = visitWith


class McCabeChecker(object):
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
        flag: str = '--max-complexity'
        kwargs: Dict[str, Any] = {'default': -1,
                                  'action': 'store',
                                  'type': int,
                                  'help': 'McCabe complexity threshold',
                                  'parse_from_config': 'True'}
        config_opts = getattr(parser, 'config_options', None)
        if isinstance(config_opts, list):
            kwargs.pop('parse_from_config')
            parser.add_option(flag, **kwargs)
            parser.config_options.append('max-complexity')
        else:
            parser.add_option(flag, **kwargs)

    @classmethod
    def parse_options(cls, options: optparse.Values) -> None:
        cls.max_complexity = int(options.max_complexity)

    def run(self) -> Generator[Tuple[int, int, str, Type[Any]], None, None]:
        if self.max_complexity < 0:
            return
        visitor: PathGraphingAstVisitor = PathGraphingAstVisitor()
        visitor.preorder(self.tree, visitor)
        for graph in visitor.graphs.values():
            if graph.complexity() > self.max_complexity:
                text: str = self._error_tmpl % (graph.entity, graph.complexity())
                yield (graph.lineno, graph.column, text, type(self))


def get_code_complexity(code: str, threshold: int = 7, filename: str = 'stdin') -> int:
    try:
        tree: ast.AST = compile(code, filename, 'exec', ast.PyCF_ONLY_AST)
    except SyntaxError:
        e: Exception = sys.exc_info()[1]
        sys.stderr.write('Unable to parse %s: %s\n' % (filename, e))
        return 0
    complx: List[str] = []
    McCabeChecker.max_complexity = threshold
    for lineno, offset, text, check in McCabeChecker(tree, filename).run():
        complx.append('%s:%d:1: %s' % (filename, lineno, text))
    if len(complx) == 0:
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
        # Read the source code.
        try:
            with open(filename, 'rb') as f:
                encoding, _ = tokenize.detect_encoding(f.readline)
        except (LookupError, SyntaxError, UnicodeError):
            with open(filename, encoding='latin-1') as f:
                return f.read()
        with open(filename, 'r', encoding=encoding) as f:
            return f.read()
    return ""


def main(argv: Optional[List[str]] = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    opar: optparse.OptionParser = optparse.OptionParser()
    opar.add_option('-d', '--dot', dest='dot', help='output a graphviz dot file', action='store_true')
    opar.add_option('-m', '--min', dest='threshold', help='minimum complexity for output', type='int', default=1)
    options, args = opar.parse_args(argv)
    code: str = _read(args[0])
    tree: ast.AST = compile(code, args[0], 'exec', ast.PyCF_ONLY_AST)
    visitor: PathGraphingAstVisitor = PathGraphingAstVisitor()
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