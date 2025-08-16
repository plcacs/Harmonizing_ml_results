import doctest
import os
import sys
from typing import List, Tuple, Union

PY2: bool = sys.version_info < (3, 0)
PY32: bool = sys.version_info < (3, 3)
PY33: bool = sys.version_info < (3, 4)
builtin_vars: List[str] = dir(__import__('__builtin__' if PY2 else 'builtins'))
try:
    import ast
except ImportError:
    import _ast as ast
    if 'decorator_list' not in ast.ClassDef._fields:
        ast.ClassDef.decorator_list = ()
        ast.FunctionDef.decorator_list = property(lambda s: s.decorators)
from pyflakes import messages

def getNodeType(node_class: type) -> str:
    if PY2:
        return str(unicode(node_class.__name__).upper())
    else:
        return node_class.__name__.upper()

def getAlternatives(n: ast.AST) -> Union[List[List[ast.AST]], List[List[ast.AST]]]:
    if PY32:
        if isinstance(n, (ast.If, ast.TryFinally)):
            return [n.body]
        if isinstance(n, ast.TryExcept):
            return [n.body + n.orelse] + [[hdl] for hdl in n.handlers]
    else:
        if isinstance(n, ast.If):
            return [n.body]
        if isinstance(n, ast.Try):
            return [n.body + n.orelse] + [[hdl] for hdl in n.handlers]

class _FieldsOrder(dict):
    def _get_fields(self, node_class: type) -> Tuple[str]:
        fields = node_class._fields
        if 'iter' in fields:
            key_first = 'iter'.find
        elif 'generators' in fields:
            key_first = 'generators'.find
        else:
            key_first = 'value'.find
        return tuple(sorted(fields, key=key_first, reverse=True)

    def __missing__(self, node_class: type) -> Tuple[str]:
        self[node_class] = fields = self._get_fields(node_class)
        return fields

def iter_child_nodes(node: ast.AST, omit: str = None, _fields_order: _FieldsOrder = _FieldsOrder()) -> None:
    for name in _fields_order[node.__class__]:
        if name == omit:
            continue
        field = getattr(node, name, None)
        if isinstance(field, ast.AST):
            yield field
        elif isinstance(field, list):
            for item in field:
                yield item

class Binding:
    def __init__(self, name: str, source: ast.AST) -> None:
        self.name = name
        self.source = source
        self.used = False

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} object {self.name} from line {self.source.lineno} at 0x{id(self)}>'

    def redefines(self, other: 'Definition') -> bool:
        return isinstance(other, Definition) and self.name == other.name

class Definition(Binding):
    pass

class Importation(Definition):
    def __init__(self, name: str, source: ast.AST) -> None:
        self.fullName = name
        self.redefined = []
        name = name.split('.')[0]
        super(Importation, self).__init__(name, source)

    def redefines(self, other: 'Importation') -> bool:
        if isinstance(other, Importation):
            return self.fullName == other.fullName
        return isinstance(other, Definition) and self.name == other.name

class Argument(Binding):
    pass

class Assignment(Binding):
    pass

class FunctionDefinition(Definition):
    pass

class ClassDefinition(Definition):
    pass

class ExportBinding(Binding):
    def __init__(self, name: str, source: ast.AST, scope: Scope) -> None:
        if '__all__' in scope and isinstance(source, ast.AugAssign):
            self.names = list(scope['__all__'].names)
        else:
            self.names = []
        if isinstance(source.value, (ast.List, ast.Tuple)):
            for node in source.value.elts:
                if isinstance(node, ast.Str):
                    self.names.append(node.s)
        super(ExportBinding, self).__init__(name, source)

class Scope(dict):
    importStarred: bool = False

    def __repr__(self) -> str:
        scope_cls = self.__class__.__name__
        return f'<{scope_cls} at 0x{id(self)} {dict.__repr__(self)}>'

class ClassScope(Scope):
    pass

class FunctionScope(Scope):
    usesLocals: bool = False
    alwaysUsed: set = {'__tracebackhide__', '__traceback_info__', '__traceback_supplement__'}

    def __init__(self) -> None:
        super(FunctionScope, self).__init__()
        self.globals = self.alwaysUsed.copy()
        self.returnValue = None
        self.isGenerator = False

    def unusedAssignments(self) -> List[Tuple[str, Assignment]]:
        for name, binding in self.items():
            if not binding.used and name not in self.globals and (not self.usesLocals) and isinstance(binding, Assignment):
                yield (name, binding)

class GeneratorScope(Scope):
    pass

class ModuleScope(Scope):
    pass

_MAGIC_GLOBALS: List[str] = ['__file__', '__builtins__', 'WindowsError']

def getNodeName(node: ast.AST) -> str:
    if hasattr(node, 'id'):
        return node.id
    if hasattr(node, 'name'):
        return node.name

class Checker:
    nodeDepth: int = 0
    offset: Union[None, Tuple[int, int]] = None
    traceTree: bool = False
    builtIns: set = set(builtin_vars).union(_MAGIC_GLOBALS)
    _customBuiltIns: Union[None, str] = os.environ.get('PYFLAKES_BUILTINS')
    if _customBuiltIns:
        builtIns.update(_customBuiltIns.split(','))
    del _customBuiltIns

    def __init__(self, tree: ast.AST, filename: str = '(none)', builtins: Union[None, List[str]] = None, withDoctest: bool = 'PYFLAKES_DOCTEST' in os.environ) -> None:
        self._nodeHandlers: dict = {}
        self._deferredFunctions: List[Tuple[callable, List[Scope], Union[None, Tuple[int, int]]]] = []
        self._deferredAssignments: List[Tuple[callable, List[Scope], Union[None, Tuple[int, int]]]] = []
        self.deadScopes: List[Scope] = []
        self.messages: List[messages.Message] = []
        self.filename: str = filename
        if builtins:
            self.builtIns = self.builtIns.union(builtins)
        self.withDoctest: bool = withDoctest
        self.scopeStack: List[Scope] = [ModuleScope()]
        self.exceptHandlers: List[List[str]] = [()]
        self.futuresAllowed: bool = True
        self.root: ast.AST = tree
        self.handleChildren(tree)
        self.runDeferred(self._deferredFunctions)
        self._deferredFunctions = None
        self.runDeferred(self._deferredAssignments)
        self._deferredAssignments = None
        del self.scopeStack[1:]
        self.popScope()
        self.checkDeadScopes()

    def deferFunction(self, callable: callable) -> None:
        self._deferredFunctions.append((callable, self.scopeStack[:], self.offset))

    def deferAssignment(self, callable: callable) -> None:
        self._deferredAssignments.append((callable, self.scopeStack[:], self.offset))

    def runDeferred(self, deferred: List[Tuple[callable, List[Scope], Union[None, Tuple[int, int]]]]) -> None:
        for handler, scope, offset in deferred:
            self.scopeStack = scope
            self.offset = offset
            handler()

    @property
    def scope(self) -> Scope:
        return self.scopeStack[-1]

    def popScope(self) -> None:
        self.deadScopes.append(self.scopeStack.pop())

    def checkDeadScopes(self) -> None:
        for scope in self.deadScopes:
            if isinstance(scope.get('__all__'), ExportBinding):
                all_names = set(scope['__all__'].names)
                if not scope.importStarred and os.path.basename(self.filename) != '__init__.py':
                    undefined = all_names.difference(scope)
                    for name in undefined:
                        self.report(messages.UndefinedExport, scope['__all__'].source, name)
            else:
                all_names = []
            for value in scope.values():
                if isinstance(value, Importation):
                    used = value.used or value.name in all_names
                    if not used:
                        messg = messages.UnusedImport
                        self.report(messg, value.source, value.name)
                    for node in value.redefined:
                        if isinstance(self.getParent(node), ast.For):
                            messg = messages.ImportShadowedByLoopVar
                        elif used:
                            continue
                        else:
                            messg = messages.RedefinedWhileUnused
                        self.report(messg, node, value.name, value.source)

    def pushScope(self, scopeClass: type = FunctionScope) -> None:
        self.scopeStack.append(scopeClass())

    def report(self, messageClass: type, *args, **kwargs) -> None:
        self.messages.append(messageClass(self.filename, *args, **kwargs))

    def getParent(self, node: ast.AST) -> ast.AST:
        while True:
            node = node.parent
            if not hasattr(node, 'elts') and (not hasattr(node, 'ctx')):
                return node

    def getCommonAncestor(self, lnode: ast.AST, rnode: ast.AST, stop: ast.AST) -> Union[None, ast.AST]:
        if stop in (lnode, rnode) or not (hasattr(lnode, 'parent') and hasattr(rnode, 'parent')):
            return None
        if lnode is rnode:
            return lnode
        if lnode.depth > rnode.depth:
            return self.getCommonAncestor(lnode.parent, rnode, stop)
        if lnode.depth < rnode.depth:
            return self.getCommonAncestor(lnode, rnode.parent, stop)
        return self.getCommonAncestor(lnode.parent, rnode.parent, stop)

    def descendantOf(self, node: ast.AST, ancestors: List[ast.AST], stop: ast.AST) -> bool:
        for a in ancestors:
            if self.getCommonAncestor(node, a, stop):
                return True
        return False

    def differentForks(self, lnode: ast.AST, rnode: ast.AST) -> bool:
        ancestor = self.getCommonAncestor(lnode, rnode, self.root)
        parts = getAlternatives(ancestor)
        if parts:
            for items in parts:
                if self.descendantOf(lnode, items, ancestor) ^ self.descendantOf(rnode, items, ancestor):
                    return True
        return False

    def addBinding(self, node: ast.AST, value: Binding) -> None:
        for scope in self.scopeStack[::-1]:
            if value.name in scope:
                break
        existing = scope.get(value.name)
        if existing and (not self.differentForks(node, existing.source)):
            parent_stmt = self.getParent(value.source)
            if isinstance(existing, Importation) and isinstance(parent_stmt, ast.For):
                self.report(messages.ImportShadowedByLoopVar, node, value.name, existing.source)
            elif scope is self.scope:
                if isinstance(parent_stmt, ast.comprehension) and (not isinstance(self.getParent(existing.source), (ast.For, ast.comprehension))):
                    self.report(messages.RedefinedInListComp, node, value.name, existing.source)
                elif not existing.used and value.redefines(existing):
                    self.report(messages.RedefinedWhileUnused, node, value.name, existing.source)
            elif isinstance(existing, Importation) and value.redefines(existing):
                existing.redefined.append(node)
        if value.name in self.scope:
            value.used = self.scope[value.name].used
        self.scope[value.name] = value

    def getNodeHandler(self, node_class: type) -> callable:
        try:
            return self._nodeHandlers[node_class]
        except KeyError:
            nodeType = getNodeType(node_class)
        self._nodeHandlers[node_class] = handler = getattr(self, nodeType)
        return handler

    def handleNodeLoad(self, node: ast.AST) -> None:
        name = getNodeName(node)
        if not name:
            return
        try:
            self.scope[name].used = (self.scope, node)
        except KeyError:
            pass
        else:
            return
        scopes = [scope for scope in self.scopeStack[:-1] if isinstance(scope, (FunctionScope, ModuleScope, GeneratorScope))]
        if isinstance(self.scope, GeneratorScope) and scopes[-1] != self.scopeStack[-2]:
            scopes.append(self.scopeStack[-2])
        importStarred = self.scope.importStarred
        for scope in reversed(scopes):
            importStarred = importStarred or scope.importStarred
            try:
                scope[name].used = (self.scope, node)
            except KeyError:
                pass
            else:
                return
        if importStarred or name in self.builtIns:
            return
        if name == '__path__' and os.path.basename(self.filename) == '__init__.py':
            return
        if 'NameError' not in self.exceptHandlers[-1]:
            self.report(messages.UndefinedName, node, name)

    def handleNodeStore(self, node: ast.AST) -> None:
        name = getNodeName(node)
        if not name:
            return
        if isinstance(self.scope, FunctionScope) and name not in self.scope:
            for scope in self.scopeStack[:-1]:
                if not isinstance(scope, (FunctionScope, ModuleScope)):
                    continue
                used = name in scope and scope[name].used
                if used and used[0] is self.scope and (name not in self.scope.globals):
                    self.report(messages.UndefinedLocal, scope[name].used[1], name, scope[name].source)
        parent_stmt = self.getParent(node)
        if isinstance(parent_stmt, (ast.For, ast.comprehension)) or (parent_stmt != node.parent and (not self.isLiteralTupleUnpacking(parent_stmt))):
            binding = Binding(name, node)
        elif name == '__all__' and isinstance(self.scope, ModuleScope):
            binding = ExportBinding(name, node.parent, self.scope)
        else:
            binding = Assignment(name, node)
        self.addBinding(node, binding)

    def handleNodeDelete(self, node: ast.AST) -> None:

        def on_conditional_branch() -> bool:
            current = getattr(node, 'parent', None)
            while current:
                if isinstance(current, (ast.If, ast.While, ast.IfExp)):
                    return True
                current = getattr(current, 'parent', None)
            return False
        name = getNodeName(node)
        if not name:
            return
        if on_conditional_branch():
            return
        if isinstance(self.scope, FunctionScope) and name in self.scope.globals:
            self.scope.globals.remove(name)
        else:
            try:
                del self.scope[name]
            except KeyError:
                self.report(messages.UndefinedName, node, name)

    def handleChildren(self, tree: ast.AST, omit: str = None) -> None:
        for node in iter_child_nodes(tree, omit=omit):
            self.handleNode(node, tree)

    def isLiteralTupleUnpacking(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Assign):
            for child in node.targets + [node.value]:
                if not hasattr(child, 'elts'):
                    return False
            return True

    def isDocstring(self, node: ast.AST) -> bool:
        return isinstance(node, ast.Str) or (isinstance(node, ast.Expr) and isinstance(node.value, ast.Str))

    def getDocstring(self, node: ast.AST) -> Tuple[Union[str, None], Union[int, None]]:
        if isinstance(node, ast.Expr):
            node = node.value
        if not isinstance(node, ast.Str):
            return (None, None)
        doctest_lineno = node.lineno - node.s.count('\n') - 1
        return (node.s, doctest_lineno)

    def handleNode(self, node: ast.AST, parent: ast.AST) -> None:
        if node is None:
            return
        if self.offset and getattr(node, 'lineno', None) is not None:
            node.lineno += self.offset[0]
            node.col_offset += self.offset[1]
        if self.traceTree:
            print('  ' * self.nodeDepth + node.__class__.__name__)
        if self.futuresAllowed and (not (isinstance(node, ast.ImportFrom) or self.isDocstring(node))):
            self.futuresAllowed = False
        self.nodeDepth += 1
        node.depth = self.nodeDepth
        node.parent = parent
        try:
            handler = self.getNodeHandler(node.__class__)
            handler(node)
        finally:
            self.nodeDepth -= 1
        if self.traceTree:
            print('  ' * self.nodeDepth + 'end ' + node.__class__.__name__)
    _getDoctestExamples = doctest.DocTestParser().get_examples

    def handleDoctests(self, node: ast.AST) -> None:
        try:
            docstring, node_lineno = self.getDocstring(node.body[0])
            examples = docstring and self._getDoctestExamples(docstring)
        except (ValueError, IndexError):
            return
        if not examples:
            return
        node_offset = self.offset or (0, 0)
        self.pushScope()
        underscore_in_builtins = '_' in self.builtIns
        if not underscore_in_builtins:
            self.builtIns.add('_')
        for example in examples:
            try:
                tree = compile(example.source, '<doctest>', 'exec', ast.PyCF_ONLY_AST)
            except SyntaxError:
                e = sys.exc_info()[1]
                position = (node_lineno + example.lineno + e.lineno, example.indent + 4 + (e.offset or 0))
                self.report(messages.DoctestSyntaxError, node, position)
            else:
                self.offset = (node_offset[0] + node_lineno + example.lineno, node_offset[1] + example.indent + 4)
                self.handleChildren(tree)
                self.offset = node_offset
        if not underscore_in_builtins:
            self.builtIns.remove('_')
        self.popScope()

    def ignore(self, node: ast.AST) -> None:
        pass

    DELETE = PRINT = FOR = ASYNCFOR = WHILE = IF = WITH = WITHITEM = ASYNCWITH = ASYNCWITHITEM = RAISE = TRYFINALLY = ASSERT