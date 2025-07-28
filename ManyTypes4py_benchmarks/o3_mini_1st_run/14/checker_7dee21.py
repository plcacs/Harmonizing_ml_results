#!/usr/bin/env python
"""
Main module.

Implement the central Checker class.
Also, it models the Bindings and Scopes.
"""
import doctest
import os
import sys
from typing import Any, List, Set, Tuple, Optional, Iterator, Type, Generator, Callable, Dict

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

if PY2:
    def getNodeType(node_class: Type[Any]) -> str:
        return str(unicode(node_class.__name__).upper())
else:
    def getNodeType(node_class: Type[Any]) -> str:
        return node_class.__name__.upper()

if PY32:
    def getAlternatives(n: ast.AST) -> Optional[List[List[ast.stmt]]]:
        if isinstance(n, (ast.If, ast.TryFinally)):
            return [n.body]  # type: ignore
        if isinstance(n, ast.TryExcept):
            return [n.body + n.orelse] + [[hdl] for hdl in n.handlers]
        return None
else:
    def getAlternatives(n: ast.AST) -> Optional[List[List[ast.stmt]]]:
        if isinstance(n, ast.If):
            return [n.body]
        if isinstance(n, ast.Try):
            return [n.body + n.orelse] + [[hdl] for hdl in n.handlers]
        return None

class _FieldsOrder(dict):
    """Fix order of AST node fields."""
    def _get_fields(self, node_class: Type[Any]) -> Tuple[str, ...]:
        fields: Tuple[str, ...] = node_class._fields  # type: ignore
        if 'iter' in fields:
            key_first = 'iter'.find
        elif 'generators' in fields:
            key_first = 'generators'.find
        else:
            key_first = 'value'.find
        return tuple(sorted(fields, key=key_first, reverse=True))

    def __missing__(self, node_class: Type[Any]) -> Tuple[str, ...]:
        self[node_class] = fields = self._get_fields(node_class)
        return fields

def iter_child_nodes(node: ast.AST, omit: Optional[str] = None, _fields_order: _FieldsOrder = _FieldsOrder()) -> Iterator[ast.AST]:
    """
    Yield all direct child nodes of *node*, that is, all fields that
    are nodes and all items of fields that are lists of nodes.
    """
    for name in _fields_order[node.__class__]:
        if name == omit:
            continue
        field = getattr(node, name, None)
        if isinstance(field, ast.AST):
            yield field
        elif isinstance(field, list):
            for item in field:
                yield item

class Binding(object):
    """
    Represents the binding of a value to a name.
    """
    used: Any

    def __init__(self, name: str, source: ast.AST) -> None:
        self.name: str = name
        self.source: ast.AST = source
        self.used: bool = False

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return '<%s object %r from line %r at 0x%x>' % (self.__class__.__name__, self.name, self.source.lineno, id(self))

    def redefines(self, other: Any) -> bool:
        return isinstance(other, Definition) and self.name == other.name

class Definition(Binding):
    """
    A binding that defines a function or a class.
    """
    pass

class Importation(Definition):
    """
    A binding created by an import statement.
    """
    fullName: str
    redefined: List[ast.AST]

    def __init__(self, name: str, source: ast.AST) -> None:
        self.fullName = name
        self.redefined = []
        name = name.split('.')[0]
        super(Importation, self).__init__(name, source)

    def redefines(self, other: Any) -> bool:
        if isinstance(other, Importation):
            return self.fullName == other.fullName
        return isinstance(other, Definition) and self.name == other.name

class Argument(Binding):
    """
    Represents binding a name as an argument.
    """
    pass

class Assignment(Binding):
    """
    Represents binding a name with an explicit assignment.
    """
    pass

class FunctionDefinition(Definition):
    pass

class ClassDefinition(Definition):
    pass

class ExportBinding(Binding):
    """
    A binding created by an C{__all__} assignment.
    """
    names: List[str]

    def __init__(self, name: str, source: ast.AST, scope: 'Scope') -> None:
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
        return '<%s at 0x%x %s>' % (scope_cls, id(self), dict.__repr__(self))

class ClassScope(Scope):
    pass

class FunctionScope(Scope):
    """
    I represent a name scope for a function.
    """
    usesLocals: bool = False
    alwaysUsed: Set[str] = set(['__tracebackhide__', '__traceback_info__', '__traceback_supplement__'])

    def __init__(self) -> None:
        super(FunctionScope, self).__init__()
        self.globals: Set[str] = self.alwaysUsed.copy()
        self.returnValue: Optional[ast.AST] = None
        self.isGenerator: bool = False

    def unusedAssignments(self) -> Generator[Tuple[str, Binding], None, None]:
        """
        Return a generator for the assignments which have not been used.
        """
        for name, binding in self.items():
            if not binding.used and name not in self.globals and (not self.usesLocals) and isinstance(binding, Assignment):
                yield (name, binding)

class GeneratorScope(Scope):
    pass

class ModuleScope(Scope):
    pass

_MAGIC_GLOBALS: List[str] = ['__file__', '__builtins__', 'WindowsError']

def getNodeName(node: ast.AST) -> Optional[str]:
    if hasattr(node, 'id'):
        return getattr(node, 'id', None)
    if hasattr(node, 'name'):
        return getattr(node, 'name', None)
    return None

class Checker(object):
    """
    I check the cleanliness and sanity of Python code.
    """
    nodeDepth: int = 0
    offset: Optional[Tuple[int, int]] = None
    traceTree: bool = False
    builtIns: Set[str] = set(builtin_vars).union(_MAGIC_GLOBALS)
    _customBuiltIns: Optional[str] = os.environ.get('PYFLAKES_BUILTINS')

    def __init__(self, tree: ast.AST, filename: str = '(none)', builtins: Optional[Set[str]] = None, withDoctest: bool = ('PYFLAKES_DOCTEST' in os.environ)) -> None:
        self._nodeHandlers: Dict[Type[Any], Callable[[ast.AST], None]] = {}
        self._deferredFunctions: List[Tuple[Callable[[], None], List[Scope], Optional[Tuple[int, int]]]] = []
        self._deferredAssignments: List[Tuple[Callable[[], None], List[Scope], Optional[Tuple[int, int]]]] = []
        self.deadScopes: List[Scope] = []
        self.messages: List[Any] = []
        self.filename: str = filename
        if builtins:
            self.builtIns = self.builtIns.union(builtins)
        self.withDoctest: bool = withDoctest
        self.scopeStack: List[Scope] = [ModuleScope()]
        self.exceptHandlers: List[Tuple[str, ...]] = [()]
        self.futuresAllowed: bool = True
        self.root: ast.AST = tree
        self.handleChildren(tree)
        self.runDeferred(self._deferredFunctions)
        self._deferredFunctions = []
        self.runDeferred(self._deferredAssignments)
        self._deferredAssignments = []
        del self.scopeStack[1:]
        self.popScope()
        self.checkDeadScopes()

    def deferFunction(self, callable_func: Callable[[], None]) -> None:
        """
        Schedule a function handler to be called just before completion.
        """
        self._deferredFunctions.append((callable_func, self.scopeStack[:], self.offset))

    def deferAssignment(self, callable_func: Callable[[], None]) -> None:
        """
        Schedule an assignment handler to be called just after deferred function handlers.
        """
        self._deferredAssignments.append((callable_func, self.scopeStack[:], self.offset))

    def runDeferred(self, deferred: List[Tuple[Callable[[], None], List[Scope], Optional[Tuple[int,int]]]]) -> None:
        """
        Run the callables in C{deferred} using their associated scope stack.
        """
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
        """
        Look at scopes which have been fully examined and report names in them
        which were imported but unused.
        """
        for scope in self.deadScopes:
            if isinstance(scope.get('__all__'), ExportBinding):
                all_names = set(scope['__all__'].names)  # type: ignore
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

    def pushScope(self, scopeClass: Type[Scope] = FunctionScope) -> None:
        self.scopeStack.append(scopeClass())

    def report(self, messageClass: Any, *args: Any, **kwargs: Any) -> None:
        self.messages.append(messageClass(self.filename, *args, **kwargs))

    def getParent(self, node: ast.AST) -> ast.AST:
        while True:
            node = node.parent  # type: ignore
            if not hasattr(node, 'elts') and (not hasattr(node, 'ctx')):
                return node

    def getCommonAncestor(self, lnode: ast.AST, rnode: ast.AST, stop: ast.AST) -> Optional[ast.AST]:
        if stop in (lnode, rnode) or not (hasattr(lnode, 'parent') and hasattr(rnode, 'parent')):
            return None
        if lnode is rnode:
            return lnode
        if getattr(lnode, 'depth', 0) > getattr(rnode, 'depth', 0):
            return self.getCommonAncestor(getattr(lnode, 'parent'), rnode, stop)
        if getattr(lnode, 'depth', 0) < getattr(rnode, 'depth', 0):
            return self.getCommonAncestor(lnode, getattr(rnode, 'parent'), stop)
        return self.getCommonAncestor(getattr(lnode, 'parent'), getattr(rnode, 'parent'), stop)

    def descendantOf(self, node: ast.AST, ancestors: List[ast.AST], stop: ast.AST) -> bool:
        for a in ancestors:
            if self.getCommonAncestor(node, a, stop):
                return True
        return False

    def differentForks(self, lnode: ast.AST, rnode: ast.AST) -> bool:
        """True, if lnode and rnode are located on different forks of IF/TRY"""
        ancestor = self.getCommonAncestor(lnode, rnode, self.root)
        parts = getAlternatives(ancestor) if ancestor is not None else None
        if parts:
            for items in parts:
                if self.descendantOf(lnode, items, ancestor) ^ self.descendantOf(rnode, items, ancestor):
                    return True
        return False

    def addBinding(self, node: ast.AST, value: Binding) -> None:
        """
        Called when a binding is altered.
        """
        for scope in self.scopeStack[::-1]:
            if value.name in scope:
                break
        existing: Optional[Binding] = scope.get(value.name)  # type: ignore
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

    def getNodeHandler(self, node_class: Type[Any]) -> Callable[[ast.AST], None]:
        try:
            return self._nodeHandlers[node_class]
        except KeyError:
            nodeType: str = getNodeType(node_class)
        self._nodeHandlers[node_class] = handler = getattr(self, nodeType)
        return handler

    def handleNodeLoad(self, node: ast.AST) -> None:
        name: Optional[str] = getNodeName(node)
        if not name:
            return
        try:
            self.scope[name].used = (self.scope, node)
        except KeyError:
            pass
        else:
            return
        scopes: List[Scope] = [scope for scope in self.scopeStack[:-1] if isinstance(scope, (FunctionScope, ModuleScope, GeneratorScope))]
        if isinstance(self.scope, GeneratorScope) and scopes[-1] != self.scopeStack[-2]:
            scopes.append(self.scopeStack[-2])
        importStarred: bool = self.scope.importStarred
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
        name: Optional[str] = getNodeName(node)
        if not name:
            return
        if isinstance(self.scope, FunctionScope) and name not in self.scope:
            for scope in self.scopeStack[:-1]:
                if not isinstance(scope, (FunctionScope, ModuleScope)):
                    continue
                used = name in scope and scope[name].used
                if used and used[0] is self.scope and (name not in self.scope.globals):
                    self.report(messages.UndefinedLocal, scope[name].used[1], name, scope[name].source)
                    break
        parent_stmt: ast.AST = self.getParent(node)
        if isinstance(parent_stmt, (ast.For, ast.comprehension)) or (parent_stmt != node.parent and (not self.isLiteralTupleUnpacking(parent_stmt))):
            binding: Binding = Binding(name, node)
        elif name == '__all__' and isinstance(self.scope, ModuleScope):
            binding = ExportBinding(name, node.parent, self.scope)  # type: ignore
        else:
            binding = Assignment(name, node)
        self.addBinding(node, binding)

    def handleNodeDelete(self, node: ast.AST) -> None:
        def on_conditional_branch() -> bool:
            """
            Return `True` if node is part of a conditional body.
            """
            current: Optional[ast.AST] = getattr(node, 'parent', None)
            while current:
                if isinstance(current, (ast.If, ast.While, ast.IfExp)):
                    return True
                current = getattr(current, 'parent', None)
            return False
        name: Optional[str] = getNodeName(node)
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

    def handleChildren(self, tree: ast.AST, omit: Optional[str] = None) -> None:
        for node in iter_child_nodes(tree, omit=omit):
            self.handleNode(node, tree)

    def isLiteralTupleUnpacking(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Assign):
            for child in node.targets + [node.value]:
                if not hasattr(child, 'elts'):
                    return False
            return True
        return False

    def isDocstring(self, node: ast.AST) -> bool:
        """
        Determine if the given node is a docstring.
        """
        return isinstance(node, ast.Str) or (isinstance(node, ast.Expr) and isinstance(node.value, ast.Str))

    def getDocstring(self, node: ast.AST) -> Tuple[Optional[str], Optional[int]]:
        if isinstance(node, ast.Expr):
            node = node.value
        if not isinstance(node, ast.Str):
            return (None, None)
        doctest_lineno: int = node.lineno - node.s.count('\n') - 1  # type: ignore
        return (node.s, doctest_lineno)

    def handleNode(self, node: Optional[ast.AST], parent: ast.AST) -> None:
        if node is None:
            return
        if self.offset and getattr(node, 'lineno', None) is not None:
            node.lineno += self.offset[0]  # type: ignore
            node.col_offset += self.offset[1]  # type: ignore
        if self.traceTree:
            print('  ' * self.nodeDepth + node.__class__.__name__)
        if self.futuresAllowed and (not (isinstance(node, ast.ImportFrom) or self.isDocstring(node))):
            self.futuresAllowed = False
        self.nodeDepth += 1
        setattr(node, 'depth', self.nodeDepth)
        setattr(node, 'parent', parent)
        try:
            handler: Callable[[ast.AST], None] = self.getNodeHandler(node.__class__)
            handler(node)
        finally:
            self.nodeDepth -= 1
        if self.traceTree:
            print('  ' * self.nodeDepth + 'end ' + node.__class__.__name__)

    _getDoctestExamples = doctest.DocTestParser().get_examples

    def handleDoctests(self, node: ast.AST) -> None:
        try:
            docstring, node_lineno = self.getDocstring(node.body[0])  # type: ignore
            examples = docstring and self._getDoctestExamples(docstring)
        except (ValueError, IndexError):
            return
        if not examples:
            return
        node_offset: Tuple[int, int] = self.offset or (0, 0)
        self.pushScope()
        underscore_in_builtins: bool = '_' in self.builtIns
        if not underscore_in_builtins:
            self.builtIns.add('_')
        for example in examples:
            try:
                tree_compiled: ast.AST = compile(example.source, '<doctest>', 'exec', ast.PyCF_ONLY_AST)
            except SyntaxError:
                e = sys.exc_info()[1]
                position: Tuple[int, int] = (node_lineno + example.lineno + e.lineno, example.indent + 4 + (e.offset or 0))
                self.report(messages.DoctestSyntaxError, node, position)
            else:
                self.offset = (node_offset[0] + node_lineno + example.lineno, node_offset[1] + example.indent + 4)
                self.handleChildren(tree_compiled)
                self.offset = node_offset
        if not underscore_in_builtins:
            self.builtIns.remove('_')
        self.popScope()

    def ignore(self, node: ast.AST) -> None:
        pass

    DELETE = PRINT = FOR = ASYNCFOR = WHILE = IF = WITH = WITHITEM = ASYNCWITH = ASYNCWITHITEM = RAISE = TRYFINALLY = ASSERT = EXEC = EXPR = ASSIGN = handleChildren
    CONTINUE = BREAK = PASS = ignore
    BOOLOP = BINOP = UNARYOP = IFEXP = DICT = SET = COMPARE = CALL = REPR = ATTRIBUTE = SUBSCRIPT = LIST = TUPLE = STARRED = NAMECONSTANT = handleChildren
    NUM = STR = BYTES = ELLIPSIS = ignore
    SLICE = EXTSLICE = INDEX = handleChildren
    LOAD = STORE = DEL = AUGLOAD = AUGSTORE = PARAM = ignore
    AND = OR = ADD = SUB = MULT = DIV = MOD = POW = LSHIFT = RSHIFT = BITOR = BITXOR = BITAND = FLOORDIV = INVERT = NOT = UADD = USUB = EQ = NOTEQ = LT = LTE = GT = GTE = IS = ISNOT = IN = NOTIN = ignore
    COMPREHENSION = KEYWORD = handleChildren

    def GLOBAL(self, node: ast.AST) -> None:
        """
        Keep track of globals declarations.
        """
        global_scope_index: int = 1 if self.withDoctest else 0
        global_scope: Scope = self.scopeStack[global_scope_index]
        if self.scope is not global_scope:
            for node_name in node.names:  # type: ignore
                node_value: Assignment = Assignment(node_name, node)
                self.messages = [m for m in self.messages if not isinstance(m, messages.UndefinedName) and (not m.message_args[0] == node_name)]
                global_scope.setdefault(node_name, node_value)
                node_value.used = (global_scope, node)
                for scope in self.scopeStack[global_scope_index + 1:]:
                    scope[node_name] = node_value
    NONLOCAL = GLOBAL

    def GENERATOREXP(self, node: ast.AST) -> None:
        self.pushScope(GeneratorScope)
        self.handleChildren(node)
        self.popScope()
    if PY2:
        LISTCOMP = handleChildren
    else:
        LISTCOMP = GENERATOREXP
    DICTCOMP = SETCOMP = GENERATOREXP

    def NAME(self, node: ast.AST) -> None:
        """
        Handle occurrence of Name (which can be a load/store/delete access.)
        """
        if isinstance(node.ctx, (ast.Load, ast.AugLoad)):
            self.handleNodeLoad(node)
            if node.id == 'locals' and isinstance(self.scope, FunctionScope) and isinstance(node.parent, ast.Call):  # type: ignore
                self.scope.usesLocals = True
        elif isinstance(node.ctx, (ast.Store, ast.AugStore)):
            self.handleNodeStore(node)
        elif isinstance(node.ctx, ast.Del):
            self.handleNodeDelete(node)
        else:
            raise RuntimeError('Got impossible expression context: %r' % (node.ctx,))

    def RETURN(self, node: ast.AST) -> None:
        if isinstance(self.scope, ClassScope):
            self.report(messages.ReturnOutsideFunction, node)
            return
        if node.value and hasattr(self.scope, 'returnValue') and (not self.scope.returnValue):
            self.scope.returnValue = node.value
        self.handleNode(node.value, node)

    def YIELD(self, node: ast.AST) -> None:
        self.scope.isGenerator = True
        self.handleNode(node.value, node)
    AWAIT = YIELDFROM = YIELD

    def FUNCTIONDEF(self, node: ast.AST) -> None:
        for deco in node.decorator_list:
            self.handleNode(deco, node)
        self.LAMBDA(node)
        self.addBinding(node, FunctionDefinition(node.name, node))  # type: ignore
        if self.withDoctest:
            self.deferFunction(lambda: self.handleDoctests(node))
    ASYNCFUNCTIONDEF = FUNCTIONDEF

    def LAMBDA(self, node: ast.AST) -> None:
        args: List[str] = []
        annotations: List[Optional[Any]] = []
        if PY2:
            def addArgs(arglist: List[Any]) -> None:
                for arg in arglist:
                    if isinstance(arg, ast.Tuple):
                        addArgs(arg.elts)  # type: ignore
                    else:
                        args.append(arg.id)  # type: ignore
            addArgs(node.args.args)  # type: ignore
            defaults: List[ast.AST] = node.args.defaults  # type: ignore
        else:
            for arg in node.args.args + node.args.kwonlyargs:  # type: ignore
                args.append(arg.arg)  # type: ignore
                annotations.append(arg.annotation)
            defaults = node.args.defaults + node.args.kw_defaults  # type: ignore
        is_py3_func: bool = hasattr(node, 'returns')
        for arg_name in ('vararg', 'kwarg'):
            wildcard = getattr(node.args, arg_name, None)  # type: ignore
            if not wildcard:
                continue
            args.append(wildcard if PY33 else wildcard.arg)  # type: ignore
            if is_py3_func:
                if PY33:
                    argannotation = arg_name + 'annotation'
                    annotations.append(getattr(node.args, argannotation))  # type: ignore
                else:
                    annotations.append(wildcard.annotation)
        if is_py3_func:
            annotations.append(getattr(node, 'returns', None))
        if len(set(args)) < len(args):
            for idx, arg in enumerate(args):
                if arg in args[:idx]:
                    self.report(messages.DuplicateArgument, node, arg)
        for child in annotations + defaults:
            if child:
                self.handleNode(child, node)

        def runFunction() -> None:
            self.pushScope()
            for name in args:
                self.addBinding(node, Argument(name, node))
            if isinstance(node.body, list):
                for stmt in node.body:
                    self.handleNode(stmt, node)
            else:
                self.handleNode(node.body, node)
            def checkUnusedAssignments() -> None:
                """
                Check to see if any assignments have not been used.
                """
                for name, binding in self.scope.unusedAssignments():
                    self.report(messages.UnusedVariable, binding.source, name)
            self.deferAssignment(checkUnusedAssignments)
            if PY32:
                def checkReturnWithArgumentInsideGenerator() -> None:
                    """
                    Check to see if there is any return statement with
                    arguments but the function is a generator.
                    """
                    if self.scope.isGenerator and self.scope.returnValue:
                        self.report(messages.ReturnWithArgsInsideGenerator, self.scope.returnValue)
                self.deferAssignment(checkReturnWithArgumentInsideGenerator)
            self.popScope()
        self.deferFunction(runFunction)

    def CLASSDEF(self, node: ast.AST) -> None:
        """
        Check names used in a class definition.
        """
        for deco in node.decorator_list:
            self.handleNode(deco, node)
        for baseNode in node.bases:
            self.handleNode(baseNode, node)
        if not PY2:
            for keywordNode in node.keywords:
                self.handleNode(keywordNode, node)
        self.pushScope(ClassScope)
        if self.withDoctest:
            self.deferFunction(lambda: self.handleDoctests(node))
        for stmt in node.body:
            self.handleNode(stmt, node)
        self.popScope()
        self.addBinding(node, ClassDefinition(node.name, node))  # type: ignore

    def AUGASSIGN(self, node: ast.AST) -> None:
        self.handleNodeLoad(node.target)
        self.handleNode(node.value, node)
        self.handleNode(node.target, node)

    def IMPORT(self, node: ast.AST) -> None:
        for alias in node.names:
            name: str = alias.asname or alias.name
            importation: Importation = Importation(name, node)
            self.addBinding(node, importation)

    def IMPORTFROM(self, node: ast.AST) -> None:
        if node.module == '__future__':
            if not self.futuresAllowed:
                self.report(messages.LateFutureImport, node, [n.name for n in node.names])
        else:
            self.futuresAllowed = False
        for alias in node.names:
            if alias.name == '*':
                self.scope.importStarred = True
                self.report(messages.ImportStarUsed, node, node.module)
                continue
            name = alias.asname or alias.name
            importation = Importation(name, node)
            if node.module == '__future__':
                importation.used = (self.scope, node)
            self.addBinding(node, importation)

    def TRY(self, node: ast.AST) -> None:
        handler_names: List[str] = []
        for handler in node.handlers:  # type: ignore
            if isinstance(handler.type, ast.Tuple):
                for exc_type in handler.type.elts:
                    handler_names.append(getNodeName(exc_type) or '')
            elif handler.type:
                handler_names.append(getNodeName(handler.type) or '')
        self.exceptHandlers.append(tuple(handler_names))
        for child in node.body:
            self.handleNode(child, node)
        self.exceptHandlers.pop()
        self.handleChildren(node, omit='body')

    TRYEXCEPT = TRY

    def EXCEPTHANDLER(self, node: ast.AST) -> None:
        if isinstance(node.name, str):
            self.handleNodeStore(node)
        self.handleChildren(node)

# End of file.
