import doctest
import os
import sys
from typing import Any, Callable, Generator, List, Optional, Tuple, Type, Union

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
    def getNodeType(node_class: Type[ast.AST]) -> str:
        return str(unicode(node_class.__name__).upper())
else:
    def getNodeType(node_class: Type[ast.AST]) -> str:
        return node_class.__name__.upper()

if PY32:
    def getAlternatives(n: ast.AST) -> Optional[List[List[ast.AST]]]:
        if isinstance(n, (ast.If, ast.TryFinally)):
            return [n.body]
        if isinstance(n, ast.TryExcept):
            return [n.body + n.orelse] + [[hdl] for hdl in n.handlers]
else:
    def getAlternatives(n: ast.AST) -> Optional[List[List[ast.AST]]]:
        if isinstance(n, ast.If):
            return [n.body]
        if isinstance(n, ast.Try):
            return [n.body + n.orelse] + [[hdl] for hdl in n.handlers]

class _FieldsOrder(dict):
    def _get_fields(self, node_class: Type[ast.AST]) -> Tuple[str, ...]:
        fields = node_class._fields
        if 'iter' in fields:
            key_first = 'iter'.find
        elif 'generators' in fields:
            key_first = 'generators'.find
        else:
            key_first = 'value'.find
        return tuple(sorted(fields, key=key_first, reverse=True))

    def __missing__(self, node_class: Type[ast.AST]) -> Tuple[str, ...]:
        self[node_class] = fields = self._get_fields(node_class)
        return fields

def iter_child_nodes(node: ast.AST, omit: Optional[str] = None, _fields_order: _FieldsOrder = _FieldsOrder()) -> Generator[ast.AST, None, None]:
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
        return '<%s object %r from line %r at 0x%x>' % (self.__class__.__name__, self.name, self.source.lineno, id(self))

    def redefines(self, other: 'Binding') -> bool:
        return isinstance(other, Definition) and self.name == other.name

class Definition(Binding):
    pass

class Importation(Definition):
    def __init__(self, name: str, source: ast.AST) -> None:
        self.fullName = name
        self.redefined = []
        name = name.split('.')[0]
        super(Importation, self).__init__(name, source)

    def redefines(self, other: Binding) -> bool:
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
    usesLocals: bool = False
    alwaysUsed: set = set(['__tracebackhide__', '__traceback_info__', '__traceback_supplement__'])

    def __init__(self) -> None:
        super(FunctionScope, self).__init__()
        self.globals = self.alwaysUsed.copy()
        self.returnValue = None
        self.isGenerator = False

    def unusedAssignments(self) -> Generator[Tuple[str, Binding], None, None]:
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
        return node.id
    if hasattr(node, 'name'):
        return node.name
    return None

class Checker:
    nodeDepth: int = 0
    offset: Optional[Tuple[int, int]] = None
    traceTree: bool = False
    builtIns: set = set(builtin_vars).union(_MAGIC_GLOBALS)
    _customBuiltIns: Optional[str] = os.environ.get('PYFLAKES_BUILTINS')
    if _customBuiltIns:
        builtIns.update(_customBuiltIns.split(','))
    del _customBuiltIns

    def __init__(self, tree: ast.AST, filename: str = '(none)', builtins: Optional[List[str]] = None, withDoctest: bool = 'PYFLAKES_DOCTEST' in os.environ) -> None:
        self._nodeHandlers: dict = {}
        self._deferredFunctions: List[Tuple[Callable[[], None], List[Scope], Optional[Tuple[int, int]]]] = []
        self._deferredAssignments: List[Tuple[Callable[[], None], List[Scope], Optional[Tuple[int, int]]]] = []
        self.deadScopes: List[Scope] = []
        self.messages: List[messages.Message] = []
        self.filename = filename
        if builtins:
            self.builtIns = self.builtIns.union(builtins)
        self.withDoctest = withDoctest
        self.scopeStack: List[Scope] = [ModuleScope()]
        self.exceptHandlers: List[Tuple[str, ...]] = [()]
        self.futuresAllowed = True
        self.root = tree
        self.handleChildren(tree)
        self.runDeferred(self._deferredFunctions)
        self._deferredFunctions = None
        self.runDeferred(self._deferredAssignments)
        self._deferredAssignments = None
        del self.scopeStack[1:]
        self.popScope()
        self.checkDeadScopes()

    def deferFunction(self, callable: Callable[[], None]) -> None:
        self._deferredFunctions.append((callable, self.scopeStack[:], self.offset))

    def deferAssignment(self, callable: Callable[[], None]) -> None:
        self._deferredAssignments.append((callable, self.scopeStack[:], self.offset))

    def runDeferred(self, deferred: List[Tuple[Callable[[], None], List[Scope], Optional[Tuple[int, int]]]]) -> None:
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

    def pushScope(self, scopeClass: Type[Scope] = FunctionScope) -> None:
        self.scopeStack.append(scopeClass())

    def report(self, messageClass: Type[messages.Message], *args: Any, **kwargs: Any) -> None:
        self.messages.append(messageClass(self.filename, *args, **kwargs))

    def getParent(self, node: ast.AST) -> Optional[ast.AST]:
        while True:
            node = node.parent
            if not hasattr(node, 'elts') and (not hasattr(node, 'ctx')):
                return node

    def getCommonAncestor(self, lnode: ast.AST, rnode: ast.AST, stop: ast.AST) -> Optional[ast.AST]:
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

    def getNodeHandler(self, node_class: Type[ast.AST]) -> Callable[[ast.AST], None]:
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
                    break
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
        return isinstance(node, ast.Str) or (isinstance(node, ast.Expr) and isinstance(node.value, ast.Str))

    def getDocstring(self, node: ast.AST) -> Tuple[Optional[str], Optional[int]]:
        if isinstance(node, ast.Expr):
            node = node.value
        if not isinstance(node, ast.Str):
            return (None, None)
        doctest_lineno = node.lineno - node.s.count('\n') - 1
        return (node.s, doctest_lineno)

    def handleNode(self, node: Optional[ast.AST], parent: ast.AST) -> None:
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

    DELETE = PRINT = FOR = ASYNCFOR = WHILE = IF = WITH = WITHITEM = ASYNCWITH = ASYNCWITHITEM = RAISE = TRYFINALLY = ASSERT = EXEC = EXPR = ASSIGN = handleChildren
    CONTINUE = BREAK = PASS = ignore
    BOOLOP = BINOP = UNARYOP = IFEXP = DICT = SET = COMPARE = CALL = REPR = ATTRIBUTE = SUBSCRIPT = LIST = TUPLE = STARRED = NAMECONSTANT = handleChildren
    NUM = STR = BYTES = ELLIPSIS = ignore
    SLICE = EXTSLICE = INDEX = handleChildren
    LOAD = STORE = DEL = AUGLOAD = AUGSTORE = PARAM = ignore
    AND = OR = ADD = SUB = MULT = DIV = MOD = POW = LSHIFT = RSHIFT = BITOR = BITXOR = BITAND = FLOORDIV = INVERT = NOT = UADD = USUB = EQ = NOTEQ = LT = LTE = GT = GTE = IS = ISNOT = IN = NOTIN = ignore
    COMPREHENSION = KEYWORD = handleChildren

    def GLOBAL(self, node: ast.Global) -> None:
        global_scope_index = 1 if self.withDoctest else 0
        global_scope = self.scopeStack[global_scope_index]
        if self.scope is not global_scope:
            for node_name in node.names:
                node_value = Assignment(node_name, node)
                self.messages = [m for m in self.messages if not isinstance(m, messages.UndefinedName) and (not m.message_args[0] == node_name)]
                global_scope.setdefault(node_name, node_value)
                node_value.used = (global_scope, node)
                for scope in self.scopeStack[global_scope_index + 1:]:
                    scope[node_name] = node_value

    NONLOCAL = GLOBAL

    def GENERATOREXP(self, node: ast.GeneratorExp) -> None:
        self.pushScope(GeneratorScope)
        self.handleChildren(node)
        self.popScope()

    LISTCOMP = handleChildren if PY2 else GENERATOREXP
    DICTCOMP = SETCOMP = GENERATOREXP

    def NAME(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Load, ast.AugLoad)):
            self.handleNodeLoad(node)
            if node.id == 'locals' and isinstance(self.scope, FunctionScope) and isinstance(node.parent, ast.Call):
                self.scope.usesLocals = True
        elif isinstance(node.ctx, (ast.Store, ast.AugStore)):
            self.handleNodeStore(node)
        elif isinstance(node.ctx, ast.Del):
            self.handleNodeDelete(node)
        else:
            raise RuntimeError('Got impossible expression context: %r' % (node.ctx,))

    def RETURN(self, node: ast.Return) -> None:
        if isinstance(self.scope, ClassScope):
            self.report(messages.ReturnOutsideFunction, node)
            return
        if node.value and hasattr(self.scope, 'returnValue') and (not self.scope.returnValue):
            self.scope.returnValue = node.value
        self.handleNode(node.value, node)

    def YIELD(self, node: ast.Yield) -> None:
        self.scope.isGenerator = True
        self.handleNode(node.value, node)

    AWAIT = YIELDFROM = YIELD

    def FUNCTIONDEF(self, node: ast.FunctionDef) -> None:
        for deco in node.decorator_list:
            self.handleNode(deco, node)
        self.LAMBDA(node)
        self.addBinding(node, FunctionDefinition(node.name, node))
        if self.withDoctest:
            self.deferFunction(lambda: self.handleDoctests(node))

    ASYNCFUNCTIONDEF = FUNCTIONDEF

    def LAMBDA(self, node: Union[ast.FunctionDef, ast.Lambda]) -> None:
        args: List[str] = []
        annotations: List[Optional[ast.AST]] = []
        if PY2:
            def addArgs(arglist: List[ast.AST]) -> None:
                for arg in arglist:
                    if isinstance(arg, ast.Tuple):
                        addArgs(arg.elts)
                    else:
                        args.append(arg.id)
            addArgs(node.args.args)
            defaults = node.args.defaults
        else:
            for arg in node.args.args + node.args.kwonlyargs:
                args.append(arg.arg)
                annotations.append(arg.annotation)
            defaults = node.args.defaults + node.args.kw_defaults
        is_py3_func = hasattr(node, 'returns')
        for arg_name in ('vararg', 'kwarg'):
            wildcard = getattr(node.args, arg_name)
            if not wildcard:
                continue
            args.append(wildcard if PY33 else wildcard.arg)
            if is_py3_func:
                if PY33:
                    argannotation = arg_name + 'annotation'
                    annotations.append(getattr(node.args, argannotation))
                else:
                    annotations.append(wildcard.annotation)
        if is_py3_func:
            annotations.append(node.returns)
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
                for name, binding in self.scope.unusedAssignments():
                    self.report(messages.UnusedVariable, binding.source, name)

            self.deferAssignment(checkUnusedAssignments)
            if PY32:
                def checkReturnWithArgumentInsideGenerator() -> None:
                    if self.scope.isGenerator and self.scope.returnValue:
                        self.report(messages.ReturnWithArgsInsideGenerator, self.scope.returnValue)

                self.deferAssignment(checkReturnWithArgumentInsideGenerator)
            self.popScope()

        self.deferFunction(runFunction)

    def CLASSDEF(self, node: ast.ClassDef) -> None:
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
        self.addBinding(node, ClassDefinition(node.name, node))

    def AUGASSIGN(self, node: ast.AugAssign) -> None:
        self.handleNodeLoad(node.target)
        self.handleNode(node.value, node)
        self.handleNode(node.target, node)

    def IMPORT(self, node: ast.Import) -> None:
        for alias in node.names:
            name = alias.asname or alias.name
            importation = Importation(name, node)
            self.addBinding(node, importation)

    def IMPORTFROM(self, node: ast.ImportFrom) -> None:
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

    def TRY(self, node: ast.Try) -> None:
        handler_names: List[str] = []
        for handler in node.handlers:
            if isinstance(handler.type, ast.Tuple):
                for exc_type in handler.type.elts:
                    handler_names.append(getNodeName(exc_type))
            elif handler.type:
                handler_names.append(getNodeName(handler.type))
        self.exceptHandlers.append(handler_names)
        for child in node.body:
            self.handleNode(child, node)
        self.exceptHandlers.pop()
        self.handleChildren(node, omit='body')

    TRYEXCEPT = TRY

    def EXCEPTHANDLER(self, node: ast.ExceptHandler) -> None:
        if isinstance(node.name, str):
            self.handleNodeStore(node)
        self.handleChildren(node)
