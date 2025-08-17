#!/usr/bin/env python3
from __future__ import annotations
import ast
import datetime
import io
import os
import re
import shlex
import shutil
import subprocess
import tokenize
import traceback
from contextlib import contextmanager, ExitStack
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

# The following is a code generator that traverses an AST.
# All visitor methods now have type annotations.

class Generator(ast.NodeVisitor):
    def __init__(self, module: Any) -> None:
        self.module: Any = module
        self.targetFragments: List[str] = []
        self.fragmentIndex: int = 0
        self.indentLevel: int = 0
        self.importHoistMemos: List[Any] = []
        self.allOwnNames: Set[str] = set()
        self.allImportedNames: Set[str] = set()
        self.noskipCodeGeneration: bool = True
        self.conditionalCodeGeneration: bool = True
        self.stripTuple: bool = False
        self.stripTuples: bool = False
        self.allowOperatorOverloading: bool = False
        self.allowJavaScriptMod: bool = False
        self.allowJavaScriptCall: bool = False
        self.allowJavaScriptKeys: bool = False
        self.allowConversionToIterable: bool = False
        self.allowConversionToTruthValue: bool = False
        self.allowKeywordArgs: bool = False
        self.allowDebugMap: bool = False
        self.allowGlobals: bool = False
        self.propertyAccessorList: List[Any] = []
        self.tempIndices: Dict[str, int] = {}
        # other attributes omitted for brevity...
    
    def visitSubExpr(self, node: ast.AST, child: ast.AST) -> None:
        # Some code inside the method...
        self.generic_visit(child)

    def filterId(self, id: str) -> str:
        # Dummy filtering implementation.
        return id

    def tabs(self, indentLevel: Optional[int] = None) -> str:
        if indentLevel is None:
            indentLevel = self.indentLevel
        return "\t" * indentLevel

    def emit(self, fragment: str, *formatter: Any) -> None:
        if (not self.targetFragments or (self.targetFragments and self.targetFragments[self.fragmentIndex - 1].endswith("\n"))):
            self.targetFragments.insert(self.fragmentIndex, self.tabs())
            self.fragmentIndex += 1
        fragment = fragment[:-1].replace("\n", "\n" + self.tabs()) + fragment[-1]
        self.targetFragments.insert(self.fragmentIndex, fragment.format(*formatter).replace("\n", self.lineNrString + "\n"))
        self.fragmentIndex += 1

    def indent(self) -> None:
        self.indentLevel += 1

    def dedent(self) -> None:
        self.indentLevel -= 1

    def inscope(self, node: ast.AST) -> None:
        # Called when entering a new scope
        # For simplicity, we store the node.
        pass

    def descope(self) -> None:
        # Called when leaving a scope.
        pass

    def getScope(self, *nodeTypes: Any) -> Any:
        # Dummy implementation – returns current scope.
        return {}

    def getAdjacentClassScopes(self, inMethod: bool = False) -> Iterable[Any]:
        # Dummy implementation – returns an iterator.
        return iter([])

    def emitComma(self, index: int, blank: bool = True) -> None:
        if index:
            self.emit(", " if blank else ",")

    def emitBeginTruthy(self) -> None:
        self.emit("(",)

    def emitEndTruthy(self) -> None:
        self.emit(")",)

    def adaptLineNrString(self, node: Optional[ast.AST] = None, offset: int = 0) -> None:
        if node and hasattr(node, 'lineno'):
            lineNr = node.lineno + offset
        else:
            lineNr = 1 + offset
        self.lineNrString = str(lineNr).rjust(6, "0")

    def isCommentString(self, statement: ast.stmt) -> bool:
        return isinstance(statement, ast.Expr) and isinstance(getattr(statement, "value", None), ast.Constant) and isinstance(statement.value.value, str)

    def emitBody(self, body: List[ast.stmt]) -> None:
        for statement in body:
            if self.isCommentString(statement):
                pass
            else:
                self.visit(statement)
                self.emit(";\n")

    def emitSubscriptAssign(self, target: ast.Subscript, value: ast.AST, emitPathIndices: Callable[[], None]) -> None:
        if isinstance(target.slice, ast.Index):
            if isinstance(target.slice.value, ast.Tuple):
                self.visit(target.value)
                self.emit(".__setitem__(")
                self.stripTuple = True
                self.visit(target.slice)
                self.emit(", ")
                self.visit(value)
                emitPathIndices()
                self.emit(")")
            elif self.allowOperatorOverloading:
                self.emit("__setitem__(")
                self.visit(target.value)
                self.emit(", ")
                self.visit(target.slice)
                self.emit(", ")
                self.visit(value)
                emitPathIndices()
                self.emit(")")
            else:
                self.expectingNonOverloadedLhsIndex = True
                self.visit(target)
                self.emit(" = ")
                self.visit(value)
                emitPathIndices()
        elif isinstance(target.slice, ast.Slice):
            if self.allowOperatorOverloading:
                self.emit("__setslice__(")
                self.visit(target.value)
                self.emit(", ")
            else:
                self.visit(target.value)
                self.emit(".__setslice__(")
            if target.slice.lower is None:
                self.emit("0")
            else:
                self.visit(target.slice.lower)
            self.emit(", ")
            if target.slice.upper is None:
                self.emit("null")
            else:
                self.visit(target.slice.upper)
            self.emit(", ")
            if target.slice.step:
                self.visit(target.slice.step)
            else:
                self.emit("null")
            self.emit(", ")
            self.visit(value)
            self.emit(")")
        elif isinstance(target.slice, ast.ExtSlice):
            self.visit(target.value)
            self.emit(".__setitem__(")
            self.emit("[")
            for index, dim in enumerate(target.slice.dims):
                self.emitComma(index)
                self.visit(dim)
            self.emit("]")
            self.emit(", ")
            self.visit(value)
            self.emit(")")

    def nextTemp(self, name: str) -> str:
        self.tempIndices[name] = self.tempIndices.get(name, -1) + 1
        return self.getTemp(name)

    def skipTemp(self, name: str) -> None:
        self.skippedTemps.add(self.nextTemp(name))

    def skippedTemp(self, name: str) -> bool:
        return self.getTemp(name) in self.skippedTemps

    def getTemp(self, name: str) -> str:
        return f"__{name}{self.tempIndices.get(name, 0)}__"

    def prevTemp(self, name: str) -> None:
        if self.getTemp(name) in self.skippedTemps:
            self.skippedTemps.remove(self.getTemp(name))
        self.tempIndices[name] = self.tempIndices.get(name, 0) - 1
        if self.tempIndices[name] < 0:
            del self.tempIndices[name]

    def useModule(self, name: str) -> Any:
        self.module.program.importStack[-1][1] = self.lineNr
        return self.module.program.provide(name, filter=self.filterId)

    def isCall(self, node: ast.AST, name: str) -> bool:
        return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name

    def getPragmaFromExpr(self, node: ast.AST) -> Optional[List[ast.AST]]:
        if isinstance(node, ast.Expr) and self.isCall(node.value, '__pragma__'):
            return node.value.args
        return None

    def getPragmaFromIf(self, node: ast.AST) -> Optional[List[ast.AST]]:
        if isinstance(node, ast.If) and self.isCall(node.test, '__pragma__'):
            return node.test.args
        return None

    def visit(self, node: ast.AST) -> Any:
        try:
            self.lineNr = node.lineno  # type: ignore
        except Exception:
            pass

        pragmaInIf = self.getPragmaFromIf(node)
        pragmaInExpr = self.getPragmaFromExpr(node)

        if pragmaInIf:
            # dummy handling for pragma in if
            definedInIf = True
        elif pragmaInExpr:
            if pragmaInExpr[0].s == 'skip':
                self.noskipCodeGeneration = False
            elif pragmaInExpr[0].s == 'noskip':
                self.noskipCodeGeneration = True
            if pragmaInExpr[0].s in ('ifdef', 'ifndef'):
                definedInExpr = bool(eval(compile(ast.Expression(pragmaInExpr[1]), '<string>', 'eval'), {}, {'__envir__': self.module.program.envir}) in self.module.program.symbols)
            if pragmaInExpr[0].s == 'ifdef':
                self.conditionalCodeGeneration = definedInExpr
            elif pragmaInExpr[0].s == 'ifndef':
                self.conditionalCodeGeneration = not definedInExpr
            elif pragmaInExpr[0].s == 'else':
                self.conditionalCodeGeneration = not self.conditionalCodeGeneration
            elif pragmaInExpr[0].s == 'endif':
                self.conditionalCodeGeneration = True

        if self.noskipCodeGeneration and self.conditionalCodeGeneration:
            if pragmaInIf:
                if True:  # definedInIf dummy check
                    self.emitBody(getattr(node, 'body', []))
            else:
                return super().visit(node)
        return None

    def visit_arg(self, node: ast.arg) -> None:
        self.emit(self.filterId(node.arg))

    def visit_arguments(self, node: ast.arguments) -> None:
        self.emit("(")
        for index, arg in enumerate(node.args):
            self.emitComma(index)
            self.visit(arg)
        self.emit(") {{\n")
        self.indent()
        for arg, expr in zip(reversed(node.args), reversed(node.defaults)):
            if expr:
                self.emit("if (typeof {0} == 'undefined' || ({0} != null && {0}.hasOwnProperty('__kwargtrans__'))) {{;\n", self.filterId(arg.arg))
                self.indent()
                self.emit("var {} = ", self.filterId(arg.arg))
                self.visit(expr)
                self.emit(";\n")
                self.dedent()
                self.emit("}};\n")
        for arg, expr in zip(node.kwonlyargs, node.kw_defaults):
            if expr:
                self.emit("var {} = ", self.filterId(arg.arg))
                self.visit(expr)
                self.emit(";\n")
        if self.allowKeywordArgs:
            if node.kwarg:
                self.emit("var {} = dict ();\n", self.filterId(node.kwarg.arg))
            self.emit("if (arguments.length) {{\n")
            self.indent()
            self.emit("var {} = arguments.length - 1;\n", self.nextTemp("ilastarg"))
            self.emit("if (arguments [{}] && arguments [{}].hasOwnProperty('__kwargtrans__')) {{\n", self.getTemp("ilastarg"), self.getTemp("ilastarg"))
            self.indent()
            self.emit("var {} = arguments[{}--];\n", self.nextTemp("allkwargs"), self.getTemp("ilastarg"))
            self.emit("for (var {} in {}) {{\n", self.nextTemp("attrib"), self.getTemp("allkwargs"))
            self.indent()
            self.emit("switch ({}) {{\n", self.getTemp("attrib"))
            self.indent()
            for arg in node.args + node.kwonlyargs:
                self.emit("case '{}': var {} = {}[{}]; break;\n", self.filterId(arg.arg), self.filterId(arg.arg), self.getTemp("allkwargs"), self.getTemp("attrib"))
            if node.kwarg:
                self.emit("default: {}[{}] = {}[{}];\n", self.filterId(node.kwarg.arg), self.getTemp("attrib"), self.getTemp("allkwargs"), self.getTemp("attrib"))
            self.dedent()
            self.emit("}}\n")
            self.prevTemp("allkwargs")
            self.prevTemp("attrib")
            self.dedent()
            self.emit("}}\n")
            if node.kwarg:
                self.emit("delete {}.__kwargtrans__;\n", self.filterId(node.kwarg.arg))
            self.dedent()
            self.emit("}}\n")
            if node.vararg:
                self.emit("var {} = tuple ([].slice.apply (arguments).slice ({}));\n", self.filterId(node.vararg.arg), len(node.args))
            self.prevTemp("ilastarg")
            self.dedent()
            self.emit("else {{\n")
            self.indent()
            if node.vararg:
                self.emit("var {} = tuple ();\n", self.filterId(node.vararg.arg))
            self.dedent()
            self.emit("}}\n")
        else:
            if node.vararg:
                self.emit("var {} = tuple ([].slice.apply (arguments).slice ({}));\n", self.filterId(node.vararg.arg), len(node.args))

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(ast.Assign(targets=[node.target], value=node.value))

    def visit_Assert(self, node: ast.Assert) -> None:
        if hasattr(self.module.program, 'commandArgs') and getattr(self.module.program.commandArgs, 'dassert', False):
            self.emit("assert (")
            self.visit(node.test)
            if node.msg:
                self.emit(", ")
                self.visit(node.msg)
            self.emit(");\n")

    def visit_Assign(self, node: ast.Assign) -> None:
        self.adaptLineNrString(node)
        targetLeafs = (ast.Attribute, ast.Subscript, ast.Name)
        def assignTarget(target: ast.expr, value: ast.AST, pathIndices: List[Any] = []) -> None:
            def emitPathIndices() -> None:
                if pathIndices:
                    self.emit(" ")
                    for pathIndex in pathIndices:
                        self.emit("[{}]".format(pathIndex))
                else:
                    pass
            if isinstance(target, ast.Subscript):
                self.emitSubscriptAssign(target, value, emitPathIndices)
            else:
                isPropertyAssign = False  # Placeholder for property assign check.
                if isPropertyAssign and isinstance(target, ast.Name) and target.id != self.getTemp("left"):
                    self.emit("Object.defineProperty ({}. '{}', ".format(self.getScope().get('node', {}).get('name', ''), target.id))
                    self.visit(value)
                    emitPathIndices()
                    self.emit(")")
                else:
                    if isinstance(target, ast.Name):
                        if type(self.getScope().get('node')) == ast.ClassDef and target.id != self.getTemp("left"):
                            self.emit("{}. ".format(".".join([scope.node.name for scope in self.getAdjacentClassScopes()])))
                        elif target.id in self.getScope().get("nonlocals", set()):
                            pass
                        else:
                            if type(self.getScope().get('node')) == ast.Module:
                                if hasattr(node, 'parentNode') and type(node.parentNode) == ast.Module and target.id not in self.allOwnNames:
                                    self.emit("export ")
                            self.emit("var ")
                    self.visit(target)
                    self.emit(" = ")
                    self.visit(value)
                    emitPathIndices()
        def walkTarget(expr: ast.expr, pathIndices: List[Any]) -> None:
            if isinstance(expr, targetLeafs):
                self.emit(";\n")
                assignTarget(expr, ast.Name(id=self.getTemp("left"), ctx=ast.Load()), pathIndices)
            else:
                pathIndices.append(None)
                if hasattr(expr, "elts"):
                    for index, elt in enumerate(expr.elts):
                        pathIndices[-1] = index
                        walkTarget(elt, pathIndices)
                pathIndices.pop()
        def getIsPropertyAssign(value: Any) -> bool:
            if self.isCall(value, "property"):
                return True
            else:
                try:
                    return getIsPropertyAssign(value.elts[0])
                except Exception:
                    return False
        isPropertyAssign: bool = (type(self.getScope().get("node")) == ast.ClassDef and getIsPropertyAssign(node.value))
        if len(node.targets) == 1 and isinstance(node.targets[0], targetLeafs):
            assignTarget(node.targets[0], node.value)
        else:
            self.visit(ast.Assign(targets=[ast.Name(id=self.nextTemp("left"), ctx=ast.Store())], value=node.value))
            for expr in node.targets:
                walkTarget(expr, [])
            self.prevTemp("left")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, (ast.BinOp, ast.BoolOp, ast.Compare)):
            self.emit("(")
        self.visit(node.value)
        if isinstance(node.value, (ast.BinOp, ast.BoolOp, ast.Compare)):
            self.emit(")")
        self.emit(".{}", self.filterId(node.attr))

    def visit_Await(self, node: ast.Await) -> None:
        self.emit("await ")
        self.visit(node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if self.allowOperatorOverloading:
            rhsFunctionName: str = self.filterId(
                '__ipow__' if isinstance(node.op, ast.Pow) else
                '__imatmul__' if isinstance(node.op, ast.MatMult) else
                ('__ijsmod__' if self.allowJavaScriptMod else '__imod__') if isinstance(node.op, ast.Mod) else
                '__imul__' if isinstance(node.op, ast.Mult) else
                '__idiv__' if isinstance(node.op, ast.Div) else
                '__iadd__' if isinstance(node.op, ast.Add) else
                '__isub__' if isinstance(node.op, ast.Sub) else
                '__ilshift__' if isinstance(node.op, ast.LShift) else
                '__irshift__' if isinstance(node.op, ast.RShift) else
                '__ior__' if isinstance(node.op, ast.BitOr) else
                '__ixor__' if isinstance(node.op, ast.BitXor) else
                '__iand__' if isinstance(node.op, ast.BitAnd) else
                'Never here'
            )
            rhsCall = ast.Call(
                func=ast.Name(id=rhsFunctionName, ctx=ast.Load()),
                args=[node.target, node.value],
                keywords=[]
            )
            if isinstance(node.target, ast.Subscript):
                self.emitSubscriptAssign(node.target, rhsCall)
            else:
                if isinstance(node.target, ast.Name) and node.target.id not in self.getScope().get("nonlocals", set()):
                    self.emit("var ")
                self.visit(node.target)
                self.emit(" = ")
                self.visit(rhsCall)
        elif (isinstance(node.op, (ast.FloorDiv, ast.MatMult, ast.Pow)) or
              (isinstance(node.op, ast.Mod) and not self.allowJavaScriptMod) or
              (isinstance(node.target, ast.Subscript) and isinstance(node.target.slice, ast.Tuple))):
            self.visit(ast.Assign(targets=[node.target], value=ast.BinOp(left=node.target, op=node.op, right=node.value)))
        else:
            self.expectingNonOverloadedLhsIndex = True
            self.visit(node.target)
            if isinstance(node.value, ast.Constant) and node.value.value == 1:
                if isinstance(node.op, ast.Add):
                    self.emit("++")
                    return
                elif isinstance(node.op, ast.Sub):
                    self.emit("--")
                    return
            elif isinstance(node.value, ast.UnaryOp) and isinstance(node.value.operand, ast.Constant) and node.value.operand.value == 1:
                if isinstance(node.op, ast.Add):
                    if isinstance(node.value.op, ast.UAdd):
                        self.emit("++")
                        return
                    elif isinstance(node.value.op, ast.USub):
                        self.emit("--")
                        return
                elif isinstance(node.op, ast.Sub):
                    if isinstance(node.value.op, ast.UAdd):
                        self.emit("--")
                        return
                    elif isinstance(node.value.op, ast.USub):
                        self.emit("++")
                        return
            self.emit(" {}= ", self.operators[type(node.op)][0])
            self.visit(node.value)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.FloorDiv):
            if self.allowOperatorOverloading:
                self.emit("__floordiv__ (")
                self.visitSubExpr(node, node.left)
                self.emit(", ")
                self.visitSubExpr(node, node.right)
                self.emit(")")
            else:
                self.emit("Math.floor (")
                self.visitSubExpr(node, node.left)
                self.emit(" / ")
                self.visitSubExpr(node, node.right)
                self.emit(")")
        elif (isinstance(node.op, (ast.Pow, ast.MatMult)) or
              (isinstance(node.op, ast.Mod) and (self.allowOperatorOverloading or not self.allowJavaScriptMod)) or
              (isinstance(node.op, (ast.Mult, ast.Div, ast.Add, ast.Sub, ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd)) and self.allowOperatorOverloading)):
            self.emit("{} (".format(self.filterId(
                '__floordiv__' if isinstance(node.op, ast.FloorDiv) else
                ('__pow__' if self.allowOperatorOverloading else 'Math.pow') if isinstance(node.op, ast.Pow) else
                '__matmul__' if isinstance(node.op, ast.MatMult) else
                ('__jsmod__' if self.allowJavaScriptMod else '__mod__') if isinstance(node.op, ast.Mod) else
                '__mul__' if isinstance(node.op, ast.Mult) else
                '__truediv__' if isinstance(node.op, ast.Div) else
                '__add__' if isinstance(node.op, ast.Add) else
                '__sub__' if isinstance(node.op, ast.Sub) else
                '__lshift__' if isinstance(node.op, ast.LShift) else
                '__rshift__' if isinstance(node.op, ast.RShift) else
                '__or__' if isinstance(node.op, ast.BitOr) else
                '__xor__' if isinstance(node.op, ast.BitXor) else
                '__and__' if isinstance(node.op, ast.BitAnd) else
                'Never here'
            )))
            self.visit(node.left)
            self.emit(", ")
            self.visit(node.right)
            self.emit(")")
        else:
            self.visitSubExpr(node, node.left)
            self.emit(" {} ".format(self.operators[type(node.op)][0]))
            self.visitSubExpr(node, node.right)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        for index, value in enumerate(node.values):
            if index:
                self.emit(" {} ".format(self.operators[type(node.op)][0]))
            if index < len(node.values) - 1:
                self.emitBeginTruthy()
            self.visitSubExpr(node, value)
            if index < len(node.values) - 1:
                self.emitEndTruthy()

    def visit_Break(self, node: ast.Break) -> None:
        if not self.skippedTemp("break"):
            self.emit("{} = true;\n", self.getTemp("break"))
        self.emit("break")

    def visit_Call(self, node: ast.Call, dataClassArgDict: Optional[Dict[str, Any]] = None) -> None:
        self.adaptLineNrString(node)
        def emitKwargTrans() -> None:
            self.emit("__kwargtrans__ (")
            hasSeparateKeyArgs = False
            hasKwargs = False
            for keyword in node.keywords:
                if keyword.arg:
                    hasSeparateKeyArgs = True
                else:
                    hasKwargs = True
                    break
            if hasSeparateKeyArgs:
                if hasKwargs:
                    self.emit("__mergekwargtrans__ (")
                self.emit("{{")
            for keywordIndex, keyword in enumerate(node.keywords):
                if keyword.arg:
                    self.emitComma(keywordIndex)
                    self.emit("{}: ", self.filterId(keyword.arg))
                    self.visit(keyword.value)
                else:
                    if hasSeparateKeyArgs:
                        self.emit("}}, ")
                    self.visit(keyword.value)
            if hasSeparateKeyArgs:
                if hasKwargs:
                    self.emit(")")
                else:
                    self.emit("}}")
            self.emit(")")
        def include(fileName: str) -> str:
            searchedIncludePaths: List[str] = []
            for searchDir in self.module.program.moduleSearchDirs:
                filePath = f"{searchDir}/{fileName}"
                if os.path.isfile(filePath):
                    includedCode = tokenize.open(filePath).read()
                    if fileName.endswith(".js"):
                        includedCode = includedCode  # digest logic omitted for brevity
                    return includedCode
                else:
                    searchedIncludePaths.append(filePath)
            else:
                raise Exception(f"\n\tAttempt to include file: {node.args[0]}... Searched: {searchedIncludePaths}")
        if isinstance(node.func, ast.Name):
            if node.func.id == "type":
                self.emit("py_typeof (")
                self.visit(node.args[0])
                self.emit(")")
                return
            elif node.func.id == "property":
                self.emit("{0}.call ({1}, {1}.{2}", node.func.id, self.getScope().get("node", {}).get("name", ""), self.filterId(node.args[0].id))
                if len(node.args) > 1:
                    self.emit(", {}.{}", self.getScope().get("node", {}).get("name", ""), node.args[1].id)
                self.emit(")")
                return
            elif node.func.id == "globals":
                self.emit("__all__")
                return
            elif node.func.id == "__pragma__":
                if node.args[0].s == "alias":
                    self.module.program.envir.aliases.insert(0, (node.args[1].s, node.args[2].s))
                elif node.args[0].s == "noalias":
                    if len(node.args) == 1:
                        self.module.program.envir.aliases = []
                    else:
                        for index in reversed(range(len(self.module.program.envir.aliases))):
                            if self.module.program.envir.aliases[index][0] == node.args[1].s:
                                self.module.program.envir.aliases.pop(index)
                elif node.args[0].s == "noanno":
                    self.allowDebugMap = False
                elif node.args[0].s == "fcall":
                    self.allowMemoizeCalls = True
                elif node.args[0].s == "nofcall":
                    self.allowMemoizeCalls = False
                elif node.args[0].s == "docat":
                    self.allowDocAttribs = True
                elif node.args[0].s == "nodocat":
                    self.allowDocAttribs = False
                elif node.args[0].s == "iconv":
                    self.allowConversionToIterable = True
                elif node.args[0].s == "noiconv":
                    self.allowConversionToIterable = False
                elif node.args[0].s == "jsiter":
                    self.allowJavaScriptIter = True
                elif node.args[0].s == "nojsiter":
                    self.allowJavaScriptIter = False
                elif node.args[0].s == "jscall":
                    self.allowJavaScriptCall = True
                elif node.args[0].s == "nojscall":
                    self.allowJavaScriptCall = False
                elif node.args[0].s == "jskeys":
                    self.allowJavaScriptKeys = True
                elif node.args[0].s == "nojskeys":
                    self.allowJavaScriptKeys = False
                elif node.args[0].s == "keycheck":
                    self.allowKeyCheck = True
                elif node.args[0].s == "nokeycheck":
                    self.allowKeyCheck = False
                elif node.args[0].s == "jsmod":
                    self.allowJavaScriptMod = True
                elif node.args[0].s == "nojsmod":
                    self.allowJavaScriptMod = False
                elif node.args[0].s == "gsend":
                    self.replaceSend = True
                elif node.args[0].s == "nogsend":
                    self.replaceSend = False
                elif node.args[0].s == "tconv":
                    self.allowConversionToTruthValue = True
                elif node.args[0].s == "notconv":
                    self.allowConversionToTruthValue = False
                elif node.args[0].s == "run":
                    pass
                elif node.args[0].s == "norun":
                    pass
                elif node.args[0].s == "js":
                    try:
                        try:
                            code = node.args[1].s.format(*[
                                eval(compile(ast.Expression(arg), '<string>', 'eval'), {}, {'__include__': include})
                                for arg in node.args[2:]
                            ])
                        except Exception:
                            code = node.args[2].s
                        for line in code.split("\n"):
                            self.emit("{}\n", line)
                    except Exception:
                        print(traceback.format_exc())
                elif node.args[0].s == "xtrans":
                    try:
                        sourceCode = node.args[2].s.format(*[
                            eval(compile(ast.Expression(arg), '<string>', 'eval'), {}, {'__include__': include})
                            for arg in node.args[3:]
                        ])
                        workDir = "."
                        for keyword in node.keywords:
                            if keyword.arg == "cwd":
                                workDir = keyword.value.s
                        process = subprocess.Popen(shlex.split(node.args[1].s), stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=workDir)
                        process.stdin.write(sourceCode.encode("utf8"))
                        process.stdin.close()
                        while process.returncode is None:
                            process.poll()
                        targetCode = process.stdout.read().decode("utf8").replace("\r\n", "\n")
                        for line in targetCode.split("\n"):
                            self.emit("{}\n", line)
                    except Exception:
                        print(traceback.format_exc())
                elif node.args[0].s == "xglobs":
                    self.allowGlobals = True
                elif node.args[0].s == "noxglobs":
                    self.allowGlobals = False
                elif node.args[0].s == "kwargs":
                    self.allowKeywordArgs = True
                elif node.args[0].s == "nokwargs":
                    self.allowKeywordArgs = False
                elif node.args[0].s == "opov":
                    self.allowOperatorOverloading = True
                elif node.args[0].s == "noopov":
                    self.allowOperatorOverloading = False
                elif node.args[0].s == "redirect":
                    if node.args[1].s == "stdout":
                        self.emit("__stdout__ = '{}'", node.args[2])
                elif node.args[0].s == "noredirect":
                    if node.args[1].s == "stdout":
                        self.emit("__stdout__ = '__console__'")
                elif node.args[0].s in ('skip', 'noskip', 'defined', 'ifdef', 'ifndef', 'else', 'endif'):
                    pass
                elif node.args[0].s == "xpath":
                    self.module.program.moduleSearchDirs[1:1] = [elt.s for elt in node.args[1].elts]
                else:
                    raise Exception(f"\n\tUnknown pragma: {node.args[0].s}")
                return
            elif node.func.id == "__new__":
                self.emit("new ")
                self.visit(node.args[0])
                return
            elif node.func.id == "__typeof__":
                self.emit("typeof ")
                self.visit(node.args[0])
                return
            elif node.func.id == "__preinc__":
                self.emit("++")
                self.visit(node.args[0])
                return
            elif node.func.id == "__postinc__":
                self.visit(node.args[0])
                self.emit("++")
                return
            elif node.func.id == "__predec__":
                self.emit("--")
                self.visit(node.args[0])
                return
            elif node.func.id == "__postdec__":
                self.visit(node.args[0])
                self.emit("--")
                return
        elif isinstance(node.func, ast.Attribute) and node.func.attr == "conjugate":
            try:
                self.visit(ast.Call(
                    func=ast.Name(id="__conj__", ctx=ast.Load()),
                    args=[node.func.value],
                    keywords=[]
                ))
                return
            except Exception:
                print(traceback.format_exc())
        elif (isinstance(node.func, ast.Attribute) and self.replaceSend and node.func.attr == "send"):
            self.emit("(function () {{return ")
            self.visit(ast.Attribute(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id=node.func.value.id, ctx=ast.Load()),
                        attr="js_next",
                        ctx=ast.Load()
                    ),
                    args=node.args,
                    keywords=node.keywords
                ),
                attr="value",
                ctx=ast.Load()
            ))
            self.emit("}}) ()")
            return
        elif (isinstance(node.func, ast.Attribute) and
              isinstance(node.func.value, ast.Call) and
              isinstance(node.func.value.func, ast.Name) and
              node.func.value.func.id == "super"):
            if node.func.value.args or node.func.value.keywords:
                raise Exception(f"\n\tBuilt in function 'super' with arguments not supported")
            else:
                self.visit(ast.Call(
                    func=ast.Call(
                        func=ast.Name(id="__super__", ctx=ast.Load()),
                        args=[
                            ast.Name(id=".".join([scope.node.name for scope in self.getAdjacentClassScopes(True)]), ctx=ast.Load()),
                            ast.Constant(value=node.func.attr)
                        ],
                        keywords=[]
                    ),
                    args=[ast.Name(id="self", ctx=ast.Load())] + node.args,
                    keywords=node.keywords
                ))
                return
        if self.allowOperatorOverloading and not (isinstance(node.func, ast.Name) and node.func.id == "__call__"):
            if isinstance(node.func, ast.Attribute):
                self.emit("(function () {{\n")
                self.inscope(ast.FunctionDef())
                self.indent()
                self.emit("var {} = ", self.nextTemp("accu"))
                self.visit(node.func.value)
                self.emit(";\n")
                self.emit("return ")
                self.visit(ast.Call(
                    func=ast.Name(id="__call__", ctx=ast.Load()),
                    args=[
                        ast.Attribute(
                            value=ast.Name(id=self.getTemp("accu"), ctx=ast.Load()),
                            attr=node.func.attr,
                            ctx=ast.Load()
                        ),
                        ast.Name(id=self.getTemp("accu"), ctx=ast.Load())
                    ] + node.args,
                    keywords=node.keywords
                ))
                self.emit(";\n")
                self.prevTemp("accu")
                self.dedent()
                self.descope()
                self.emit("}}) ()")
            else:
                self.visit(ast.Call(
                    func=ast.Name(id="__call__", ctx=ast.Load()),
                    args=[node.func, ast.Constant(value=None)] + node.args,
                    keywords=node.keywords
                ))
            return
        if dataClassArgDict is not None:
            import copy
            dataClassArgTuple: List[Tuple[str, bool]] = copy.deepcopy([("init", True), ("repr", True), ("eq", True), ("order", False), ("unsafe_hash", False), ("frozen", False)])
            for index, expr in enumerate(node.args):
                value = None
                if expr == ast.Constant:
                    value = True if expr.value == "True" else False if expr.value == "False" else None
                if value is not None:
                    dataClassArgTuple[index] = (dataClassArgTuple[index][0], value)
                else:
                    raise Exception("Arguments to @dataclass can only be constants True or False")
            dataClassArgDict.update(dict(dataClassArgTuple))
            for keyword in node.keywords:
                dataClassArgDict[keyword.arg] = keyword.value
            return
        self.visit(node.func)
        self.emit(" (")
        for index, expr in enumerate(node.args):
            self.emitComma(index)
            if isinstance(expr, ast.Starred):
                self.emit("...")
            self.visit(expr)
        if node.keywords:
            self.emitComma(len(node.args))
            emitKwargTrans()
        self.emit(")")

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.adaptLineNrString(node)
        if type(self.getScope().get("node")) == ast.Module:
            self.emit("export var {} = ".format(self.filterId(node.name)))
            self.allOwnNames.add(node.name)
        elif type(self.getScope().get("node")) == ast.ClassDef:
            self.emit("\n{}:".format(self.filterId(node.name)))
        else:
            self.emit("var {} =".format(self.filterId(node.name)))
        isDataClass: bool = False
        if node.decorator_list:
            if isinstance(node.decorator_list[-1], ast.Name) and node.decorator_list[-1].id == "dataclass":
                isDataClass = True
                dataClassArgDict: Dict[str, Any] = dict([("init", True), ("repr", True), ("eq", True), ("order", False), ("unsafe_hash", False), ("frozen", False)])
                node.decorator_list.pop()
            elif isinstance(node.decorator_list[-1], ast.Call) and node.decorator_list[-1].func.id == "dataclass":
                isDataClass = True
                dataClassArgDict = {}
                self.visit_Call(node.decorator_list.pop(), dataClassArgDict)
        decoratorsUsed: int = 0
        if node.decorator_list:
            self.emit(" ")
            if self.allowOperatorOverloading:
                self.emit("__call__ (")
            for decorator in node.decorator_list:
                if decoratorsUsed > 0:
                    self.emit(" (")
                self.visit(decorator)
                decoratorsUsed += 1
            if self.allowOperatorOverloading:
                self.emit(", null, ")
            else:
                self.emit(" (")
        self.emit(" __class__ ('{}', [", self.filterId(node.name))
        if node.bases:
            for index, expr in enumerate(node.bases):
                try:
                    self.emitComma(index)
                    self.visit(expr)
                except Exception as exception:
                    raise Exception("\n\tInvalid base class") from exception
        else:
            self.emit("object")
        self.emit("], {{")
        self.inscope(node)
        self.indent()
        self.emit("\n__module__: __name__,")
        inlineAssigns: List[Any] = []
        propertyAssigns: List[Any] = []
        initAssigns: List[Any] = []
        delayedAssigns: List[Any] = []
        reprAssigns: List[Any] = []
        compareAssigns: List[Any] = []
        index: int = 0
        if isDataClass:
            initHoistFragmentIndex: int = self.fragmentIndex
            initHoistIndentLevel: int = self.indentLevel
        for statement in node.body:
            if self.isCommentString(statement):
                pass
            elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.emitComma(index, False)
                self.visit(statement)
                index += 1
            elif isinstance(statement, ast.Assign):
                if len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
                    if isinstance(statement.value, ast.Call) and isinstance(statement.value.func, ast.Name) and statement.value.func.id == "property":
                        propertyAssigns.append(statement)
                    else:
                        inlineAssigns.append(statement)
                        self.emitComma(index, False)
                        self.emit("\n{}: ", self.filterId(statement.targets[0].id))
                        self.visit(statement.value)
                        self.adaptLineNrString(statement)
                        index += 1
                else:
                    delayedAssigns.append(statement)
            elif isinstance(statement, ast.AnnAssign):
                if isinstance(statement.value, ast.Call) and isinstance(statement.value.func, ast.Name) and statement.value.func.id == "property":
                    propertyAssigns.append(statement)
                    if isDataClass:
                        reprAssigns.append(statement)
                        compareAssigns.append(statement)
                elif isDataClass and isinstance(statement.annotation, ast.Name) and statement.annotation.id != "ClassVar":
                    inlineAssigns.append(statement)
                    initAssigns.append(statement)
                    reprAssigns.append(statement)
                    compareAssigns.append(statement)
                    self.emitComma(index, False)
                    self.emit("\n{}: ", self.filterId(statement.target.id))
                    self.visit(statement.value)
                    self.adaptLineNrString(statement)
                    index += 1
                elif isinstance(statement.target, ast.Name):
                    try:
                        inlineAssigns.append(statement)
                        self.emitComma(index, False)
                        self.emit("\n{}: ", self.filterId(statement.target.id))
                        self.visit(statement.value)
                        self.adaptLineNrString(statement)
                        index += 1
                    except Exception:
                        print(traceback.format_exc())
                else:
                    delayedAssigns.append(statement)
            elif self.getPragmaFromExpr(statement):
                self.visit(statement)
        self.dedent()
        self.emit("\n}}")
        if node.keywords:
            if node.keywords[0].arg == "metaclass":
                self.emit(", ")
                self.visit(node.keywords[0].value)
            else:
                raise Exception("\n\tUnknown keyword argument {} definition of class {}".format(node.keywords[0].arg, node.name))
        self.emit(")")
        if decoratorsUsed:
            self.emit(")" * decoratorsUsed)
        if self.allowDocAttribs:
            docString: Optional[str] = ast.get_docstring(node)
            if docString:
                self.emit(" .__setdoc__ ('{}')", docString.replace("\n", "\\n "))
        if isDataClass:
            nrOfFragmentsToJump: int = self.fragmentIndex - initHoistFragmentIndex
            self.fragmentIndex = initHoistFragmentIndex
            originalIndentLevel: int = self.indentLevel
            self.indentLevel = initHoistIndentLevel
            initArgs = [(initAssign.targets[0] if isinstance(initAssign, ast.Assign) else initAssign.target).id for initAssign in initAssigns]
            reprNames = [(reprAssign.targets[0] if isinstance(reprAssign, ast.Assign) else reprAssign.target).id for reprAssign in reprAssigns]
            compareNames = [(compareAssign.targets[0] if isinstance(compareAssign, ast.Assign) else compareAssign.target).id for compareAssign in compareAssigns]
            if dataClassArgDict["repr"]:
                originalAllowKeywordArgs = self.allowKeywordArgs
                self.allowKeywordArgs = True
                self.visit(ast.FunctionDef(
                    name="__init__",
                    args=ast.arguments(args=[ast.arg(arg="self", annotation=None)], vararg=ast.arg(arg="args", annotation=None), kwonlyargs=[], kw_defaults=[], kwarg=ast.arg(arg="kwargs", annotation=None), defaults=[]),
                    body=[ast.Expr(value=ast.Call(func=ast.Name(id="__pragma__", ctx=ast.Load()), args=[ast.Constant(value="js"), ast.Constant(value="{}"), ast.Constant(value='''\nlet names = self.__initfields__.values ();\nfor (let arg of args) {\n    self [names.next ().value] = arg;\n}\nfor (let name of kwargs.py_keys ()) {\n    self [name] = kwargs [name];\n}\n'''.strip())], keywords=[])],
                    decorator_list=[],
                    returns=None,
                    type_comment=None
                ))
                self.emit(",")
                self.allowKeywordArgs = originalAllowKeywordArgs
            if dataClassArgDict["repr"]:
                self.visit(ast.FunctionDef(
                    name="__repr__",
                    args=ast.arguments(args=[ast.arg(arg="self", annotation=None)], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
                    body=[ast.Expr(value=ast.Call(func=ast.Name(id="__pragma__", ctx=ast.Load()), args=[ast.Constant(value="js"), ast.Constant(value="{}"), ast.Constant(value='''\nlet names = self.__reprfields__.values ();\nlet fields = [];\nfor (let name of names) {{\n    fields.push (name + '=' + repr (self [name]));\n}}\nreturn  self.__name__ + '(' + ', '.join (fields) + ')'\n'''.strip())], keywords=[]))],
                    decorator_list=[],
                    returns=None,
                    type_comment=None
                ))
                self.emit(",")
            comparatorNames: List[str] = []
            if "eq" in dataClassArgDict:
                comparatorNames += ["__eq__", "__ne__"]
            if "order" in dataClassArgDict:
                comparatorNames += ["__lt__", "__le__", "__gt__", "__ge__"]
            for comparatorName in comparatorNames:
                self.visit(ast.FunctionDef(
                    name=comparatorName,
                    args=ast.arguments(args=[ast.arg(arg="self", annotation=None), ast.arg(arg="other", annotation=None)], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
                    body=[ast.Expr(value=ast.Call(func=ast.Name(id="__pragma__", ctx=ast.Load()), args=[ast.Constant(value="js"), ast.Constant(value="{}"), ast.Constant(value=(f'''\nlet names = self.__comparefields__.values ();\nlet selfFields = [];\nlet otherFields = [];\nfor (let name of names) {{\n    selfFields.push (self [name]);\n    otherFields.push (other [name]);\n}}\nreturn list (selfFields).{comparatorName}(list (otherFields));\n''').strip())], keywords=[]))],
                    decorator_list=[]
                ))
                self.emit(",")
            self.fragmentIndex += nrOfFragmentsToJump
            self.indentLevel = originalIndentLevel
        for assign in delayedAssigns + propertyAssigns:
            self.emit(";\n")
            self.visit(assign)
        self.mergeList.append({
            "className": ".".join([scope.node.name for scope in self.getAdjacentClassScopes()]),
            "isDataClass": isDataClass,
            "reprAssigns": reprAssigns,
            "compareAssigns": compareAssigns,
            "initAssigns": initAssigns
        })
        self.descope()
        def emitMerges() -> None:
            def emitMerge(merge: Any) -> None:
                if merge["isDataClass"]:
                    self.emit("\nfor (let aClass of {}.__bases__) {{\n", self.filterId(merge["className"]))
                    self.indent()
                    self.emit("__mergefields__ ({}, aClass);\n", self.filterId(merge["className"]))
                    self.dedent()
                    self.emit("}}")
                    self.emit(";\n__mergefields__ ({}, {{", self.filterId(merge["className"]))
                    self.emit("__reprfields__: new Set ([{}]), ", ", ".join(f"'{name}'" for name in merge["reprAssigns"]))
                    self.emit("__comparefields__: new Set ([{}]), ", ", ".join(f"'{name}'" for name in merge["compareAssigns"]))
                    self.emit("__initfields__: new Set ([{}])", ", ".join(f"'{name}'" for name in merge["initAssigns"]))
                    self.emit("}})")
            for merge in self.mergeList:
                emitMerge(merge)
            self.mergeList = []
        def emitProperties() -> None:
            def emitProperty(className: str, propertyName: str, getterName: str, setterName: Optional[str] = None) -> None:
                self.emit("\nObject.defineProperty ({}, '{}', ", className, propertyName)
                if setterName:
                    self.emit("property.call ({0}, {0}.{1}, {0}.{2})", className, getterName, setterName)
                else:
                    self.emit("property.call ({0}, {0}.{1})", className, getterName)
                self.emit(");")
            if self.propertyAccessorList:
                self.emit(";")
            while self.propertyAccessorList:
                propertyAccessor = self.propertyAccessorList.pop()
                className = propertyAccessor.className
                functionName = propertyAccessor.functionName
                propertyName = functionName[5:]
                isGetter = functionName[:5] == "_get_"
                for propertyAccessor2 in self.propertyAccessorList:
                    className2 = propertyAccessor2.className
                    functionName2 = propertyAccessor2.functionName
                    propertyName2 = functionName2[5:]
                    isGetter2 = functionName2[:5] == "_get_"
                    if className == className2 and propertyName == propertyName2 and isGetter != isGetter2:
                        self.propertyAccessorList.remove(propertyAccessor2)
                        if isGetter:
                            emitProperty(className, propertyName, functionName, functionName2)
                        else:
                            emitProperty(className, propertyName, functionName2, functionName)
                        break
                else:
                    if isGetter:
                        emitProperty(className, propertyName, functionName)
                    else:
                        raise Exception("\n\tProperty setter declared without getter\n")
        if type(self.getScope().get("node")) != ast.ClassDef:
            emitProperties()
            emitMerges()
    def visit_Compare(self, node: ast.Compare) -> None:
        if len(node.comparators) > 1:
            self.emit("(")
        left = node.left
        for index, (op, right) in enumerate(zip(node.ops, node.comparators)):
            if index:
                self.emit(" && ")
            if isinstance(op, (ast.In, ast.NotIn)) or (self.allowOperatorOverloading and isinstance(op, (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE))):
                self.emit("{} (".format(self.filterId(
                    '__in__' if isinstance(op, ast.In) else
                    '!__in__' if isinstance(op, ast.NotIn) else
                    '__eq__' if isinstance(op, ast.Eq) else
                    '__ne__' if isinstance(op, ast.NotEq) else
                    '__lt__' if isinstance(op, ast.Lt) else
                    '__le__' if isinstance(op, ast.LtE) else
                    '__gt__' if isinstance(op, ast.Gt) else
                    '__ge__' if isinstance(op, ast.GtE) else
                    'Never here'
                )))
                self.visitSubExpr(node, left)
                self.emit(", ")
                self.visitSubExpr(node, right)
                self.emit(")")
            else:
                self.visitSubExpr(node, left)
                self.emit(" {0} ".format(self.operators[type(op)][0]))
                self.visitSubExpr(node, right)
            left = right
        if len(node.comparators) > 1:
            self.emit(")")
    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            self.emit("{}", repr(node.value))
        elif isinstance(node.value, bytes):
            self.emit("bytes ('{}')", node.value.decode("ASCII"))
        elif isinstance(node.value, complex):
            self.emit("complex (0, {})".format(node.value.imag))
        elif isinstance(node.value, (float, int)):
            self.emit("{}".format(node.value))
        else:
            self.emit(self.module.program.envir.nameConsts.get(node.value, str(node.value)))
    def visit_Continue(self, node: ast.Continue) -> None:
        self.emit("continue")
    def visit_Delete(self, node: ast.Delete) -> None:
        for expr in node.targets:
            if not isinstance(expr, ast.Name):
                self.emit("delete ")
                self.visit(expr)
                self.emit(";\n")
    def visit_Dict(self, node: ast.Dict) -> None:
        if not self.allowJavaScriptKeys:
            for key in node.keys:
                if not isinstance(key, ast.Constant):
                    self.emit("dict ([")
                    for index, (key, value) in enumerate(zip(node.keys, node.values)):
                        self.emitComma(index)
                        self.emit("[")
                        self.visit(key)
                        self.emit(", ")
                        self.visit(value)
                        self.emit("]")
                    self.emit("])")
                    return
        if self.allowJavaScriptIter:
            self.emit("{{")
        else:
            self.emit("dict ({{")
        for index, (key, value) in enumerate(zip(node.keys, node.values)):
            self.emitComma(index)
            self.idFiltering = False
            self.visit(key)
            self.idFiltering = True
            self.emit(": ")
            self.visit(value)
        if self.allowJavaScriptIter:
            self.emit("}}")
        else:
            self.emit("}})")
    def visit_DictComp(self, node: ast.DictComp) -> None:
        self.visit_ListComp(node, isDict=True)
    def visit_Expr(self, node: ast.Expr) -> None:
        self.visit(node.value)
    def visit_For(self, node: ast.For) -> None:
        self.adaptLineNrString(node)
        if node.orelse and not self.allowJavaScriptIter:
            self.emit("var {} = false;\n", self.nextTemp("break"))
        else:
            self.skipTemp("break")
        optimize: bool = (isinstance(node.target, ast.Name) and self.isCall(node.iter, "range") and
                          not any(isinstance(arg, ast.Starred) for arg in node.iter.args))
        if self.allowJavaScriptIter:
            self.emit("for (var ")
            self.visit(node.target)
            self.emit(" in ")
            self.visit(node.iter)
            self.emit(") {{\n")
            self.indent()
        elif optimize:
            step: int
            if len(node.iter.args) <= 2:
                step = 1
            else:
                if isinstance(node.iter.args[2], ast.Constant):
                    step = node.iter.args[2].value
                elif isinstance(node.iter.args[2], ast.UnaryOp) and isinstance(node.iter.args[2].operand, ast.Constant):
                    step = -node.iter.args[2].operand.value if isinstance(node.iter.args[2].op, ast.USub) else node.iter.args[2].operand.value
                else:
                    step = 1
            self.emit("for (var ")
            self.visit(node.target)
            self.emit(" = ")
            if len(node.iter.args) > 1:
                self.visit(node.iter.args[0])
            else:
                self.visit(ast.Constant(value=0))
            self.emit("; ")
            self.visit(node.target)
            self.emit(" < " if step > 0 else " > ")
            if len(node.iter.args) > 1:
                self.visit(node.iter.args[1])
            else:
                self.visit(node.iter.args[0])
            self.emit("; ")
            self.visit(node.target)
            if step == 1:
                self.emit("++")
            elif step == -1:
                self.emit("--")
            elif step >= 0:
                self.emit(" += {}", step)
            else:
                self.emit(" -= {}", -step)
            self.emit(") {{\n")
            self.indent()
        elif not self.allowOperatorOverloading:
            self.emit("for (var ")
            self.stripTuples = True
            self.visit(node.target)
            self.stripTuples = False
            self.emit(" of ")
            if self.allowConversionToIterable:
                self.emit("__i__ (")
            self.visit(node.iter)
            if self.allowConversionToIterable:
                self.emit(")")
            self.emit(") {{\n")
            self.indent()
        else:
            self.emit("var {} = ", self.nextTemp("iterable"))
            self.visit(node.iter)
            self.emit(";\n")
            if self.allowConversionToIterable:
                self.emit("{0} = __i__ ({0});\n", self.getTemp("iterable"))
            self.emit("for (var {0} = 0; {0} < len ({1}); {0}++) {{\n", self.nextTemp("index"), self.getTemp("iterable"))
            self.indent()
            self.visit(ast.Assign(
                targets=[node.target],
                value=ast.Subscript(
                    value=ast.Name(id=self.getTemp("iterable"), ctx=ast.Load()),
                    slice=ast.Name(id=self.getTemp("index"), ctx=ast.Load()),
                    ctx=ast.Load()
                )
            ))
            self.emit(";\n")
        self.emitBody(node.body)
        self.dedent()
        self.emit("}}\n")
        if not (self.allowJavaScriptIter or optimize):
            if self.allowOperatorOverloading:
                self.prevTemp("index")
                self.prevTemp("iterable")
        if node.orelse:
            self.adaptLineNrString(node.orelse, 1)
            self.emit("if (!{}) {{\n", self.getTemp("break"))
            self.prevTemp("break")
            self.indent()
            self.emitBody(node.orelse)
            self.dedent()
            self.emit("}}\n")
        else:
            self.prevTemp("break")
    def visit_FormattedValue(self, node: ast.FormattedValue) -> None:
        self.visit(node.value)
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node, anAsync=True)
    def visit_FunctionDef(self, node: ast.FunctionDef, anAsync: bool = False) -> None:
        def emitScopedBody() -> None:
            self.inscope(node)
            self.emitBody(node.body)
            self.dedent()
            if self.getScope(nodeTypes=(ast.AsyncFunctionDef if anAsync else ast.FunctionDef)).get("containsYield"):
                self.targetFragments.insert(yieldStarIndex, "*")
            self.descope()
        def pushPropertyAccessor(functionName: str) -> None:
            self.propertyAccessorList.append({"functionName": functionName, "className": ".".join([scope.node.name for scope in self.getAdjacentClassScopes()])})
        nodeName: str = node.name
        if nodeName != "__pragma__":
            isGlobal: bool = type(self.getScope().get("node")) == ast.Module
            isMethod: bool = not (isGlobal or type(self.getScope().get("node")) in (ast.FunctionDef, ast.AsyncFunctionDef))
            if isMethod:
                self.emit("\n")
            self.adaptLineNrString(node)
            decorate: bool = False
            isClassMethod: bool = False
            isStaticMethod: bool = False
            isProperty: bool = False
            getter: str = "__get__"
            if node.decorator_list:
                for decorator in node.decorator_list:
                    decoratorNode = decorator
                    decoratorType = type(decoratorNode)
                    nameCheck: str = ""
                    while decoratorType != ast.Name:
                        if decoratorType == ast.Call:
                            decoratorNode = decoratorNode.func
                        elif decoratorType == ast.Attribute:
                            nameCheck = "." + decoratorNode.attr + nameCheck
                            decoratorNode = decoratorNode.value
                        decoratorType = type(decoratorNode)
                    nameCheck = decoratorNode.id + nameCheck
                    if nameCheck == "classmethod":
                        isClassMethod = True
                        getter = "__getcm__"
                    elif nameCheck == "staticmethod":
                        isStaticMethod = True
                        getter = "__getsm__"
                    elif nameCheck == "property":
                        isProperty = True
                        nodeName = "_get_" + node.name
                        pushPropertyAccessor(nodeName)
                    elif re.match("[a-zA-Z0-9_]+\.setter", nameCheck):
                        isProperty = True
                        nodeName = "_set_" + re.match("([a-zA-Z0-9_]+)\.setter", nameCheck).group(1)
                        pushPropertyAccessor(nodeName)
                    else:
                        decorate = True
            if sum([isClassMethod, isStaticMethod, isProperty]) > 1:
                raise Exception("\n\tstaticmethod, classmethod and property decorators can't be mixed\n")
            jsCall: bool = self.allowJavaScriptCall and nodeName != "__init__"
            decoratorsUsed: int = 0
            if decorate:
                if isMethod:
                    if jsCall:
                        raise Exception("\n\tdecorators are not supported with jscall\n")
                    else:
                        self.emit("get {} () {{return {} (this, ", self.filterId(nodeName), getter)
                elif isGlobal:
                    if type(node.parentNode) == ast.Module and nodeName not in self.allOwnNames:
                        self.emit("export ")
                    self.emit("var {} = ", self.filterId(nodeName))
                else:
                    self.emit("var {} = ", self.filterId(nodeName))
                if self.allowOperatorOverloading:
                    self.emit("__call__ (")
                for decorator in node.decorator_list:
                    if not (isinstance(decorator, ast.Name) and decorator.id in ("classmethod", "staticmethod")):
                        if decoratorsUsed > 0:
                            self.emit(" (")
                        self.visit(decorator)
                        decoratorsUsed += 1
                if self.allowOperatorOverloading:
                    self.emit(", null, ")
                else:
                    self.emit(" (")
            else:
                if isMethod:
                    if jsCall:
                        self.emit("{}: function", self.filterId(nodeName))
                    else:
                        if isStaticMethod:
                            self.emit("get {} () {{return {}function", self.filterId(nodeName))
                        else:
                            self.emit("get {} () {{return {} (this, {}function", self.filterId(nodeName), getter)
                elif isGlobal:
                    if type(node.parentNode) == ast.Module and nodeName not in self.allOwnNames:
                        self.emit("export ")
                    self.emit("var {} = {}function", self.filterId(nodeName))
                else:
                    self.emit("var {} = {}function", self.filterId(nodeName))
            yieldStarIndex: int = self.fragmentIndex
            self.emit(" ")
            skipFirstArg: bool = jsCall and not (not isMethod or isStaticMethod or isProperty)
            if skipFirstArg:
                firstArg: str = node.args.args[0].arg
                node.args.args = node.args.args[1:]
            self.visit(node.args)
            if skipFirstArg:
                if isClassMethod:
                    self.emit("var {} = '__class__' in this ? this.__class__ : this;\n", firstArg)
                else:
                    self.emit("var {} = this;\n", firstArg)
            emitScopedBody()
            self.emit("}}")
            if self.allowDocAttribs:
                docString = ast.get_docstring(node)
                if docString:
                    self.emit(" .__setdoc__ ('{}')", docString.replace("\n", "\\n "))
            if decorate:
                self.emit(")" * decoratorsUsed)
            if isMethod:
                if not jsCall:
                    if isStaticMethod:
                        self.emit(";}}")
                    else:
                        if self.allowMemoizeCalls:
                            self.emit(", '{}'", nodeName)
                        self.emit(";}}")
                if nodeName == "__iter__":
                    self.emit(",\n[Symbol.iterator] () {{return this.__iter__ ()}}")
                if nodeName == "__next__":
                    self.emit(",\nnext: __jsUsePyNext__")
            if isGlobal:
                self.allOwnNames.add(nodeName)
    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self.visit_ListComp(node, isGenExp=True)
    def visit_Global(self, node: ast.Global) -> None:
        self.getScope(*[ast.FunctionDef, ast.AsyncFunctionDef]).get("nonlocals", set()).update(node.names)
    def visit_If(self, node: ast.If) -> None:
        self.adaptLineNrString(node)
        self.emit("if (")
        self.emitBeginTruthy()
        global inIf
        inIf = True
        self.visit(node.test)
        inIf = False
        self.emitEndTruthy()
        self.emit(") {{\n")
        self.indent()
        self.emitBody(node.body)
        self.dedent()
        self.emit("}}\n")
        if node.orelse:
            if len(node.orelse) == 1 and node.orelse[0].__class__.__name__ == "If":
                self.emit("else ")
                self.visit(node.orelse[0])
            else:
                self.adaptLineNrString(node.orelse, 1)
                self.emit("else {{\n")
                self.indent()
                self.emitBody(node.orelse)
                self.dedent()
                self.emit("}}\n")
    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.emit("(")
        self.emitBeginTruthy()
        self.visit(node.test)
        self.emitEndTruthy()
        self.emit(" ? ")
        self.visit(node.body)
        self.emit(" : ")
        self.visit(node.orelse)
        self.emit(")")
    def visit_Import(self, node: ast.Import) -> None:
        self.importHoistMemos.append({"node": node, "lineNr": self.lineNr})
    def revisit_Import(self, importHoistMemo: Any) -> None:
        self.lineNr = importHoistMemo["lineNr"]
        node: ast.Import = importHoistMemo["node"]
        self.adaptLineNrString(node)
        names = [alias for alias in node.names if not alias.name.startswith(self.module.program.runtimeModuleName)]
        if not names:
            return
        for index, alias in enumerate(names):
            try:
                module = self.useModule(alias.name)
            except Exception as exception:
                raise Exception(f"\n\tCan't import module '{alias.name}'") from exception
            if alias.asname and alias.asname not in (self.allOwnNames | self.allImportedNames):
                self.allImportedNames.add(alias.asname)
                self.emit("import * as {} from '{}';\n", self.filterId(alias.asname), module.importRelPath)
            else:
                self.emit("import * as __module_{}__ from '{}';\n", self.filterId(module.name).replace(".", "_"), module.importRelPath)
                aliasSplit = alias.name.split(".", 1)
                head = aliasSplit[0]
                tail = aliasSplit[1] if len(aliasSplit) > 1 else ""
                self.importHeads.add(head)
                self.emit("__nest__ ({}, '{}', __module_{}__);\n", self.filterId(head), self.filterId(tail), self.filterId(module.name).replace(".", "_"))
            if index < len(names) - 1:
                self.emit(";\n")
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.importHoistMemos.append({"node": node, "lineNr": self.lineNr})
    def revisit_ImportFrom(self, importHoistMemo: Any) -> None:
        self.lineNr = importHoistMemo["lineNr"]
        node: ast.ImportFrom = importHoistMemo["node"]
        self.adaptLineNrString(node)
        if node.module.startswith(self.module.program.runtimeModuleName):
            return
        try:
            self.module.program.searchedModulePaths = []
            namePairs: List[Any] = []
            facilityImported: bool = False
            for index, alias in enumerate(node.names):
                if alias.name == "*":
                    if len(node.names) > 1:
                        raise Exception("\n\tCan't import module '*'")
                    module = self.useModule(node.module)
                    for aName in module.exportedNames:
                        namePairs.append({"name": aName, "asName": None})
                else:
                    try:
                        module = self.useModule(f"{node.module}.{alias.name}")
                        self.emit("import * as {} from '{}';\n", self.filterId(alias.asname) if alias.asname else self.filterId(alias.name), module.importRelPath)
                        self.allImportedNames.add(alias.asname or alias.name)
                    except Exception:
                        module = self.useModule(node.module)
                        namePairs.append({"name": alias.name, "asName": alias.asname})
                        facilityImported = True
            if facilityImported:
                module = self.useModule(node.module)
                namePairs.append({"name": alias.name, "asName": alias.asname})
            if namePairs:
                try:
                    self.emit("import {{")
                    for index, namePair in enumerate(sorted(namePairs, key=lambda np: np["asName"] if np["asName"] else np["name"])):
                        if (namePair["asName"] if namePair["asName"] else namePair["name"]) not in (self.allOwnNames | self.allImportedNames):
                            self.emitComma(index)
                            self.emit("{}", self.filterId(namePair["name"]))
                            if namePair["asName"]:
                                self.emit(" as {}", self.filterId(namePair["asName"]))
                                self.allImportedNames.add(namePair["asName"])
                            else:
                                self.allImportedNames.add(namePair["name"])
                    self.emit("}} from '{}';\n", module.importRelPath)
                except Exception:
                    print("Unexpected import error:", traceback.format_exc())
        except Exception as exception:
            raise Exception(f"\n\tCan't import from module '{node.module}'") from exception
    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        self.emit(repr("".join([value.value if isinstance(value, ast.Constant) else "{}" for value in node.values])))
        self.emit(".format (")
        index: int = 0
        for value in node.values:
            if isinstance(value, ast.FormattedValue):
                self.emitComma(index)
                self.visit(value)
                index += 1
        self.emit(")")
    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.emit("(function __lambda__ ")
        self.visit(node.args)
        self.emit("return ")
        self.visit(node.body)
        self.dedent()
        self.emit(";\n}})")
    def visit_List(self, node: ast.List) -> None:
        self.emit("[")
        for index, elt in enumerate(node.elts):
            self.emitComma(index)
            self.visit(elt)
        self.emit("]")
    def visit_ListComp(self, node: Union[ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp], isSet: bool = False, isDict: bool = False, isGenExp: bool = False) -> None:
        self.emit("(function () {{\n")
        self.inscope(ast.FunctionDef())
        self.indent()
        self.emit("var {} = [];\n", self.nextTemp("accu"))
        def nestLoops(generators: List[Any]) -> None:
            for comprehension in generators:
                target = comprehension.target
                iter_expr = comprehension.iter
                bodies: List[List[Any]] = []
                bodies.append([])
                bodies[-2].append(ast.For(target, iter_expr, bodies[-1], []))
                for expr in comprehension.ifs:
                    test = expr
                    bodies.append([])
                    bodies[-2].append(ast.If(test=test, body=bodies[-1], orelse=[]))
            bodies[-1].append(ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id=self.getTemp("accu"), ctx=ast.Load()),
                    attr="append",
                    ctx=ast.Load()
                ),
                args=[ast.List(elts=[node.key, node.value], ctx=ast.Load())] if isDict else node.elt,
                keywords=[]
            ))
            self.visit(bodies[0][0])
        nestLoops(node.generators[:])
        self.emit("return {}{}{};\n", "set (" if isSet else "dict (" if isDict else "{} (".format(self.filterId("iter")) if isGenExp else "", self.getTemp("accu"), ")" if isSet or isDict or isGenExp else "")
        self.prevTemp("accu")
        self.dedent()
        self.descope()
        self.emit("}}) ()")
    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit_Assign(ast.Assign(targets=[node.target], value=node.value))
    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.slice, ast.Index):
            if isinstance(node.slice.value, ast.Tuple):
                self.visit(node.value)
                self.emit(".__getitem__ (")
                self.stripTuple = True
                self.visit(node.slice.value)
                self.emit(")")
            elif self.allowOperatorOverloading:
                self.emit("__getitem__ (")
                self.visit(node.value)
                self.emit(", ")
                self.visit(node.slice.value)
                self.emit(")")
            else:
                try:
                    isRhsIndex: bool = not self.expectingNonOverloadedLhsIndex
                    self.expectingNonOverloadedLhsIndex = False
                    if isRhsIndex and self.allowKeyCheck:
                        self.emit("__k__ (")
                        self.visit(node.value)
                        self.emit(", ")
                        self.visit(node.slice.value)
                        self.emit(")")
                    else:
                        self.visit(node.value)
                        self.emit(" [")
                        self.visit(node.slice.value)
                        self.emit("]")
                except Exception:
                    print(traceback.format_exc())
        elif isinstance(node.slice, ast.Slice):
            if self.allowOperatorOverloading:
                self.emit("__getslice__ (")
                self.visit(node.value)
                self.emit(", ")
            else:
                self.visit(node.value)
                self.emit(".__getslice__ (")
            if node.slice.lower is None:
                self.emit("0")
            else:
                self.visit(node.slice.lower)
            self.emit(", ")
            if node.slice.upper is None:
                self.emit("null")
            else:
                self.visit(node.slice.upper)
            self.emit(", ")
            if node.slice.step is None:
                self.emit("1")
            else:
                self.visit(node.slice.step)
            self.emit(")")
        elif isinstance(node.slice, ast.ExtSlice):
            self.visit(node.value)
            self.emit(".__getitem__ (")
            self.emit("[")
            for index, dim in enumerate(node.slice.dims):
                self.emitComma(index)
                self.visit(dim)
            self.emit("]")
            self.emit(")")
    def visit_Try(self, node: ast.Try) -> None:
        self.adaptLineNrString(node)
        self.emit("try {{\n")
        self.indent()
        self.emitBody(node.body)
        if node.orelse:
            self.emit("try {{\n")
            self.indent()
            self.emitBody(node.orelse)
            self.dedent()
            self.emit("}}\n")
            self.emit("catch ({}) {{\n", self.nextTemp("except"))
            self.emit("}}\n")
            self.prevTemp("except")
        self.dedent()
        self.emit("}}\n")
        if node.handlers:
            self.emit("catch ({}) {{\n", self.nextTemp("except"))
            self.indent()
            for index, exceptionHandler in enumerate(node.handlers):
                if index:
                    self.emit("else ")
                if exceptionHandler.type:
                    self.emit("if (isinstance ({}, ", self.getTemp("except"))
                    self.visit(exceptionHandler.type)
                    self.emit(")) {{\n")
                    self.indent()
                    if exceptionHandler.name:
                        self.emit("var {} = {};\n", exceptionHandler.name, self.getTemp("except"))
                    self.emitBody(exceptionHandler.body)
                    self.dedent()
                    self.emit("}}\n")
                else:
                    self.emitBody(exceptionHandler.body)
                    break
            else:
                self.emit("else {{\n")
                self.indent()
                self.emit("throw {};\n", self.getTemp("except"))
                self.dedent()
                self.emit("}}\n")
            self.dedent()
            self.prevTemp("except")
            self.emit("}}\n")
        if node.finalbody:
            self.emit("finally {{\n")
            self.indent()
            self.emitBody(node.finalbody)
            self.dedent()
            self.emit("}}\n")
    def visit_Tuple(self, node: ast.Tuple) -> None:
        keepTuple: bool = not (self.stripTuple or self.stripTuples)
        self.stripTuple = False
        if keepTuple:
            self.emit("tuple (")
        self.emit("[")
        for index, elt in enumerate(node.elts):
            self.emitComma(index)
            self.visit(elt)
        self.emit("]")
        if keepTuple:
            self.emit(")")
    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if self.allowOperatorOverloading and isinstance(node.op, ast.USub):
            self.emit("{} (".format(self.filterId("__neg__")))
            self.visit(node.operand)
            self.emit(")")
        else:
            self.emit(self.operators[type(node.op)][0])
            self.emitBeginTruthy()
            self.visitSubExpr(node, node.operand)
            self.emitEndTruthy()
    def visit_While(self, node: ast.While) -> None:
        self.adaptLineNrString(node)
        if node.orelse:
            self.emit("var {} = false;\n", self.nextTemp("break"))
        else:
            self.skipTemp("break")
        self.emit("while (")
        self.emitBeginTruthy()
        self.visit(node.test)
        self.emitEndTruthy()
        self.emit(") {{\n")
        self.indent()
        self.emitBody(node.body)
        self.dedent()
        self.emit("}}\n")
        if node.orelse:
            self.adaptLineNrString(node.orelse, 1)
            self.emit("if (!{}) {{\n", self.getTemp("break"))
            self.prevTemp("break")
            self.indent()
            self.emitBody(node.orelse)
            self.dedent()
            self.emit("}}\n")
        else:
            self.prevTemp("break")
    def visit_With(self, node: ast.With) -> None:
        from contextlib import ExitStack
        self.adaptLineNrString(node)
        @contextmanager
        def itemContext(item: ast.withitem) -> Any:
            if not self.noskipCodeGeneration:
                yield
                return
            self.emit("var ")
            if item.optional_vars:
                self.visit(item.optional_vars)
                withId = item.optional_vars.id  # type: ignore
            else:
                withId = self.nextTemp("withid")
                self.emit(withId)
            self.emit(" = ")
            self.visit(item.context_expr)
            self.emit(";\n")
            self.emit("try {{\n")
            self.indent()
            self.emit("{}.__enter__ ();\n", withId)
            yield
            self.emit("{}.__exit__ ();\n", withId)
            self.dedent()
            self.emit("}}\n")
            self.emit("catch ({}) {{\n", self.nextTemp("except"))
            self.indent()
            self.emit("if (! ({}.__exit__ ({}.name, {}, {}.stack))) {{\n", withId, self.getTemp("except"), self.getTemp("except"))
            self.indent()
            self.emit("throw {};\n", self.getTemp("except"))
            self.dedent()
            self.emit("}}\n")
            self.dedent()
            self.emit("}}\n")
            self.prevTemp("except")
            if withId == self.getTemp("withid"):
                self.prevTemp("withid")
        @contextmanager
        def pragmaContext(item: ast.withitem) -> Any:
            expr = item.context_expr
            name: str = expr.args[0].s  # type: ignore
            if name.startswith("no"):
                revName = name[2:]
            else:
                revName = "no" + name
            self.visit(expr)
            yield
            self.visit(ast.Call(expr.func, [ast.Constant(value=name)] + expr.args[1:], keywords=[]))
        @contextmanager
        def skipContext(item: ast.withitem) -> Any:
            self.noskipCodeGeneration = False
            yield
            self.noskipCodeGeneration = True
        with ExitStack() as stack:
            for item in node.items:
                expr = item.context_expr
                if self.isCall(expr, "__pragma__"):
                    if expr.args[0].s == "skip":  # type: ignore
                        stack.enter_context(skipContext(item))
                    else:
                        stack.enter_context(pragmaContext(item))
                else:
                    stack.enter_context(itemContext(item))
            self.emitBody(node.body)
    def visit_Yield(self, node: ast.Yield) -> None:
        self.getScope(*[ast.FunctionDef, ast.AsyncFunctionDef])["containsYield"] = True
        self.emit("yield")
        if node.value is not None:
            self.emit(" ")
            self.visit(node.value)
    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
        self.getScope(*[ast.FunctionDef, ast.AsyncFunctionDef])["containsYield"] = True
        self.emit("yield* ")
        self.visit(node.value)
# End of annotated code.
