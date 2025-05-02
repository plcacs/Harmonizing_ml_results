import os
import os.path
import sys
import ast
import re
import copy
import datetime
import math
import traceback
import io
import subprocess
import shlex
import shutil
import tokenize
import collections
import json
from contextlib import contextmanager, ExitStack
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from org.transcrypt import utils, sourcemaps, minify, static_check, type_check
inIf: bool = False
ecom: bool = True
noecom: bool = False
dataClassDefaultArgTuple: Tuple[List[Union[str, bool]], ...] = (['init', True], ['repr', True], ['eq', True], ['order', False], ['unsafe_hash', False], ['frozen', False])
"\nAll files required for deployment are placed in subdirectory __target__ of the application.\nEach module has an unambiguous dotted path, always from one of the module roots, never relative.\nDotted paths are translated to dotted filenames.\nThe __runtime__ module is just another Python module with lots of JS code inside __pragma__ ('js', '{}', include...) fragments,\nnamely the __core__ and __builtin__ parts.\n\nSourcemaps are generated per module.\nThere's no need for a link with other modules.\nSince import paths are static, names of minified JS files simply end on .js just like non-minified files, so not on .min.js.\nSourcemaps are named <module name>.map.\n"

class Program:

    def __init__(self, moduleSearchDirs, symbols, envir):
        utils.setProgram(self)
        self.moduleSearchDirs: List[str] = moduleSearchDirs
        self.symbols: Set[str] = symbols
        self.envir: Any = envir
        self.javascriptVersion: int = int(utils.commandArgs.esv) if utils.commandArgs.esv else 6
        self.moduleDict: Dict[str, 'Module'] = {}
        self.importStack: List[List[Union['Module', Optional[int]]]] = []
        self.sourcePrepath: str = os.path.abspath(utils.commandArgs.source).replace('\\', '/')
        self.sourceDir: str = '/'.join(self.sourcePrepath.split('/')[:-1])
        self.mainModuleName: str = self.sourcePrepath.split('/')[-1]
        if utils.commandArgs.outdir:
            if os.path.isabs(utils.commandArgs.outdir):
                self.targetDir: str = utils.commandArgs.outdir.replace('\\', '/')
            else:
                self.targetDir: str = f'{self.sourceDir}/{utils.commandArgs.outdir}'.replace('\\', '/')
        else:
            self.targetDir: str = f'{self.sourceDir}/__target__'.replace('\\', '/')
        self.projectPath: str = f'{self.targetDir}/{self.mainModuleName}.project'
        try:
            with open(self.projectPath, 'r') as projectFile:
                project: Dict[str, Any] = json.load(projectFile)
        except:
            project = {}
        self.optionsChanged: bool = utils.commandArgs.projectOptions != project.get('options')
        if utils.commandArgs.build or self.optionsChanged:
            shutil.rmtree(self.targetDir, ignore_errors=True)
        try:
            self.runtimeModuleName: str = 'org.transcrypt.__runtime__'
            self.searchedModulePaths: List[str] = []
            self.provide(self.runtimeModuleName)
            self.searchedModulePaths = []
            self.provide(self.mainModuleName, '__main__')
        except Exception as exception:
            utils.enhanceException(exception, message=f'\n\t{exception}')
        project = {'options': utils.commandArgs.projectOptions, 'modules': [{'source': module.sourcePath, 'target': module.targetPath} for module in self.moduleDict.values()]}
        with utils.create(self.projectPath) as projectFile:
            json.dump(project, projectFile)

    def provide(self, moduleName, __moduleName__=None, filter=None):
        if moduleName in self.moduleDict:
            return self.moduleDict[moduleName]
        else:
            return Module(self, moduleName, __moduleName__, filter)

class Module:

    def __init__(self, program, name, __name__, filter):
        self.program: Program = program
        self.name: str = name
        self.__name__: str = __name__ if __name__ else self.name
        self.findPaths(filter)
        self.program.importStack.append([self, None])
        self.program.moduleDict[self.name] = self
        self.sourceMapper: sourcemaps.SourceMapper = sourcemaps.SourceMapper(self.name, self.program.targetDir, not utils.commandArgs.nomin, utils.commandArgs.dmap)
        if utils.commandArgs.build or self.program.optionsChanged or (not os.path.isfile(self.targetPath)) or (os.path.getmtime(self.sourcePath) > os.path.getmtime(self.targetPath)):
            if self.isJavascriptOnly:
                self.loadJavascript()
                javascriptDigest: Any = utils.digestJavascript(self.targetCode, self.program.symbols, not utils.commandArgs.dnostrip, False)
            else:
                if utils.commandArgs.dstat:
                    try:
                        type_check.run(self.sourcePath)
                    except Exception as exception:
                        utils.log(True, 'Validating: {} and dependencies\n\tInternal error in static typing validator\n', self.sourcePath)
                self.parse()
                if utils.commandArgs.dtree:
                    self.dumpTree()
                if utils.commandArgs.dcheck:
                    try:
                        static_check.run(self.sourcePath, self.parseTree)
                    except Exception as exception:
                        utils.log(True, 'Checking: {}\n\tInternal error in lightweight consistency checker, remainder of module skipped\n', self.sourcePath)
                self.generateJavascriptAndPrettyMap()
                javascriptDigest: Any = utils.digestJavascript(self.targetCode, self.program.symbols, False, self.generator.allowDebugMap)
            utils.log(True, 'Saving target code in: {}\n', self.targetPath)
            filePath: str = self.targetPath if utils.commandArgs.nomin else self.prettyTargetPath
            with utils.create(filePath) as aFile:
                aFile.write(self.targetCode)
            if not utils.commandArgs.nomin:
                utils.log(True, 'Saving minified target code in: {}\n', self.targetPath)
                minify.run(self.program.targetDir, self.prettyTargetName, self.targetName, mapFileName=self.shrinkMapName if utils.commandArgs.map else None)
                if utils.commandArgs.map:
                    if self.isJavascriptOnly:
                        if os.path.isfile(self.mapPath):
                            os.remove(self.mapPath)
                        os.rename(self.shrinkMapPath, self.mapPath)
                    else:
                        self.sourceMapper.generateMultilevelMap()
            with open(self.targetPath, 'a') as targetFile:
                targetFile.write(self.mapRef)
        else:
            self.targetCode: str = open(self.targetPath, 'r').read()
            javascriptDigest: Any = utils.digestJavascript(self.targetCode, self.program.symbols, True, False, refuseIfAppearsMinified=True)
            if not javascriptDigest:
                minify.run(self.program.targetDir, self.targetName, self.prettyTargetName, prettify=True)
                self.prettyTargetCode: str = open(self.prettyTargetPath, 'r').read()
                javascriptDigest: Any = utils.digestJavascript(self.prettyTargetCode, self.program.symbols, True, False)
        self.targetCode: str = javascriptDigest.digestedCode
        self.importedModuleNames: Set[str] = javascriptDigest.importedModuleNames
        self.exportedNames: Set[str] = javascriptDigest.exportedNames
        for importedModuleName in self.importedModuleNames:
            self.program.searchedModulePaths = []
            self.program.provide(importedModuleName)
        utils.tryRemove(self.prettyTargetPath)
        utils.tryRemove(self.shrinkMapPath)
        utils.tryRemove(self.prettyMapPath)
        self.program.importStack.pop()

    def findPaths(self, filter):
        rawRelSourceSlug: str = self.name.replace('.', '/')
        relSourceSlug: str = filter(rawRelSourceSlug) if filter and utils.commandArgs.alimod else rawRelSourceSlug
        "\n        # BEGIN DEBUGGING CODE\n        print ()\n        print ('Raw slug   :', rawRelSourceSlug)\n        print ('Cooked slug:', relSourceSlug)\n        print ()\n        # END DEBUGGING CODE\n        "
        for searchDir in self.program.moduleSearchDirs:
            sourceSlug: str = f'{searchDir}/{relSourceSlug}'
            if os.path.isdir(sourceSlug):
                self.sourceDir: str = sourceSlug
                self.sourcePrename: str = '__init__'
            else:
                self.sourceDir, self.sourcePrename = sourceSlug.rsplit('/', 1)
            self.sourcePrepath: str = f'{self.sourceDir}/{self.sourcePrename}'
            self.pythonSourcePath: str = f'{self.sourcePrepath}.py'
            self.javascriptSourcePath: str = f'{self.sourcePrepath}.js'
            self.targetPrepath: str = f'{self.program.targetDir}/{self.name}'
            self.targetName: str = f'{self.name}.js'
            self.targetPath: str = f'{self.targetPrepath}.js'
            self.prettyTargetName: str = f'{self.name}.pretty.js'
            self.prettyTargetPath: str = f'{self.targetPrepath}.pretty.js'
            self.importRelPath: str = f'./{self.name}.js'
            self.treePath: str = f'{self.targetPrepath}.tree'
            self.mapPath: str = f'{self.targetPrepath}.map'
            self.prettyMapPath: str = f'{self.targetPrepath}.shrink.map'
            self.shrinkMapName: str = f'{self.name}.shrink.map'
            self.shrinkMapPath: str = f'{self.targetPrepath}.shrink.map'
            self.mapSourcePath: str = f'{self.targetPrepath}.py'
            self.mapRef: str = f'\n//# sourceMappingURL={self.name}.map'
            if os.path.isfile(self.pythonSourcePath) or os.path.isfile(self.javascriptSourcePath):
                self.isJavascriptOnly: bool = os.path.isfile(self.javascriptSourcePath) and (not os.path.isfile(self.pythonSourcePath))
                self.sourcePath: str = self.javascriptSourcePath if self.isJavascriptOnly else self.pythonSourcePath
                break
            self.program.searchedModulePaths.extend([self.pythonSourcePath, self.javascriptSourcePath])
        else:
            raise utils.Error(message=f"\n\tImport error, can't find any of:\n\t\t{chr(10).join(self.program.searchedModulePaths)}\n")

    def generateJavascriptAndPrettyMap(self):
        utils.log(False, 'Generating code for module: {}\n', self.targetPath)
        self.generator: Generator = Generator(self)
        if utils.commandArgs.map or utils.commandArgs.anno:
            instrumentedTargetLines: List[str] = ''.join(self.generator.targetFragments).split('\n')
            if utils.commandArgs.map:
                self.sourceLineNrs: List[int] = []
            targetLines: List[str] = []
            for targetLine in instrumentedTargetLines:
                sourceLineNrString: str = targetLine[-sourcemaps.lineNrLength:]
                sourceLineNr: int = int('1' + sourceLineNrString) - sourcemaps.maxNrOfSourceLinesPerModule
                targetLine = targetLine[:-sourcemaps.lineNrLength]
                if targetLine.strip() != ';':
                    if self.generator.allowDebugMap:
                        targetLine = f'/* {sourceLineNrString} */ {targetLine}'
                    targetLines.append(targetLine)
                    if utils.commandArgs.map:
                        self.sourceLineNrs.append(sourceLineNr)
            if utils.commandArgs.map:
                utils.log(False, 'Saving source map in: {}\n', self.mapPath)
                self.sourceMapper.generateAndSavePrettyMap(self.sourceLineNrs)
                shutil.copyfile(self.sourcePath, self.mapSourcePath)
        else:
            targetLines: List[str] = [line for line in ''.join(self.generator.targetFragments).split('\n') if line.strip() != ';']
        self.targetCode: str = '\n'.join(targetLines)

    def loadJavascript(self):
        with tokenize.open(self.sourcePath) as sourceFile:
            self.targetCode: str = sourceFile.read()

    def parse(self):

        def pragmasFromComments(sourceCode):
            tokens = tokenize.tokenize(io.BytesIO(sourceCode.encode('utf-8')).readline)
            pragmaCommentLineIndices: List[int] = []
            shortPragmaCommentLineIndices: List[int] = []
            ecomPragmaLineIndices: List[int] = []
            noecomPragmaLineIndices: List[int] = []
            pragmaIndex: int = -1000
            for tokenIndex, (tokenType, tokenString, startRowColumn, endRowColumn, logicalLine) in enumerate(tokens):
                if tokenType == tokenize.COMMENT:
                    strippedComment: str = tokenString[1:].lstrip()
                    if strippedComment.startswith('__pragma__'):
                        pragmaCommentLineIndices.append(startRowColumn[0] - 1)
                    elif strippedComment.replace(' ', '').replace('\t', '').startswith('__:'):
                        shortPragmaCommentLineIndices.append(startRowColumn[0] - 1)
                if tokenType == tokenize.NAME and tokenString == '__pragma__':
                    pragmaIndex = tokenIndex
                if pragmaIndex - tokenIndex == 2:
                    pragmaKind: str = tokenString[1:-1]
                    if pragmaKind == 'ecom':
                        ecomPragmaLineIndices.append(startRowColumn[0] - 1)
                    elif pragmaKind == 'noecom':
                        noecomPragmaLineIndices.append(startRowColumn[0] - 1)
            sourceLines: List[str] = sourceCode.split('\n')
            for ecomPragmaLineIndex in ecomPragmaLineIndices:
                sourceLines[ecomPragmaLineIndex] = ecom
            for noecomPragmaLineIndex in noecomPragmaLineIndices:
                sourceLines[noecomPragmaLineIndex] = noecom
            allowExecutableComments: bool = utils.commandArgs.ecom
            for pragmaCommentLineIndex in pragmaCommentLineIndices:
                indentation, separator, tail = sourceLines[pragmaCommentLineIndex].partition('#')
                pragma, separator, comment = tail.partition('#')
                pragma = pragma.replace(' ', '').replace('\t', '')
                if "('ecom')" in pragma or '("ecom")' in pragma:
                    allowExecutableComments = True
                    sourceLines[pragmaCommentLineIndex] = ecom
                elif "('noecom')" in pragma or '("noecom")' in pragma:
                    allowExecutableComments = False
                    sourceLines[pragmaCommentLineIndex] = noecom
                else:
                    sourceLines[pragmaCommentLineIndex] = indentation + tail.lstrip()
            for shortPragmaCommentLineIndex in shortPragmaCommentLineIndices:
                head, tail = sourceLines[shortPragmaCommentLineIndex].rsplit('#', 1)
                strippedHead: str = head.lstrip()
                indent: str = head[:len(head) - len(strippedHead)]
                pragmaName: str = tail.replace(' ', '').replace('\t', '')[3:]
                if pragmaName == 'ecom':
                    sourceLines[shortPragmaCommentLineIndex] = ecom
                elif pragmaName == 'noecom':
                    sourceLines[shortPragmaCommentLineIndex] = noecom
                elif pragmaName.startswith('no'):
                    sourceLines[shortPragmaCommentLineIndex] = f"{indent}__pragma__ ('{pragmaName}'); {head}; __pragma__ ('{pragmaName[2:]}'"
                else:
                    sourceLines[shortPragmaCommentLineIndex] = f"{indent}__pragma__ ('{pragmaName}'); {head}; __pragma__ ('no{pragmaName}'"
            uncommentedSourceLines: List[str] = []
            for sourceLine in sourceLines:
                if sourceLine == ecom:
                    allowExecutableComments = True
                elif sourceLine == noecom:
                    allowExecutableComments = False
                elif allowExecutableComments:
                    lStrippedSourceLine: str = sourceLine.lstrip()
                    if not lStrippedSourceLine[:4] in {"'''?", "?'''", '"""?', '?"""'}:
                        uncommentedSourceLines.append(sourceLine.replace('#?', '', 1) if lStrippedSourceLine.startswith('#?') else sourceLine)
                else:
                    uncommentedSourceLines.append(sourceLine)
            return '\n'.join(uncommentedSourceLines)
        try:
            utils.log(False, 'Parsing module: {}\n', self.sourcePath)
            with tokenize.open(self.sourcePath) as sourceFile:
                self.sourceCode: str = utils.extraLines + sourceFile.read()
            self.parseTree: ast.AST = ast.parse(pragmasFromComments(self.sourceCode))
            for node in ast.walk(self.parseTree):
                for childNode in ast.iter_child_nodes(node):
                    setattr(childNode, 'parentNode', node)
        except SyntaxError as syntaxError:
            utils.enhanceException(syntaxError, lineNr=syntaxError.lineno, message=f'\n\t{syntaxError.text[:syntaxError.offset].lstrip()} [<-SYNTAX FAULT] {syntaxError.text[syntaxError.offset:].rstrip()}' if syntaxError.text else syntaxError.args[0])

    def dumpTree(self):
        utils.log(False, 'Dumping syntax tree for module: {}\n', self.sourcePath)

        def walk(name, value, tabLevel):
            self.treeFragments.append(f'\n{'\t' * tabLevel}{name}: {type(value).__name__} ')
            if isinstance(value, ast.AST):
                for field in ast.iter_fields(value):
                    walk(field[0], field[1], tabLevel + 1)
            elif isinstance(value, list):
                for element in value:
                    walk('element', element, tabLevel + 1)
            else:
                self.treeFragments.append(f'= {value}')
        self.treeFragments: List[str] = []
        walk('file', self.parseTree, 0)
        self.textTree: str = ''.join(self.treeFragments)[1:]
        with utils.create(self.treePath) as treeFile:
            treeFile.write(self.textTree)

class Generator(ast.NodeVisitor):

    def __init__(self, module):
        self.module: Module = module
        self.targetFragments: List[str] = []
        self.fragmentIndex: int = 0
        self.indentLevel: int = 0
        self.scopes: List[Any] = []
        self.importHeads: Set[str] = set()
        self.importHoistMemos: List[Any] = []
        self.allOwnNames: Set[str] = set()
        self.allImportedNames: Set[str] = set()
        self.expectingNonOverloadedLhsIndex: bool = False
        self.lineNr: int = 1
        self.propertyAccessorList: List[Any] = []
        self.mergeList: List[Any] = []
        self.aliases: List[Tuple[str, str]] = [('js_and', 'and'), ('arguments', 'py_arguments'), ('js_arguments', 'arguments'), ('case', 'py_case'), ('clear', 'py_clear'), ('js_clear', 'clear'), ('js_conjugate', 'conjugate'), ('default', 'py_default'), ('del', 'py_del'), ('js_del', 'del'), ('false', 'py_false'), ('js_from', 'from'), ('get', 'py_get'), ('js_get', 'get'), ('js_global', 'global'), ('Infinity', 'py_Infinity'), ('js_Infinity', 'Infinity'), ('is', 'py_is'), ('js_is', 'is'), ('isNaN', 'py_isNaN'), ('js_isNaN', 'isNaN'), ('iter', 'py_iter'), ('js_iter', 'iter'), ('items', 'py_items'), ('js_items', 'items'), ('keys', 'py_keys'), ('js_keys', 'keys'), ('name', 'py_name'), ('js_name', 'name'), ('NaN', 'py_NaN'), ('js_NaN', 'NaN'), ('new', 'py_new'), ('next', 'py_next'), ('js_next', 'next'), ('js_not', 'not'), ('js_or', 'or'), ('pop', 'py_pop'), ('js_pop', 'pop'), ('popitem', 'py_popitem'), ('js_popitem', 'popitem'), ('replace', 'py_replace'), ('js_replace', 'replace'), ('selector', 'py_selector'), ('js_selector', 'selector'), ('sort', 'py_sort'), ('js_sort', 'sort'), ('split', 'py_split'), ('js_split', 'split'), ('switch', 'py_switch'), ('type', 'py_metatype'), ('js_type', 'type'), ('TypeError', 'py_TypeError'), ('js_TypeError', 'TypeError'), ('update', 'py_update'), ('js_update', 'update'), ('values', 'py_values'), ('js_values', 'values'), ('reversed', 'py_reversed'), ('js_reversed', 'reversed'), ('setdefault', 'py_setdefault'), ('js_setdefault', 'setdefault'), ('js_super', 'super'), ('true', 'py_true'), ('undefined', 'py_undefined'), ('js_undefined', 'undefined')]
        self.idFiltering: bool = True
        self.tempIndices: Dict[str, int] = {}
        self.skippedTemps: Set[str] = set()
        self.stubsName: str = f'org.{self.module.program.envir.transpiler_name}.stubs.'
        self.nameConsts: Dict[Any, str] = {None: 'null', True: 'true', False: 'false'}
        "\n        The precedences explicitly given as integers in the list below are JavaScript precedences as specified by:\n        https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/Operator_Precedence .\n\n        Python precedences are implicitly present in branch ordering of the AST generated by CPython's parser.\n        "
        self.operators: Dict[Any, Tuple[str, int]] = {ast.Not: ('!', 16), ast.Invert: ('~', 16), ast.UAdd: ('+', 16), ast.USub: ('-', 16), ast.Pow: (None, 15), ast.Mult: ('*', 14), ast.MatMult: (None, 14), ast.Div: ('/', 14), ast.FloorDiv: (None, 14), ast.Mod: ('%', 14), ast.Add: ('+', 13), ast.Sub: ('-', 13), ast.LShift: ('<<', 12), ast.RShift: ('>>', 12), ast.Lt: ('<', 11), ast.LtE: ('<=', 11), ast.Gt: ('>', 11), ast.GtE: ('>=', 11), ast.In: (None, 11), ast.NotIn: (None, 11), ast.Eq: ('==', 10), ast.NotEq: ('!=', 10), ast.Is: ('===', 10), ast.IsNot: ('!==', 10), ast.BitAnd: ('&', 9), ast.BitOr: ('|', 8), ast.BitXor: ('^', 7), ast.And: ('&&', 6), ast.Or: ('||', 5)}
        self.allowKeywordArgs: bool = utils.commandArgs.kwargs
        self.allowOperatorOverloading: bool = utils.commandArgs.opov
        self.allowConversionToIterable: bool = utils.commandArgs.iconv
        self.allowConversionToTruthValue: bool = utils.commandArgs.tconv
        self.allowKeyCheck: bool = utils.commandArgs.keycheck
        self.allowDebugMap: bool = utils.commandArgs.anno and (not self.module.sourcePath.endswith('.js'))
        self.allowDocAttribs: bool = utils.commandArgs.docat
        self.allowGlobals: bool = utils.commandArgs.xglobs
        self.allowJavaScriptIter: bool = False
        self.allowJavaScriptCall: bool = utils.commandArgs.jscall
        self.allowJavaScriptKeys: bool = utils.commandArgs.jskeys
        self.allowJavaScriptMod: bool = utils.commandArgs.jsmod
        self.allowMemoizeCalls: bool = utils.commandArgs.fcall
        self.noskipCodeGeneration: bool = True
        self.conditionalCodeGeneration: bool = True
        self.stripTuple: bool = False
        self.stripTuples: bool = False
        self.replaceSend: bool = False
        try:
            self.visit(self.module.parseTree)
            self.targetFragments.append(self.lineNrString)
        except Exception as exception:
            utils.enhanceException(exception, lineNr=self.lineNr)
        if self.tempIndices:
            raise utils.Error(message=f'\n\tTemporary variables leak in code generator: {self.tempIndices}')

    def visitSubExpr(self, node, child):

        def getPriority(exprNode):
            if type(exprNode) in (ast.BinOp, ast.BoolOp):
                return self.operators[type(exprNode.op)][1]
            elif type(exprNode) == ast.Compare:
                return self.operators[type(exprNode.ops[0])][1]
            elif type(exprNode) == ast.Yield:
                return -1000000
            else:
                return 1000000
        if getPriority(child) <= getPriority(node):
            self.emit('(')
            self.visit(child)
            self.emit(')')
        else:
            self.visit(child)

    def filterId(self, qualifiedId):
        if not self.idFiltering or (qualifiedId.startswith('__') and qualifiedId.endswith('__')):
            return qualifiedId
        else:
            for alias in self.aliases:
                qualifiedId = re.sub(f'(^|(?P<preDunder>__)){re.escape(alias[0])}((?P<postDunder>__)|(?=[./])|$)', lambda matchObject: ('=' if matchObject.group('preDunder') else '') + alias[1] + ('=' if matchObject.group('postDunder') else ''), qualifiedId)
                qualifiedId = re.sub(f'(^|(?<=[./=])){re.escape(alias[0])}((?=[./=])|$)', alias[1], qualifiedId)
            return qualifiedId.replace('=', '')

    def tabs(self, indentLevel=None):
        if indentLevel is None:
            indentLevel = self.indentLevel
        return indentLevel * '\t'

    def emit(self, fragment, *formatter: Any):
        if not self.targetFragments or (self.targetFragments and self.targetFragments[self.fragmentIndex - 1].endswith('\n')):
            self.targetFragments.insert(self.fragmentIndex, self.tabs())
            self.fragmentIndex += 1
        fragment = fragment[:-1].replace('\n', '\n' + self.tabs()) + fragment[-1]
        self.targetFragments.insert(self.fragmentIndex, fragment.format(*formatter).replace('\n', self.lineNrString + '\n'))
        self.fragmentIndex += 1

    def indent(self):
        self.indentLevel += 1

    def dedent(self):
        self.indentLevel -= 1

    def inscope(self, node):
        self.scopes.append(utils.Any(node=node, nonlocals=set(), containsYield=False))

    def descope(self):
        self.scopes.pop()

    def getScope(self, *nodeTypes: type):
        if nodeTypes:
            for scope in reversed(self.scopes):
                if type(scope.node) in nodeTypes:
                    return scope
        else:
            return self.scopes[-1]

    def getAdjacentClassScopes(self, inMethod=False):
        reversedClassScopes: List[Any] = []
        for scope in reversed(self.scopes):
            if inMethod:
                if type(scope.node) in (ast.FunctionDef, ast.AsyncFunctionDef):
                    continue
                else:
                    inMethod = False
            if type(scope.node) != ast.ClassDef:
                break
            reversedClassScopes.append(scope)
        return list(reversed(reversedClassScopes))

    def emitComma(self, index, blank=True):
        if self.noskipCodeGeneration and self.conditionalCodeGeneration and index:
            self.emit(', ' if blank else ',')

    def emitBeginTruthy(self):
        if self.allowConversionToTruthValue:
            self.emit('__t__ (')

    def emitEndTruthy(self):
        if self.allowConversionToTruthValue:
            self.emit(')')

    def adaptLineNrString(self, node=None, offset=0):
        if utils.commandArgs.map or utils.commandArgs.anno:
            if node:
                if hasattr(node, 'lineno'):
                    lineNr: int = node.lineno + offset
                else:
                    lineNr: int = self.lineNr + offset
            else:
                lineNr = 1 + offset
            self.lineNrString: str = str(sourcemaps.maxNrOfSourceLinesPerModule + lineNr)[1:]
        else:
            self.lineNrString: str = ''

    def isCommentString(self, statement):
        return isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str)

    def emitBody(self, body):
        for statement in body:
            if self.isCommentString(statement):
                pass
            else:
                self.visit(statement)
                self.emit(';\n')

    def emitSubscriptAssign(self, target, value, emitPathIndices=lambda: None):
        if isinstance(target.slice, ast.Index):
            if isinstance(target.slice.value, ast.Tuple):
                self.visit(target.value)
                self.emit('.__setitem__ (')
                self.stripTuple = True
                self.visit(target.slice.value)
                self.emit(', ')
                self.visit(value)
                emitPathIndices()
                self.emit(')')
            elif self.allowOperatorOverloading:
                self.emit('__setitem__ (')
                self.visit(target.value)
                self.emit(', ')
                self.visit(target.slice.value)
                self.emit(', ')
                self.visit(value)
                emitPathIndices()
                self.emit(')')
            else:
                self.expectingNonOverloadedLhsIndex = True
                self.visit(target)
                self.emit(' = ')
                self.visit(value)
                emitPathIndices()
        elif isinstance(target.slice, ast.Slice):
            if self.allowOperatorOverloading:
                self.emit('__setslice__ (')
                self.visit(target.value)
                self.emit(', ')
            else:
                self.visit(target.value)
                self.emit('.__setslice__ (')
            if target.slice.lower is None:
                self.emit('0')
            else:
                self.visit(target.slice.lower)
            self.emit(', ')
            if target.slice.upper is None:
                self.emit('null')
            else:
                self.visit(target.slice.upper)
            self.emit(', ')
            if target.slice.step:
                self.visit(target.slice.step)
            else:
                self.emit('null')
            self.emit(', ')
            self.visit(value)
            self.emit(')')
        elif isinstance(target.slice, ast.ExtSlice):
            self.visit(target.value)
            self.emit('.__setitem__ (')
            self.emit('[')
            for index, dim in enumerate(target.slice.dims):
                self.emitComma(index)
                self.visit(dim)
            self.emit(']')
            self.emit(', ')
            self.visit(value)
            self.emit(')')

    def nextTemp(self, name):
        if name in self.tempIndices:
            self.tempIndices[name] += 1
        else:
            self.tempIndices[name] = 0
        return self.getTemp(name)

    def skipTemp(self, name):
        self.skippedTemps.add(self.nextTemp(name))

    def skippedTemp(self, name):
        return self.getTemp(name) in self.skippedTemps

    def getTemp(self, name):
        if name in self.tempIndices:
            return f'__{name}{self.tempIndices[name]}__'
        else:
            return None

    def prevTemp(self, name):
        temp: Optional[str] = self.getTemp(name)
        if temp and temp in self.skippedTemps:
            self.skippedTemps.remove(temp)
        if name in self.tempIndices:
            self.tempIndices[name] -= 1
            if self.tempIndices[name] < 0:
                del self.tempIndices[name]

    def useModule(self, name):
        self.module.program.importStack[-1][1] = self.lineNr
        return self.module.program.provide(name, filter=self.filterId)

    def isCall(self, node, name):
        return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and (node.func.id == name)

    def getPragmaFromExpr(self, node):
        return node.value.args if isinstance(node, ast.Expr) and self.isCall(node.value, '__pragma__') else None

    def getPragmaFromIf(self, node):
        return node.test.args if isinstance(node, ast.If) and self.isCall(node.test, '__pragma__') else None

    def visit(self, node=None):
        if node is None:
            return super().visit(node)
        super().visit(node)

    def visit_FunctionDef(self, node, anAsync=False):
        pass

    def visit_JoinStr(self, node):
        pass