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
from typing import List, Dict, Any, Optional, Callable, Set, Tuple
from org.transcrypt import utils, sourcemaps, minify, static_check, type_check

inIf: bool = False
ecom: bool = True
noecom: bool = False
dataClassDefaultArgTuple: List[List[Any]] = [['init', True], ['repr', True], ['eq', True], ['order', False], ['unsafe_hash', False], ['frozen', False]]
"\nAll files required for deployment are placed in subdirectory __target__ of the application.\nEach module has an unambiguous dotted path, always from one of the module roots, never relative.\nDotted paths are translated to dotted filenames.\nThe __runtime__ module is just another Python module with lots of JS code inside __pragma__ ('js', '{}', include...) fragments,\nnamely the __core__ and __builtin__ parts.\n\nSourcemaps are generated per module.\nThere's no need for a link with other modules.\nSince import paths are static, names of minified JS files simply end on .js just like non-minified files, so not on .min.js.\nSourcemaps are named <module name>.map.\n"

class Program:
    moduleSearchDirs: List[str]
    symbols: Set[str]
    envir: Any
    javascriptVersion: int
    moduleDict: Dict[str, 'Module']
    importStack: List[List[Any]]
    sourcePrepath: str
    sourceDir: str
    mainModuleName: str
    targetDir: str
    projectPath: str
    optionsChanged: bool
    runtimeModuleName: str
    searchedModulePaths: List[str]

    def __init__(self, moduleSearchDirs: List[str], symbols: Set[str], envir: Any) -> None:
        utils.setProgram(self)
        self.moduleSearchDirs = moduleSearchDirs
        self.symbols = symbols
        self.envir = envir
        self.javascriptVersion = int(utils.commandArgs.esv) if utils.commandArgs.esv else 6
        self.moduleDict: Dict[str, Module] = {}
        self.importStack: List[List[Any]] = []
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
        project: Dict[str, Any] = {
            'options': utils.commandArgs.projectOptions,
            'modules': [{'source': module.sourcePath, 'target': module.targetPath} for module in self.moduleDict.values()]
        }
        with utils.create(self.projectPath) as projectFile:
            json.dump(project, projectFile)

    def provide(self, moduleName: str, __moduleName__: Optional[str] = None, filter: Optional[Callable[[str], str]] = None) -> 'Module':
        if moduleName in self.moduleDict:
            return self.moduleDict[moduleName]
        else:
            return Module(self, moduleName, __moduleName__, filter)

class Module:
    program: Program
    name: str
    __name__: str
    sourceDir: str
    sourcePrename: str
    sourcePrepath: str
    pythonSourcePath: str
    javascriptSourcePath: str
    targetPrepath: str
    targetName: str
    targetPath: str
    prettyTargetName: str
    prettyTargetPath: str
    importRelPath: str
    treePath: str
    mapPath: str
    prettyMapPath: str
    shrinkMapName: str
    shrinkMapPath: str
    mapSourcePath: str
    mapRef: str
    isJavascriptOnly: bool
    sourcePath: str
    sourceMapper: sourcemaps.SourceMapper
    targetCode: str
    prettyTargetCode: str
    importedModuleNames: List[str]
    exportedNames: Set[str]
    generator: 'Generator'

    def __init__(self, program: Program, name: str, __name__: Optional[str], filter: Optional[Callable[[str], str]]) -> None:
        self.program = program
        self.name = name
        self.__name__ = __name__ if __name__ else self.name
        self.findPaths(filter)
        self.program.importStack.append([self, None])
        self.program.moduleDict[self.name] = self
        self.sourceMapper: sourcemaps.SourceMapper = sourcemaps.SourceMapper(
            self.name, 
            self.program.targetDir, 
            not utils.commandArgs.nomin, 
            utils.commandArgs.dmap
        )
        if utils.commandArgs.build or self.program.optionsChanged or (not os.path.isfile(self.targetPath)) or (os.path.getmtime(self.sourcePath) > os.path.getmtime(self.targetPath)):
            if self.isJavascriptOnly:
                self.loadJavascript()
                javascriptDigest: Any = utils.digestJavascript(
                    self.targetCode, 
                    self.program.symbols, 
                    not utils.commandArgs.dnostrip, 
                    False
                )
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
                javascriptDigest = utils.digestJavascript(
                    self.targetCode, 
                    self.program.symbols, 
                    False, 
                    self.generator.allowDebugMap
                )
            utils.log(True, 'Saving target code in: {}\n', self.targetPath)
            filePath: str = self.targetPath if utils.commandArgs.nomin else self.prettyTargetPath
            with utils.create(filePath) as aFile:
                aFile.write(self.targetCode)
            if not utils.commandArgs.nomin:
                utils.log(True, 'Saving minified target code in: {}\n', self.targetPath)
                minify.run(
                    self.program.targetDir, 
                    self.prettyTargetName, 
                    self.targetName, 
                    mapFileName=self.shrinkMapName if utils.commandArgs.map else None
                )
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
            self.targetCode = open(self.targetPath, 'r').read()
            javascriptDigest: Any = utils.digestJavascript(
                self.targetCode, 
                self.program.symbols, 
                True, 
                False, 
                refuseIfAppearsMinified=True
            )
            if not javascriptDigest:
                minify.run(
                    self.program.targetDir, 
                    self.targetName, 
                    self.prettyTargetName, 
                    prettify=True
                )
                self.prettyTargetCode = open(self.prettyTargetPath, 'r').read()
                javascriptDigest = utils.digestJavascript(
                    self.prettyTargetCode, 
                    self.program.symbols, 
                    True, 
                    False
                )
        self.targetCode = javascriptDigest.digestedCode
        self.importedModuleNames = javascriptDigest.importedModuleNames
        self.exportedNames = javascriptDigest.exportedNames
        for importedModuleName in self.importedModuleNames:
            self.program.searchedModulePaths = []
            self.program.provide(importedModuleName)
        utils.tryRemove(self.prettyTargetPath)
        utils.tryRemove(self.shrinkMapPath)
        utils.tryRemove(self.prettyMapPath)
        self.program.importStack.pop()

    def findPaths(self, filter: Optional[Callable[[str], str]]) -> None:
        rawRelSourceSlug: str = self.name.replace('.', '/')
        relSourceSlug: str = filter(rawRelSourceSlug) if filter and utils.commandArgs.alimod else rawRelSourceSlug
        "\n        # BEGIN DEBUGGING CODE\n        print ()\n        print ('Raw slug   :', rawRelSourceSlug)\n        print ('Cooked slug:', relSourceSlug)\n        print ()\n        # END DEBUGGING CODE\n        "
        for searchDir in self.program.moduleSearchDirs:
            sourceSlug: str = f'{searchDir}/{relSourceSlug}'
            if os.path.isdir(sourceSlug):
                self.sourceDir = sourceSlug
                self.sourcePrename = '__init__'
            else:
                self.sourceDir, self.sourcePrename = sourceSlug.rsplit('/', 1)
            self.sourcePrepath = f'{self.sourceDir}/{self.sourcePrename}'
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
            raise utils.Error(message="\n\tImport error, can't find any of:\n\t\t{}\n".format('\n\t\t'.join(self.program.searchedModulePaths)))

    def generateJavascriptAndPrettyMap(self) -> None:
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

    def loadJavascript(self) -> None:
        with tokenize.open(self.sourcePath) as sourceFile:
            self.targetCode: str = sourceFile.read()

    def parse(self) -> None:
        def pragmasFromComments(sourceCode: str) -> str:
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
                if tokenIndex - pragmaIndex == 2:
                    pragmaKind = tokenString[1:-1]
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
                indentation: str
                separator: str
                tail: str
                indentation, separator, tail = sourceLines[pragmaCommentLineIndex].partition('#')
                pragma: str
                comment: str
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
                head: str
                tail: str
                head, tail = sourceLines[shortPragmaCommentLineIndex].rsplit('#', 1)
                strippedHead: str = head.lstrip()
                indent: str = head[:len(head) - len(strippedHead)]
                pragmaName: str = tail.replace(' ', '').replace('\t', '')[3:]
                if pragmaName == 'ecom':
                    sourceLines[shortPragmaCommentLineIndex] = ecom
                elif pragmaName == 'noecom':
                    sourceLines[shortPragmaCommentLineIndex] = noecom
                elif pragmaName.startswith('no'):
                    sourceLines[shortPragmaCommentLineIndex] = f"{indent}__pragma__ ('{pragmaName}'); {head}; __pragma__ ('{pragmaName[2:]}')"
                else:
                    sourceLines[shortPragmaCommentLineIndex] = f"{indent}__pragma__ ('{pragmaName}'); {head}; __pragma__ ('no{pragmaName}')"
            uncommentedSourceLines: List[str] = []
            for sourceLine in sourceLines:
                if sourceLine == ecom:
                    allowExecutableComments = True
                elif sourceLine == noecom:
                    allowExecutableComments = False
                elif allowExecutableComments:
                    lStrippedSourceLine: str = sourceLine.lstrip()
                    if not lStrippedSourceLine[:4] in {"'''?", "?'''", '"""?', '?"""'}:
                        uncommentedSourceLines.append(
                            sourceLine.replace('#?', '', 1) 
                            if lStrippedSourceLine.startswith('#?') 
                            else sourceLine
                        )
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
                    childNode.parentNode = node  # type: ignore
        except SyntaxError as syntaxError:
            utils.enhanceException(
                syntaxError, 
                lineNr=syntaxError.lineno, 
                message=f'\n\t{syntaxError.text[:syntaxError.offset].lstrip()} [<-SYNTAX FAULT] {syntaxError.text[syntaxError.offset:].rstrip()}' 
                if syntaxError.text 
                else syntaxError.args[0]
            )

    def dumpTree(self) -> None:
        utils.log(False, 'Dumping syntax tree for module: {}\n', self.sourcePath)

        def walk(name: str, value: Any, tabLevel: int) -> None:
            self.treeFragments.append('\n{0}{1}: {2} '.format(tabLevel * '\t', name, type(value).__name__))
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
    module: Module
    targetFragments: List[str]
    fragmentIndex: int
    indentLevel: int
    scopes: List[Any]
    importHeads: Set[str]
    importHoistMemos: List[Any]
    allOwnNames: Set[str]
    allImportedNames: Set[str]
    expectingNonOverloadedLhsIndex: bool
    lineNr: int
    propertyAccessorList: List[Any]
    mergeList: List[Any]
    aliases: List[Tuple[str, str]]
    idFiltering: bool
    tempIndices: Dict[str, int]
    skippedTemps: Set[str]
    stubsName: str
    nameConsts: Dict[Any, str]
    operators: Dict[Any, Tuple[Optional[str], int]]
    allowKeywordArgs: bool
    allowOperatorOverloading: bool
    allowConversionToIterable: bool
    allowConversionToTruthValue: bool
    allowKeyCheck: bool
    allowDebugMap: bool
    allowDocAttribs: bool
    allowGlobals: bool
    allowJavaScriptIter: bool
    allowJavaScriptCall: bool
    allowJavaScriptKeys: bool
    allowJavaScriptMod: bool
    allowMemoizeCalls: bool
    noskipCodeGeneration: bool
    conditionalCodeGeneration: bool
    stripTuple: bool
    stripTuples: bool
    replaceSend: bool

    def __init__(self, module: Module) -> None:
        self.module = module
        self.targetFragments = []
        self.fragmentIndex = 0
        self.indentLevel = 0
        self.scopes = []
        self.importHeads = set()
        self.importHoistMemos = []
        self.allOwnNames = set()
        self.allImportedNames = set()
        self.expectingNonOverloadedLhsIndex = False
        self.lineNr = 1
        self.propertyAccessorList = []
        self.mergeList = []
        self.aliases: List[Tuple[str, str]] = [
            ('js_and', 'and'), ('arguments', 'py_arguments'), ('js_arguments', 'arguments'),
            ('case', 'py_case'), ('clear', 'py_clear'), ('js_clear', 'clear'), ('js_conjugate', 'conjugate'),
            ('default', 'py_default'), ('del', 'py_del'), ('js_del', 'del'), ('js_from', 'from'),
            ('false', 'py_false'), ('js_from', 'from'), ('js_infinity', 'Infinity'), ('js_is', 'is'),
            ('js_isnan', 'isNaN'), ('js_iter', 'iter'), ('js_items', 'items'), ('js_keys', 'keys'),
            ('js_name', 'name'), ('js_nano', 'NaN'), ('js_new', 'new'), ('js_next', 'next'),
            ('js_not', 'not'), ('js_or', 'or'), ('js_pop', 'pop'), ('js_popitem', 'popitem'),
            ('js_replace', 'replace'), ('js_selector', 'selector'), ('js_sort', 'sort'),
            ('js_split', 'split'), ('js_switch', 'switch'), ('js_type', 'type'),
            ('js_typeerror', 'TypeError'), ('js_update', 'update'), ('js_values', 'values'),
            ('js_reversed', 'reversed'), ('js_setdefault', 'setdefault'), ('js_super', 'super'),
            ('js_true', 'py_true'), ('js_undefined', 'undefined')
        ]
        self.idFiltering = True
        self.tempIndices = {}
        self.skippedTemps = set()
        self.stubsName: str = f'org.{self.module.program.envir.transpiler_name}.stubs.'
        self.nameConsts: Dict[Any, str] = {None: 'null', True: 'true', False: 'false'}
        "\n        The precendences explicitly given as integers in the list below are JavaScript precedences as specified by:\n        https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/Operator_Precedence .\n\n        Python precedences are implicitly present in branch ordering of the AST generated by CPython's parser.\n        "
        self.operators: Dict[Any, Tuple[Optional[str], int]] = {
            ast.Not: ('!', 16), ast.Invert: ('~', 16), ast.UAdd: ('+', 16), ast.USub: ('-', 16),
            ast.Pow: (None, 15), ast.Mult: ('*', 14), ast.MatMult: (None, 14), ast.Div: ('/', 14),
            ast.FloorDiv: (None, 14), ast.Mod: ('%', 14), ast.Add: ('+', 13), ast.Sub: ('-', 13),
            ast.LShift: ('<<', 12), ast.RShift: ('>>', 12), ast.Lt: ('<', 11), ast.LtE: ('<=', 11),
            ast.Gt: ('>', 11), ast.GtE: ('>=', 11), ast.In: (None, 11), ast.NotIn: (None, 11),
            ast.Eq: ('==', 10), ast.NotEq: ('!=', 10), ast.Is: ('===', 10), ast.IsNot: ('!==', 10),
            ast.BitAnd: ('&', 9), ast.BitOr: ('|', 8), ast.BitXor: ('^', 7), ast.And: ('&&', 6),
            ast.Or: ('||', 5)
        }
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
            super().__init__()
            self.visit(module.parseTree)
            self.targetFragments.append(self.lineNrString)
        except Exception as exception:
            utils.enhanceException(exception, lineNr=self.lineNr)
        if self.tempIndices:
            raise utils.Error(message='\n\tTemporary variables leak in code generator: {}'.format(self.tempIndices))

    def visitSubExpr(self, node: ast.AST, child: ast.AST) -> None:
        def getPriority(exprNode: ast.AST) -> int:
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

    def filterId(self, qualifiedId: str) -> str:
        if not self.idFiltering or (qualifiedId.startswith('__') and qualifiedId.endswith('__')):
            return qualifiedId
        else:
            for alias in self.aliases:
                qualifiedId = re.sub(
                    f'(^|(?P<preDunder>__)|(?<=[./])){re.escape(alias[0])}((?P<postDunder>__)|(?=[./])|$)', 
                    lambda matchObject: ('=' if matchObject.group('preDunder') else '') + alias[1] + ('=' if matchObject.group('postDunder') else ''), 
                    qualifiedId
                )
                qualifiedId = re.sub(
                    f'(^|(?<=[./=])){re.escape(alias[0])}((?=[./=])|$)', 
                    alias[1], 
                    qualifiedId
                )
            return qualifiedId.replace('=', '')

    def tabs(self, indentLevel: Optional[int] = None) -> str:
        if indentLevel is None:
            indentLevel = self.indentLevel
        return indentLevel * '\t'

    def emit(self, fragment: str, *formatter: Any) -> None:
        if not self.targetFragments or (self.targetFragments and self.targetFragments[self.fragmentIndex - 1].endswith('\n')):
            self.targetFragments.insert(self.fragmentIndex, self.tabs())
            self.fragmentIndex += 1
        fragment = fragment[:-1].replace('\n', '\n' + self.tabs()) + fragment[-1]
        self.targetFragments.insert(self.fragmentIndex, fragment.format(*formatter).replace('\n', self.lineNrString + '\n'))
        self.fragmentIndex += 1

    def indent(self) -> None:
        self.indentLevel += 1

    def dedent(self) -> None:
        self.indentLevel -= 1

    def inscope(self, node: ast.AST) -> None:
        self.scopes.append(utils.Any(node=node, nonlocals=set(), containsYield=False))

    def descope(self) -> None:
        self.scopes.pop()

    def getScope(self, *nodeTypes: type) -> Optional[Any]:
        if nodeTypes:
            for scope in reversed(self.scopes):
                if type(scope.node) in nodeTypes:
                    return scope
        else:
            return self.scopes[-1]

    def getAdjacentClassScopes(self, inMethod: bool = False) -> List[Any]:
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

    def emitComma(self, index: int, blank: bool = True) -> None:
        if self.noskipCodeGeneration and self.conditionalCodeGeneration and index:
            self.emit(', ' if blank else ',')

    def emitBeginTruthy(self) -> None:
        if self.allowConversionToTruthValue:
            self.emit('__t__ (')

    def emitEndTruthy(self) -> None:
        if self.allowConversionToTruthValue:
            self.emit(')')

    def adaptLineNrString(self, node: Optional[ast.AST] = None, offset: int = 0) -> None:
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
            self.lineNrString = ''

    def isCommentString(self, statement: ast.stmt) -> bool:
        return (
            isinstance(statement, ast.Expr) and 
            isinstance(statement.value, ast.Constant) and 
            (type(statement.value.value) == str)
        )

    def emitBody(self, body: List[ast.stmt]) -> None:
        for statement in body:
            if self.isCommentString(statement):
                pass
            else:
                self.visit(statement)
                self.emit(';\n')

    def emitSubscriptAssign(self, target: ast.Subscript, value: ast.AST, emitPathIndices: Callable[[], None] = lambda: None) -> None:
        if isinstance(target.slice, ast.Index):
            if isinstance(target.slice, ast.Tuple):
                self.visit(target.value)
                self.emit('.__setitem__ (')
                self.stripTuple = True
                self.visit(target.slice)
                self.emit(', ')
                self.visit(value)
                emitPathIndices()
                self.emit(')')
            elif self.allowOperatorOverloading:
                self.emit('__setitem__ (')
                self.visit(target.value)
                self.emit(', ')
                self.visit(target.slice)
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

    def nextTemp(self, name: str) -> str:
        if name in self.tempIndices:
            self.tempIndices[name] += 1
        else:
            self.tempIndices[name] = 0
        return self.getTemp(name)

    def skipTemp(self, name: str) -> None:
        self.skippedTemps.add(self.nextTemp(name))

    def skippedTemp(self, name: str) -> bool:
        return self.getTemp(name) in self.skippedTemps

    def getTemp(self, name: str) -> Optional[str]:
        if name in self.tempIndices:
            return f'__{name}{self.tempIndices[name]}__'
        else:
            return None

    def prevTemp(self, name: str) -> None:
        if self.getTemp(name) in self.skippedTemps:
            self.skippedTemps.remove(self.getTemp(name))
        self.tempIndices[name] -= 1
        if self.tempIndices[name] < 0:
            del self.tempIndices[name]

    def useModule(self, name: str) -> 'Module':
        self.module.program.importStack[-1][1] = self.lineNr
        return self.module.program.provide(name, filter=self.filterId)

    def isCall(self, node: ast.AST, name: str) -> bool:
        return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and (node.func.id == name)

    def getPragmaFromExpr(self, node: ast.AST) -> Optional[List[ast.AST]]:
        return node.value.args if isinstance(node, ast.Expr) and self.isCall(node.value, '__pragma__') else None

    def getPragmaFromIf(self, node: ast.AST) -> Optional[List[ast.AST]]:
        return node.test.args if isinstance(node, ast.If) and self.isCall(node.test, '__pragma__') else None

    def visit(self, node: ast.AST) -> Any:
        try:
            self.lineNr = getattr(node, 'lineno', self.lineNr)
        except:
            pass
        pragmaInIf = self.getPragmaFromIf(node)
        pragmaInExpr = self.getPragmaFromExpr(node)
        if pragmaInIf:
            if pragmaInIf[0].s == 'defined':
                for symbol in pragmaInIf[1:]:
                    if symbol.s in self.module.program.symbols:
                        definedInIf: bool = True
                        break
                else:
                    definedInIf = False
        elif pragmaInExpr:
            if pragmaInExpr[0].s == 'skip':
                self.noskipCodeGeneration = False
            elif pragmaInExpr[0].s == 'noskip':
                self.noskipCodeGeneration = True
            if pragmaInExpr[0].s in ('ifdef', 'ifndef'):
                definedInExpr: bool = eval(
                    compile(ast.Expression(pragmaInExpr[1]), '<string>', 'eval'), 
                    {}, 
                    {'__envir__': self.module.program.envir}
                ) in self.module.program.symbols
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
                if definedInIf:
                    self.emitBody(node.body)
            else:
                super().visit(node)

    def visit_arg(self, node: ast.arg) -> None:
        self.emit(self.filterId(node.arg))

    def visit_arguments(self, node: ast.arguments) -> None:
        self.emit('(')
        for index, arg in enumerate(node.args):
            self.emitComma(index)
            self.visit(arg)
        self.emit(') {{\n')
        self.indent()
        for arg, expr in reversed(list(zip(reversed(node.args), reversed(node.defaults)))):
            if expr:
                self.emit('if (typeof {0} == \'undefined\' || ({0} != null && {0}.hasOwnProperty ("__kwargtrans__"))) {{;\n', self.filterId(arg.arg))
                self.indent()
                self.emit('var {} = ', self.filterId(arg.arg))
                self.visit(expr)
                self.emit(';\n')
                self.dedent()
                self.emit('}};\n')
        for arg, expr in zip(node.kwonlyargs, node.kw_defaults):
            if expr:
                self.emit('var {} = ', self.filterId(arg.arg))
                self.visit(expr)
                self.emit(';\n')
        if self.allowKeywordArgs:
            if node.kwarg:
                self.emit('var {} = dict ();\n', self.filterId(node.kwarg.arg))
            self.emit('if (arguments.length) {{\n')
            self.indent()
            self.emit('var {} = arguments.length - 1;\n', self.nextTemp('ilastarg'))
            self.emit('if (arguments [{}] && arguments [{}].hasOwnProperty ("__kwargtrans__")) {{\n', self.getTemp('ilastarg'), self.getTemp('ilastarg'))
            self.indent()
            self.emit('var {} = arguments [{}--];\n', self.nextTemp('allkwargs'), self.getTemp('ilastarg'))
            self.emit('for (var {} in {}) {{\n', self.nextTemp('attrib'), self.getTemp('allkwargs'))
            self.indent()
            if node.args + node.kwonlyargs or node.kwarg:
                self.emit('switch ({}) {{\n', self.getTemp('attrib'))
                self.indent()
                for arg in node.args + node.kwonlyargs:
                    self.emit("case '{0}': var {0} = {1} [{2}]; break;\n", self.filterId(arg.arg), self.getTemp('allkwargs'), self.getTemp('attrib'))
                if node.kwarg:
                    self.emit('default: {0} [{1}] = {2} [{1}];\n', self.filterId(node.kwarg.arg), self.getTemp('attrib'), self.getTemp('allkwargs'))
                self.dedent()
                self.emit('}}\n')
            self.prevTemp('allkwargs')
            self.prevTemp('attrib')
            self.dedent()
            self.emit('}}\n')
            if node.kwarg:
                self.emit('delete {}.__kwargtrans__;\n', self.filterId(node.kwarg.arg))
            self.dedent()
            self.emit('}}\n')
            if node.vararg:
                self.emit('var {} = tuple ([].slice.apply (arguments).slice ({}, {} + 1));\n', self.filterId(node.vararg.arg), len(node.args), self.getTemp('ilastarg'))
            self.prevTemp('ilastarg')
            self.dedent()
            self.emit('}}\n')
        elif node.vararg:
            self.emit('var {} = tuple ([].slice.apply (arguments).slice ({}));\n', self.filterId(node.vararg.arg), len(node.args))

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(ast.Assign(targets=[node.target], value=node.value))

    def visit_Assert(self, node: ast.Assert) -> None:
        if utils.commandArgs.dassert:
            self.emit('assert (')
            self.visit(node.test)
            if node.msg:
                self.emit(', ')
                self.visit(node.msg)
            self.emit(');\n')

    def visit_Assign(self, node: ast.Assign) -> None:
        self.adaptLineNrString(node)
        targetLeafs = (ast.Attribute, ast.Subscript, ast.Name)

        def assignTarget(target: ast.AST, value: ast.AST, pathIndices: List[Any] = []) -> None:
            def emitPathIndices() -> None:
                if pathIndices:
                    self.emit(' ')
                    for pathIndex in pathIndices:
                        self.emit(f'[{pathIndex}]')
                else:
                    pass
            if isinstance(target, ast.Subscript):
                self.emitSubscriptAssign(target, value, emitPathIndices)
            elif isinstance(target, ast.Name) and self.isCall(value, 'property') and isinstance(target, ast.Name):
                self.emit("Object.defineProperty ({}, '{}', ".format(self.getScope().node.name, target.id))
                self.visit(value)
                self.emit(')')
            else:
                if isinstance(target, ast.Name):
                    if isinstance(self.getScope().node, ast.ClassDef) and target.id != self.getTemp('left'):
                        self.emit('{}.{}'.format('.'.join([scope.node.name for scope in self.getAdjacentClassScopes()]), target.id))
                    elif target.id in self.getScope().nonlocals:
                        pass
                    else:
                        if isinstance(self.getScope().node, ast.Module):
                            if hasattr(node, 'parentNode') and isinstance(node.parentNode, ast.Module) and (target.id not in self.allOwnNames):
                                self.emit('export ')
                        self.emit('var ')
                self.visit(target)
                self.emit(' = ')
                self.visit(value)
                emitPathIndices()

        def walkTarget(expr: ast.AST, pathIndices: List[Any]) -> None:
            if isinstance(expr, targetLeafs):
                self.emit(';\n')
                assignTarget(expr, ast.Name(id=self.getTemp('left'), ctx=ast.Load()), pathIndices)
            else:
                pathIndices.append(None)
                if isinstance(expr, ast.Tuple):
                    for index, elt in enumerate(expr.elts):
                        pathIndices[-1] = index
                        walkTarget(elt, pathIndices)
                pathIndices.pop()

        def getIsPropertyAssign(value: ast.AST) -> bool:
            if self.isCall(value, 'property'):
                return True
            else:
                try:
                    return getIsPropertyAssign(value.elts[0])
                except:
                    return False
        isPropertyAssign: bool = isinstance(self.getScope().node, ast.ClassDef) and getIsPropertyAssign(node.value)
        if len(node.targets) == 1 and isinstance(node.targets[0], targetLeafs):
            assignTarget(node.targets[0], node.value)
        else:
            self.visit(ast.Assign(targets=[ast.Name(id=self.nextTemp('left'), ctx=ast.Store())], value=node.value))
            for expr in node.targets:
                walkTarget(expr, [])
            self.prevTemp('left')

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if type(node.value) in (ast.BinOp, ast.BoolOp, ast.Compare):
            self.emit('(')
        self.visit(node.value)
        if type(node.value) in (ast.BinOp, ast.BoolOp, ast.Compare):
            self.emit(')')
        self.emit('.{}', self.filterId(node.attr))

    def visit_Await(self, node: ast.Await) -> None:
        self.emit('await ')
        self.visit(node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if self.allowOperatorOverloading:
            rhsFunctionName: str = self.filterId(
                '__ipow__' if isinstance(node.op, ast.Pow) else 
                '__imatmul__' if isinstance(node.op, ast.MatMult) else 
                '__ijsmod__' if self.allowJavaScriptMod and isinstance(node.op, ast.Mod) else 
                '__imod__' if isinstance(node.op, ast.Mod) else 
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
            rhsCall: ast.Call = ast.Call(
                func=ast.Name(id=rhsFunctionName, ctx=ast.Load()), 
                args=[node.target, node.value], 
                keywords=[]
            )
            if isinstance(node.target, ast.Subscript):
                self.emitSubscriptAssign(node.target, rhsCall)
            else:
                if isinstance(node.target, ast.Name) and (node.target.id not in self.getScope().nonlocals):
                    self.emit('var ')
                self.visit(node.target)
                self.emit(' = ')
                self.visit(rhsCall)
        elif isinstance(node.op, (ast.FloorDiv, ast.MatMult, ast.Pow)) or (isinstance(node.op, ast.Mod) and (not self.allowJavaScriptMod)) or (isinstance(node.target, ast.Subscript) and isinstance(node.target.slice, ast.Tuple)):
            self.visit(ast.Assign(targets=[node.target], value=ast.BinOp(left=node.target, op=node.op, right=node.value)))
        else:
            self.expectingNonOverloadedLhsIndex = True
            self.visit(node.target)
            if isinstance(node.value, ast.Constant) and node.value.value == 1:
                if isinstance(node.op, ast.Add):
                    self.emit('++')
                    return
                elif isinstance(node.op, ast.Sub):
                    self.emit('--')
                    return
            elif isinstance(node.value, ast.UnaryOp) and isinstance(node.value.operand, ast.Constant) and (node.value.operand.value == 1):
                if isinstance(node.op, ast.Add):
                    if isinstance(node.value.op, ast.UAdd):
                        self.emit('++')
                        return
                    elif isinstance(node.value.op, ast.USub):
                        self.emit('--')
                        return
                elif isinstance(node.op, ast.Sub):
                    if isinstance(node.value.op, ast.UAdd):
                        self.emit('--')
                        return
                    elif isinstance(node.value.op, ast.USub):
                        self.emit('++')
                        return
            self.emit(' {}= ', self.operators[type(node.op)][0])
            self.visit(node.value)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.FloorDiv):
            if self.allowOperatorOverloading:
                self.emit('__floordiv__ (')
                self.visitSubExpr(node, node.left)
                self.emit(', ')
                self.visitSubExpr(node, node.right)
                self.emit(')')
            else:
                self.emit('Math.floor (')
                self.visitSubExpr(node, node.left)
                self.emit(' / ')
                self.visitSubExpr(node, node.right)
                self.emit(')')
        elif isinstance(node.op, (ast.Pow, ast.MatMult)) or (isinstance(node.op, ast.Mod) and (self.allowOperatorOverloading or not self.allowJavaScriptMod)) or (isinstance(node.op, (ast.Mult, ast.Div, ast.Add, ast.Sub, ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd)) and self.allowOperatorOverloading):
            op_map = {
                ast.FloorDiv: '__floordiv__',
                ast.Pow: '__pow__',
                ast.MatMult: '__matmul__',
                ast.Mod: '__jsmod__' if self.allowJavaScriptMod else '__mod__',
                ast.Mult: '__mul__',
                ast.Div: '__truediv__',
                ast.Add: '__add__',
                ast.Sub: '__sub__',
                ast.LShift: '__lshift__',
                ast.RShift: '__rshift__',
                ast.BitOr: '__or__',
                ast.BitXor: '__xor__',
                ast.BitAnd: '__and__'
            }
            func_name = self.filterId(op_map.get(type(node.op), 'Never here'))
            self.emit('{} (', func_name)
            self.visit(node.left)
            self.emit(', ')
            self.visit(node.right)
            self.emit(')')
        else:
            self.visitSubExpr(node, node.left)
            self.emit(f' {self.operators[type(node.op)][0]} ')
            self.visitSubExpr(node, node.right)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        for index, value in enumerate(node.values):
            if index:
                self.emit(f' {self.operators[type(node.op)][0]} ')
            if index < len(node.values) - 1:
                self.emitBeginTruthy()
            self.visitSubExpr(node, value)
            if index < len(node.values) - 1:
                self.emitEndTruthy()

    def visit_Break(self, node: ast.Break) -> None:
        if not self.skippedTemp('break'):
            self.emit('{} = true;\n', self.getTemp('break'))
        self.emit('break')

    def visit_Call(self, node: ast.Call, dataClassArgDict: Optional[Dict[str, Any]] = None) -> None:
        self.adaptLineNrString(node)

        def emitKwargTrans() -> None:
            self.emit('__kwargtrans__ (')
            hasSeparateKeyArgs: bool = False
            hasKwargs: bool = False
            for keyword in node.keywords:
                if keyword.arg:
                    hasSeparateKeyArgs = True
                else:
                    hasKwargs = True
                    break
            if hasSeparateKeyArgs:
                if hasKwargs:
                    self.emit('__mergekwargtrans__ (')
                self.emit('{{')
            for keywordIndex, keyword in enumerate(node.keywords):
                if keyword.arg:
                    self.emitComma(keywordIndex)
                    self.emit('{}: ', self.filterId(keyword.arg))
                    self.visit(keyword.value)
                else:
                    if hasSeparateKeyArgs:
                        self.emit('}}, ')
                    self.visit(keyword.value)
            if hasSeparateKeyArgs:
                if hasKwargs:
                    self.emit(')')
                else:
                    self.emit('}}')
            self.emit(')')

        def include(fileName: str) -> str:
            try:
                searchedIncludePaths: List[str] = []
                for searchDir in self.module.program.moduleSearchDirs:
                    filePath: str = f'{searchDir}/{fileName}'
                    if os.path.isfile(filePath):
                        includedCode: str = tokenize.open(filePath).read()
                        if fileName.endswith('.js'):
                            includedCode = utils.digestJavascript(
                                includedCode, 
                                self.module.program.symbols, 
                                not utils.commandArgs.dnostrip or utils.commandArgs.anno, 
                                self.allowDebugMap
                            ).digestedCode
                        return includedCode
                    else:
                        searchedIncludePaths.append(filePath)
                else:
                    raise utils.Error(
                        lineNr=self.lineNr, 
                        message="\n\tAttempt to include file: {}\n\tCan't find any of:\n\t\t{}\n".format(
                            node.args[0], 
                            '\n\t\t'.join(searchedIncludePaths)
                        )
                    )
            except:
                print(traceback.format_exc())

        if isinstance(node.func, ast.Name):
            if node.func.id == 'type':
                self.emit('py_typeof (')
                self.visit(node.args[0])
                self.emit(')')
                return
            elif node.func.id == 'property':
                self.emit('{0}.call ({1}, {1}.{2}'.format(node.func.id, self.getScope(ast.ClassDef).node.name, self.filterId(node.args[0].id)))
                if len(node.args) > 1:
                    self.emit(', {}.{}'.format(self.getScope(ast.ClassDef).node.name, node.args[1].id))
                self.emit(')')
                return
            elif node.func.id == 'globals':
                self.emit('__all__')
                return
            elif node.func.id == '__pragma__':
                if node.args[0].s == 'alias':
                    self.aliases.insert(0, (node.args[1].s, node.args[2].s))
                elif node.args[0].s == 'noalias':
                    if len(node.args) == 1:
                        self.aliases = []
                    else:
                        for index in reversed(range(len(self.aliases))):
                            if self.aliases[index][0] == node.args[1].s:
                                self.aliases.pop(index)
                elif node.args[0].s == 'noanno':
                    self.allowDebugMap = False
                elif node.args[0].s == 'fcall':
                    self.allowMemoizeCalls = True
                elif node.args[0].s == 'nofcall':
                    self.allowMemoizeCalls = False
                elif node.args[0].s == 'docat':
                    self.allowDocAttribs = True
                elif node.args[0].s == 'nodocat':
                    self.allowDocAttribs = False
                elif node.args[0].s == 'iconv':
                    self.allowConversionToIterable = True
                elif node.args[0].s == 'noiconv':
                    self.allowConversionToIterable = False
                elif node.args[0].s == 'jsiter':
                    self.allowJavaScriptIter = True
                elif node.args[0].s == 'nojsiter':
                    self.allowJavaScriptIter = False
                elif node.args[0].s == 'jscall':
                    self.allowJavaScriptCall = True
                elif node.args[0].s == 'nojscall':
                    self.allowJavaScriptCall = False
                elif node.args[0].s == 'jskeys':
                    self.allowJavaScriptKeys = True
                elif node.args[0].s == 'nojskeys':
                    self.allowJavaScriptKeys = False
                elif node.args[0].s == 'keycheck':
                    self.allowKeyCheck = True
                elif node.args[0].s == 'nokeycheck':
                    self.allowKeyCheck = False
                elif node.args[0].s == 'jsmod':
                    self.allowJavaScriptMod = True
                elif node.args[0].s == 'nojsmod':
                    self.allowJavaScriptMod = False
                elif node.args[0].s == 'gsend':
                    self.replaceSend = True
                elif node.args[0].s == 'nogsend':
                    self.replaceSend = False
                elif node.args[0].s == 'tconv':
                    self.allowConversionToTruthValue = True
                elif node.args[0].s == 'notconv':
                    self.allowConversionToTruthValue = False
                elif node.args[0].s == 'run':
                    pass
                elif node.args[0].s == 'norun':
                    pass
                elif node.args[0].s == 'js':
                    try:
                        try:
                            code = node.args[1].s.format(*[
                                eval(compile(ast.Expression(arg), '<string>', 'eval'), {}, {'__include__': include}) for arg in node.args[2:]
                            ])
                        except:
                            code = node.args[2].s
                        for line in code.split('\n'):
                            self.emit('{}\n', line)
                    except:
                        print(traceback.format_exc())
                elif node.args[0].s == 'xtrans':
                    try:
                        sourceCode: str = node.args[2].s.format(*[
                            eval(compile(ast.Expression(arg), '<string>', 'eval'), {}, {'__include__': include}) for arg in node.args[3:]
                        ])
                        workDir: str = '.'
                        for keyword in node.keywords:
                            if keyword.arg == 'cwd':
                                workDir = keyword.value.s
                        process: subprocess.Popen = subprocess.Popen(
                            shlex.split(node.args[1].s), 
                            stdin=subprocess.PIPE, 
                            stdout=subprocess.PIPE, 
                            cwd=workDir
                        )
                        process.stdin.write(sourceCode.encode('utf8'))
                        process.stdin.close()
                        while process.returncode is None:
                            process.poll()
                        targetCode: str = process.stdout.read().decode('utf8').replace('\r\n', '\n')
                        for line in targetCode.split('\n'):
                            self.emit('{}\n', line)
                    except:
                        print(traceback.format_exc())
                elif node.args[0].s == 'xpath':
                    self.module.program.moduleSearchDirs[1:1] = [elt.s for elt in node.args[1].elts]
                else:
                    raise utils.Error(lineNr=self.lineNr, message='\n\tUnknown pragma: {}'.format(
                        node.args[0].value if isinstance(node.args[0], ast.Constant) else node.args[0]
                    ))
                return
            elif node.func.id == '__new__':
                self.emit('new ')
                self.visit(node.args[0])
                return
            elif node.func.id == '__typeof__':
                self.emit('typeof ')
                self.visit(node.args[0])
                return
            elif node.func.id == '__preinc__':
                self.emit('++')
                self.visit(node.args[0])
                return
            elif node.func.id == '__postinc__':
                self.visit(node.args[0])
                self.emit('++')
                return
            elif node.func.id == '__predec__':
                self.emit('--')
                self.visit(node.args[0])
                return
            elif node.func.id == '__postdec__':
                self.visit(node.args[0])
                self.emit('--')
                return
        elif isinstance(node.func, ast.Attribute) and node.func.attr == 'conjugate':
            try:
                self.visit(ast.Call(
                    func=ast.Name(id='__conj__', ctx=ast.Load()), 
                    args=[node.func.value], 
                    keywords=[]
                ))
                return
            except:
                print(traceback.format_exc())
        elif isinstance(node.func, ast.Attribute) and self.replaceSend and (node.func.attr == 'send'):
            self.emit('(function () {{return ')
            self.visit(ast.Attribute(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id=node.func.value.id, ctx=ast.Load()), 
                        attr='js_next', 
                        ctx=ast.Load()
                    ), 
                    args=node.args, 
                    keywords=node.keywords
                ), 
                attr='value', 
                ctx=ast.Load()
            ))
            self.emit('}}) ()')
            return
        elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Call) and isinstance(node.func.value.func, ast.Name) and (node.func.value.func.id == 'super'):
            if node.func.value.args or node.func.value.keywords:
                raise utils.Error(lineNr=self.lineNr, message="\n\tBuilt in function 'super' with arguments not supported")
            else:
                self.visit(ast.Call(
                    func=ast.Call(
                        func=ast.Name(id='__super__', ctx=ast.Load()), 
                        args=[ast.Name(id='.'.join([scope.node.name for scope in self.getAdjacentClassScopes(True)]), ctx=ast.Load())], 
                        keywords=[]
                    ), 
                    args=[ast.Name(id='self', ctx=ast.Load())] + node.args, 
                    keywords=node.keywords
                ))
                return
        if self.allowOperatorOverloading and (not (isinstance(node.func, ast.Name) and node.func.id == '__call__')):
            if isinstance(node.func, ast.Attribute):
                self.emit('(function () {{\n')
                self.inscope(ast.FunctionDef())
                self.indent()
                self.emit('var {} = ', self.nextTemp('accu'))
                self.visit(node.func.value)
                self.emit(';\n')
                self.emit('return ')
                self.visit(ast.Call(
                    func=ast.Name(id='__call__', ctx=ast.Load()), 
                    args=[
                        ast.Attribute(value=ast.Name(id=self.getTemp('accu'), ctx=ast.Load()), attr=node.func.attr, ctx=ast.Load()), 
                        ast.Name(id=self.getTemp('accu'), ctx=ast.Load())
                    ] + node.args, 
                    keywords=node.keywords
                ))
                self.emit(';\n')
                self.prevTemp('accu')
                self.dedent()
                self.descope()
                self.emit('}}) ()')
            else:
                self.visit(ast.Call(
                    func=ast.Name(id='__call__', ctx=ast.Load()), 
                    args=[node.func, ast.Constant(value=None)] + node.args, 
                    keywords=node.keywords
                ))
            return
        if dataClassArgDict is not None:
            dataClassArgTuple: List[List[Any]] = copy.deepcopy(dataClassDefaultArgTuple)
            for index, expr in enumerate(node.args):
                value: Optional[bool] = None
                if isinstance(expr, ast.Constant):
                    value = True if expr.value == 'True' else False if expr.value == 'False' else None
                if value is not None:
                    dataClassArgTuple[index][1] = value
                else:
                    raise utils.Error(message='Arguments to @dataclass can only be constants True or False')
            dataClassArgDict.update(dict(dataClassArgTuple))
            for keyword in node.keywords:
                dataClassArgDict[keyword.arg] = keyword.value
            return
        self.visit(node.func)
        self.emit(' (')
        for index, expr in enumerate(node.args):
            self.emitComma(index)
            if isinstance(expr, ast.Starred):
                self.emit('...')
            self.visit(expr)
        if node.keywords:
            self.emitComma(len(node.args))
            emitKwargTrans()
        self.emit(')')

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.adaptLineNrString(node)
        if isinstance(self.getScope().node, ast.Module):
            self.emit('export var {} = '.format(self.filterId(node.name)))
            self.allOwnNames.add(node.name)
        elif isinstance(self.getScope().node, ast.ClassDef):
            self.emit('\n{}:'.format(self.filterId(node.name)))
        else:
            self.emit('var {} ='.format(self.filterId(node.name)))
        isDataClass: bool = False
        if node.decorator_list:
            if isinstance(node.decorator_list[-1], ast.Name) and node.decorator_list[-1].id == 'dataclass':
                isDataClass = True
                dataClassArgDict: Dict[str, Any] = dict(dataClassDefaultArgTuple)
                node.decorator_list.pop()
            elif isinstance(node.decorator_list[-1], ast.Call) and node.decorator_list[-1].func.id == 'dataclass':
                isDataClass = True
                dataClassArgDict: Dict[str, Any] = {}
                self.visit_Call(node.decorator_list.pop(), dataClassArgDict)
        decoratorsUsed: int = 0
        if node.decorator_list:
            self.emit(' ')
            if self.allowOperatorOverloading:
                self.emit('__call__ (')
            for decorator in node.decorator_list:
                if decoratorsUsed > 0:
                    self.emit(' (')
                self.visit(decorator)
                decoratorsUsed += 1
            if self.allowOperatorOverloading:
                self.emit(', null, ')
            else:
                self.emit(' (')
        self.emit(" __class__ ('{}', [".format(self.filterId(node.name)))
        if node.bases:
            for index, expr in enumerate(node.bases):
                try:
                    self.emitComma(index)
                    self.visit(expr)
                except Exception as exception:
                    utils.enhanceException(exception, lineNr=self.lineNr, message='\n\tInvalid base class')
        else:
            self.emit('object')
        self.emit('], {{')
        self.inscope(node)
        self.indent()
        self.emit('\n__module__: __name__,')
        inlineAssigns: List[ast.stmt] = []
        propertyAssigns: List[ast.stmt] = []
        initAssigns: List[ast.stmt] = []
        delayedAssigns: List[ast.stmt] = []
        reprAssigns: List[ast.stmt] = []
        compareAssigns: List[ast.stmt] = []
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
                    if isinstance(statement.value, ast.Call) and isinstance(statement.value.func, ast.Name) and (statement.value.func.id == 'property'):
                        propertyAssigns.append(statement)
                    else:
                        inlineAssigns.append(statement)
                        self.emitComma(index, False)
                        self.emit('\n{}: '.format(self.filterId(statement.targets[0].id)))
                        self.visit(statement.value)
                        self.adaptLineNrString(statement)
                        index += 1
                else:
                    delayedAssigns.append(statement)
            elif isinstance(statement, ast.AnnAssign):
                if isinstance(statement.value, ast.Call) and isinstance(statement.value.func, ast.Name) and (statement.value.func.id == 'property'):
                    propertyAssigns.append(statement)
                    if isDataClass:
                        reprAssigns.append(statement)
                        compareAssigns.append(statement)
                elif isDataClass and isinstance(statement.annotation, ast.Name) and (statement.annotation.id != 'ClassVar'):
                    inlineAssigns.append(statement)
                    initAssigns.append(statement)
                    reprAssigns.append(statement)
                    compareAssigns.append(statement)
                    self.emitComma(index, False)
                    self.emit('\n{}: '.format(self.filterId(statement.target.id)))
                    self.visit(statement.value)
                    self.adaptLineNrString(statement)
                    index += 1
                elif isinstance(statement.target, ast.Name):
                    try:
                        inlineAssigns.append(statement)
                        self.emitComma(index, False)
                        self.emit('\n{}: '.format(self.filterId(statement.target.id)))
                        self.visit(statement.value)
                        self.adaptLineNrString(statement)
                        index += 1
                    except:
                        print(traceback.format_exc())
                else:
                    delayedAssigns.append(statement)
            elif self.getPragmaFromExpr(statement):
                self.visit(statement)
        self.dedent()
        self.emit('\n}}')
        if node.keywords:
            if node.keywords[0].arg == 'metaclass':
                self.emit(', ')
                self.visit(node.keywords[0].value)
            else:
                raise utils.Error(lineNr=self.lineNr, message='\n\tUnknown keyword argument {} definition of class {}'.format(node.keywords[0].arg, node.name))
        self.emit(')')
        if decoratorsUsed:
            self.emit(')' * decoratorsUsed)
        if self.allowDocAttribs:
            docString: Optional[str] = ast.get_docstring(node)
            if docString:
                self.emit(" .__setdoc__ ('{}')", docString.replace('\n', '\\n '))
        if isDataClass:
            nrOfFragmentsToJump: int = self.fragmentIndex - initHoistFragmentIndex
            self.fragmentIndex = initHoistFragmentIndex
            originalIndentLevel: int = self.indentLevel
            self.indentLevel = initHoistIndentLevel
            initArgs: List[Tuple[str, ast.AST]] = [
                ((initAssign.targets[0].id if isinstance(initAssign, ast.Assign) else initAssign.target.id), initAssign.value) 
                for initAssign in initAssigns
            ]
            reprNames: List[str] = [
                (reprAssign.targets[0].id if isinstance(reprAssign, ast.Assign) else reprAssign.target.id) 
                for reprAssign in reprAssigns
            ]
            compareNames: List[str] = [
                (compareAssign.targets[0].id if isinstance(compareAssign, ast.Assign) else compareAssign.target.id) 
                for compareAssign in compareAssigns
            ]
            if dataClassArgDict.get('repr', True):
                originalAllowKeywordArgs: bool = self.allowKeywordArgs
                self.allowKeywordArgs = True
                self.visit(ast.FunctionDef(
                    name='__init__', 
                    args=ast.arguments(
                        args=[ast.arg(arg='self', annotation=None)], 
                        vararg=ast.arg(arg='args', annotation=None), 
                        kwonlyargs=[], 
                        kw_defaults=[], 
                        kwarg=ast.arg(arg='kwargs', annotation=None), 
                        defaults=[]
                    ), 
                    body=[
                        ast.Expr(
                            value=ast.Call(
                                func=ast.Name(id='__pragma__', ctx=ast.Load()), 
                                args=[
                                    ast.Constant(value='js'), 
                                    ast.Constant(value='{}'), 
                                    ast.Constant(value='\nlet names = self.__initfields__.values ();\nfor (let arg of args) {\n    self [names.next () .value] = arg;\n}\nfor (let name of kwargs.py_keys ()) {\n    self [name] = kwargs [name];\n}\n                                        '.strip())
                                ], 
                                keywords=[]
                            )
                        )
                    ], 
                    decorator_list=[], 
                    returns=None, 
                    docstring=None
                ))
                self.emit(',')
                self.allowKeywordArgs = originalAllowKeywordArgs
            if dataClassArgDict.get('repr', True):
                self.visit(ast.FunctionDef(
                    name='__repr__', 
                    args=ast.arguments(
                        args=[ast.arg(arg='self', annotation=None)], 
                        vararg=None, 
                        kwonlyargs=[], 
                        kw_defaults=[], 
                        kwarg=None, 
                        defaults=[]
                    ), 
                    body=[
                        ast.Expr(
                            value=ast.Call(
                                func=ast.Name(id='__pragma__', ctx=ast.Load()), 
                                args=[
                                    ast.Constant(value='js'), 
                                    ast.Constant(value='{}'), 
                                    ast.Constant(value="\nlet names = self.__reprfields__.values ();\nlet fields = [];\nfor (let name of names) {{\n    fields.push (name + '=' + repr (self [name]));\n}}\nreturn  self.__name__ + '(' + ', '.join (fields) + ')'\n                                        ".strip())
                                ], 
                                keywords=[]
                            )
                        )
                    ], 
                    decorator_list=[], 
                    returns=None, 
                    docstring=None
                ))
                self.emit(',')
            comparatorNames: List[str] = []
            if 'eq' in dataClassArgDict:
                comparatorNames += ['__eq__', '__ne__']
            if 'order' in dataClassArgDict:
                comparatorNames += ['__lt__', '__le__', '__gt__', '__ge__']
            for comparatorName in comparatorNames:
                self.visit(ast.FunctionDef(
                    name=comparatorName, 
                    args=ast.arguments(
                        args=[
                            ast.arg(arg='self', annotation=None), 
                            ast.arg(arg='other', annotation=None)
                        ], 
                        vararg=None, 
                        kwonlyargs=[], 
                        kw_defaults=[], 
                        kwarg=None, 
                        defaults=[]
                    ), 
                    body=[
                        ast.Expr(
                            value=ast.Call(
                                func=ast.Name(id='__pragma__', ctx=ast.Load()), 
                                args=[
                                    ast.Constant(value='js'), 
                                    ast.Constant(value='{}'), 
                                    ast.Constant(value=(
                                        '\nlet names = self.__comparefields__.values ();\nlet selfFields = [];\nlet otherFields = [];\nfor (let name of names) {\n    selfFields.push (self [name]);\n    otherFields.push (other [name]);\n}\nreturn list (selfFields).' + comparatorName + '(list (otherFields));\n                                        ').strip()
                                ], 
                                keywords=[]
                            )
                        )
                    ], 
                    decorator_list=[]
                ))
                returns = (None,)
                self.emit(',')
            self.fragmentIndex += nrOfFragmentsToJump
            self.indentLevel = originalIndentLevel
        for assign in delayedAssigns + propertyAssigns:
            self.emit(';\n')
            self.visit(assign)
        self.mergeList.append(utils.Any(
            className='.'.join([scope.node.name for scope in self.getAdjacentClassScopes()]), 
            isDataClass=isDataClass, 
            reprAssigns=reprAssigns, 
            compareAssigns=compareAssigns, 
            initAssigns=initAssigns
        ))
        self.descope()

        def emitMerges() -> None:
            def emitMerge(merge: Any) -> None:
                if merge.isDataClass:
                    self.emit('\nfor (let aClass of {}.__bases__) {{\n', self.filterId(merge.className))
                    self.indent()
                    self.emit('__mergefields__ ({}, aClass);\n', self.filterId(merge.className))
                    self.dedent()
                    self.emit('}}')
                    self.emit(';\n__mergefields__ ({}, {{'.format(self.filterId(merge.className)))
                    self.emit('__reprfields__: new Set ([{}]), '.format(', '.join(f"'{name}'" for name in merge.reprAssigns)))
                    self.emit('__comparefields__: new Set ([{}]), '.format(', '.join(f"'{name}'" for name in merge.compareAssigns)))
                    self.emit('__initfields__: new Set ([{}])'.format(', '.join(f"'{name}'" for name in merge.initAssigns)))
                    self.emit('}})')
            for merge in self.mergeList:
                emitMerge(merge)
            self.mergeList = []

        def emitProperties() -> None:
            def emitProperty(className: str, propertyName: str, getterName: str, setterName: Optional[str] = None) -> None:
                self.emit("\nObject.defineProperty ({}, '{}', ".format(className, propertyName))
                if setterName:
                    self.emit('property.call ({0}, {0}.{1}, {0}.{2})'.format(className, getterName, setterName))
                else:
                    self.emit('property.call ({0}, {0}.{1})'.format(className, getterName))
                self.emit(');')
            if self.propertyAccessorList:
                self.emit(';')
            while self.propertyAccessorList:
                propertyAccessor = self.propertyAccessorList.pop()
                className: str = propertyAccessor.className
                functionName: str = propertyAccessor.functionName
                propertyName: str = functionName[5:]
                isGetter: bool = functionName[:5] == '_get_'
                for propertyAccessor2 in self.propertyAccessorList:
                    className2: str = propertyAccessor2.className
                    functionName2: str = propertyAccessor2.functionName
                    propertyName2: str = functionName2[5:]
                    isGetter2: bool = functionName2[:5] == '_get_'
                    if className == className2 and propertyName == propertyName2 and (isGetter != isGetter2):
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
                        raise utils.Error(message='\n\tProperty setter declared without getter\n')

        if isinstance(self.getScope().node, ast.ClassDef):
            emitProperties()
            emitMerges()

    def visit_FunctionDef(self, node: ast.FunctionDef, anAsync: bool = False) -> None:
        def emitScopedBody() -> None:
            self.inscope(node)
            self.emitBody(node.body)
            self.dedent()
            if self.getScope(ast.AsyncFunctionDef if anAsync else ast.FunctionDef).containsYield:
                self.targetFragments.insert(yieldStarIndex, '*')
            self.descope()

        def pushPropertyAccessor(functionName: str) -> None:
            self.propertyAccessorList.append(utils.Any(functionName=functionName, className='.'.join([scope.node.name for scope in self.getAdjacentClassScopes()])))

        nodeName: str = node.name
        if not nodeName == '__pragma__':
            isGlobal: bool = isinstance(self.getScope().node, ast.Module)
            isMethod: bool = not (isGlobal or isinstance(self.getScope().node, (ast.FunctionDef, ast.AsyncFunctionDef)))
            if isMethod:
                self.emit('\n')
            self.adaptLineNrString(node)
            decorate: bool = False
            isClassMethod: bool = False
            isStaticMethod: bool = False
            isProperty: bool = False
            getter: str = '__get__'
            if node.decorator_list:
                for decorator in node.decorator_list:
                    decoratorNode: ast.AST = decorator
                    decoratorType: type = type(decoratorNode)
                    nameCheck: str = ''
                    while decoratorType != ast.Name:
                        if decoratorType == ast.Call:
                            decoratorNode = decoratorNode.func
                        elif decoratorType == ast.Attribute:
                            nameCheck = '.' + decoratorNode.attr + nameCheck
                            decoratorNode = decoratorNode.value
                        decoratorType = type(decoratorNode)
                    nameCheck = decoratorNode.id + nameCheck
                    if nameCheck == 'classmethod':
                        isClassMethod = True
                        getter = '__getcm__'
                    elif nameCheck == 'staticmethod':
                        isStaticMethod = True
                        getter = '__getsm__'
                    elif nameCheck == 'property':
                        isProperty = True
                        nodeName = '_get_' + node.name
                        pushPropertyAccessor(nodeName)
                    elif re.match('[a-zA-Z0-9_]+\\.setter', nameCheck):
                        isProperty = True
                        nodeName = '_set_' + re.match('([a-zA-Z0-9_]+)\\.setter', nameCheck).group(1)
                        pushPropertyAccessor(nodeName)
                    else:
                        decorate = True
            if sum([isClassMethod, isStaticMethod, isProperty]) > 1:
                raise utils.Error(lineNr=self.lineNr, message="\n\tstaticmethod, classmethod and property decorators can't be mixed\n")
            jsCall: bool = self.allowJavaScriptCall and nodeName != '__init__'
            decoratorsUsed: int = 0
            if decorate:
                if isMethod:
                    if jsCall:
                        raise utils.Error(lineNr=self.lineNr, message='\n\tdecorators are not supported with jscall\n')
                        self.emit('{}: ', self.filterId(nodeName))
                    else:
                        self.emit('get {} () {{return {} (this, '.format(self.filterId(nodeName), getter))
                elif isGlobal:
                    if isinstance(node.parentNode, ast.Module) and (nodeName not in self.allOwnNames):
                        self.emit('export ')
                    self.emit('var {} = '.format(self.filterId(nodeName)))
                else:
                    self.emit('var {} = '.format(self.filterId(nodeName)))
                if self.allowOperatorOverloading:
                    self.emit('__call__ (')
                for decorator in node.decorator_list:
                    if not (isinstance(decorator, ast.Name) and decorator.id in ('classmethod', 'staticmethod')):
                        if decoratorsUsed > 0:
                            self.emit(' (')
                        self.visit(decorator)
                        decoratorsUsed += 1
                if self.allowOperatorOverloading:
                    self.emit(', null, ')
                else:
                    self.emit(' (')
            elif isMethod:
                if jsCall:
                    self.emit('{}: function'.format(self.filterId(nodeName)))
                elif isStaticMethod:
                    self.emit('get {} () {{return {}function'.format(self.filterId(nodeName), 'async ' if anAsync else '')
                else:
                    self.emit('get {} () {{return {}function'.format(self.filterId(nodeName), getter, 'async ' if anAsync else ''))
            elif isGlobal:
                if isinstance(node.parentNode, ast.Module) and (nodeName not in self.allOwnNames):
                    self.emit('export ')
                self.emit('var {} = {}function'.format(self.filterId(nodeName), 'async ' if anAsync else ''))
            else:
                self.emit('var {} = {}function'.format(self.filterId(nodeName), 'async ' if anAsync else ''))
            yieldStarIndex: int = self.fragmentIndex
            self.emit(' ')
            skipFirstArg: bool = jsCall and (not (not isMethod or isStaticMethod or isProperty))
            if skipFirstArg:
                firstArg: str = node.args.args[0].arg
                node.args.args = node.args.args[1:]
            self.visit(node.args)
            if skipFirstArg:
                if isClassMethod:
                    self.emit("var {} = '__class__' in this ? this.__class__ : this;\n".format(firstArg))
                else:
                    self.emit('var {} = this;\n'.format(firstArg))
            emitScopedBody()
            self.emit('}}')
            if self.allowDocAttribs:
                docString: Optional[str] = ast.get_docstring(node)
                if docString:
                    self.emit(" .__setdoc__ ('{}')".format(docString.replace('\n', '\\n ')))
            if isDataClass:
                self.fragmentIndex += nrOfFragmentsToJump
                self.indentLevel = originalIndentLevel
            if isMethod:
                if not jsCall:
                    if isStaticMethod:
                        self.emit(';}}')
                    else:
                        if self.allowMemoizeCalls:
                            self.emit(", '{}'", nodeName)
                        self.emit(');}}')
                if nodeName == '__iter__':
                    self.emit(',\n[Symbol.iterator] () {{return this.__iter__ ()}}')
                if nodeName == '__next__':
                    self.emit(',\nnext: __jsUsePyNext__')
            if isGlobal:
                self.allOwnNames.add(nodeName)

    def visit_Compare(self, node: ast.Compare) -> None:
        if len(node.comparators) > 1:
            self.emit('(')
        left = node.left
        for index, (op, right) in enumerate(zip(node.ops, node.comparators)):
            if index:
                self.emit(' && ')
            if isinstance(op, (ast.In, ast.NotIn)) or (self.allowOperatorOverloading and isinstance(op, (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE))):
                op_map = {
                    ast.In: '__in__',
                    ast.NotIn: '!__in__',
                    ast.Eq: '__eq__',
                    ast.NotEq: '__ne__',
                    ast.Lt: '__lt__',
                    ast.LtE: '__le__',
                    ast.Gt: '__gt__',
                    ast.GtE: '__ge__'
                }
                self.emit('{} ('.format(self.filterId(op_map.get(type(op), 'Never here'))))
                self.visitSubExpr(node, left)
                self.emit(', ')
                self.visitSubExpr(node, right)
                self.emit(')')
            else:
                self.visitSubExpr(node, left)
                self.emit(' {0} '.format(self.operators[type(op)][0]))
                self.visitSubExpr(node, right)
            left = right
        if len(node.comparators) > 1:
            self.emit(')')

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            self.emit('{}', repr(node.value))
        elif isinstance(node.value, bytes):
            self.emit("bytes ('{}')".format(node.value.decode('ASCII')))
        elif isinstance(node.value, complex):
            self.emit('complex (0, {})'.format(node.value.imag))
        elif isinstance(node.value, (float, int)):
            self.emit('{}'.format(node.value))
        else:
            self.emit(self.nameConsts.get(node.value, ''))

    def visit_Continue(self, node: ast.Continue) -> None:
        if not self.skippedTemp('continue'):
            self.emit('{} = true;\n', self.getTemp('continue'))
        self.emit('continue')

    def visit_Delete(self, node: ast.Delete) -> None:
        for expr in node.targets:
            if not isinstance(expr, ast.Name):
                self.emit('delete ')
                self.visit(expr)
                self.emit(';\n')

    def visit_Dict(self, node: ast.Dict) -> None:
        if not self.allowJavaScriptKeys:
            for key in node.keys:
                if not isinstance(key, ast.Constant):
                    self.emit('dict ([')
                    for index, (key_elem, value) in enumerate(zip(node.keys, node.values)):
                        self.emitComma(index)
                        self.emit('[')
                        self.visit(key_elem)
                        self.emit(', ')
                        self.visit(value)
                        self.emit(']')
                    self.emit('])')
                    return
        if self.allowJavaScriptIter:
            self.emit('{{')
        else:
            self.emit('dict ({{')
        for index, (key, value) in enumerate(zip(node.keys, node.values)):
            self.emitComma(index)
            self.idFiltering = False
            self.visit(key)
            self.idFiltering = True
            self.emit(': ')
            self.visit(value)
        if self.allowJavaScriptIter:
            self.emit('}}')
        else:
            self.emit('}})')

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self.visit_ListComp(node, isDict=True)

    def visit_Expr(self, node: ast.Expr) -> None:
        self.visit(node.value)

    def visit_For(self, node: ast.For) -> None:
        self.adaptLineNrString(node)
        if node.orelse and (not self.allowJavaScriptIter):
            self.emit('var {} = false;\n', self.nextTemp('break'))
        else:
            self.skipTemp('break')
        optimize: bool = (
            isinstance(node.target, ast.Name) and
            self.isCall(node.iter, 'range') and
            (not isinstance(node.iter.args[0], ast.Starred)) and
            (len(node.iter.args) < 3 or (
                isinstance(node.iter.args[2], ast.Constant) and isinstance(node.iter.args[2].value, int)
            ) or (
                isinstance(node.iter.args[2], ast.UnaryOp) and 
                isinstance(node.iter.args[2].operand, ast.Constant) and 
                isinstance(node.iter.args[2].operand.value, int)
            ))
        )
        if self.allowJavaScriptIter:
            self.emit('for (var ')
            self.visit(node.target)
            self.emit(' in ')
            self.visit(node.iter)
            self.emit(') {{\n')
            self.indent()
        elif optimize:
            step: int = 1 if len(node.iter.args) <= 2 else (
                node.iter.args[2].value if isinstance(node.iter.args[2], ast.Constant) else (
                    node.iter.args[2].operand.value if isinstance(node.iter.args[2].op, ast.UAdd) else 
                    -node.iter.args[2].operand.value
                )
            )
            self.emit('for (var ')
            self.visit(node.target)
            self.emit(' = ')
            self.visit(node.iter.args[0] if len(node.iter.args) > 1 else ast.Constant(value=0))
            self.emit('; ')
            self.visit(node.target)
            self.emit(' < ' if step > 0 else ' > ')
            self.visit(node.iter.args[1] if len(node.iter.args) > 1 else node.iter.args[0])
            self.emit('; ')
            self.visit(node.target)
            if step == 1:
                self.emit('++')
            elif step == -1:
                self.emit('--')
            elif step >= 0:
                self.emit(' += {}'.format(step))
            else:
                self.emit(' -= {}'.format(-step))
            self.emit(') {{\n')
            self.indent()
        elif not self.allowOperatorOverloading:
            self.emit('for (var ')
            self.stripTuples = True
            self.visit(node.target)
            self.stripTuples = False
            self.emit(' of ')
            if self.allowConversionToIterable:
                self.emit('__i__ (')
            self.visit(node.iter)
            if self.allowConversionToIterable:
                self.emit(')')
            self.emit(') {{\n')
            self.indent()
        else:
            self.emit('var {} = '.format(self.nextTemp('iterable')))
            self.visit(node.iter)
            self.emit(';\n')
            if self.allowConversionToIterable:
                self.emit('{0} = __i__ ({0});\n'.format(self.getTemp('iterable')))
            self.emit('for (var {0} = 0; {0} < len ({1}); {0}++) {{\n'.format(
                self.nextTemp('index'), 
                self.getTemp('iterable')
            ))
            self.indent()
            self.visit(ast.Assign(
                targets=[node.target], 
                value=ast.Subscript(
                    value=ast.Name(id=self.getTemp('iterable'), ctx=ast.Load()), 
                    slice=ast.Name(id=self.getTemp('index'), ctx=ast.Load()), 
                    ctx=ast.Load()
                )
            ))
            self.emit(';\n')
        self.emitBody(node.body)
        self.dedent()
        self.emit('}}\n')
        if not (self.allowJavaScriptIter or optimize):
            if self.allowOperatorOverloading:
                self.prevTemp('index')
                self.prevTemp('iterable')
        if node.orelse:
            self.adaptLineNrString(node.orelse, 1)
            self.emit('if (!{}) {{\n'.format(self.getTemp('break')))
            self.prevTemp('break')
            self.indent()
            self.emitBody(node.orelse)
            self.dedent()
            self.emit('}}\n')
        else:
            self.prevTemp('break')

    def visit_FormattedValue(self, node: ast.FormattedValue) -> None:
        self.visit(node.value)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node, anAsync=True)

    def visit_FunctionDef(self, node: ast.FunctionDef, anAsync: bool = False) -> None:
        # This method is already implemented above.
        pass

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self.visit_ListComp(node, isGenExp=True)

    def visit_Global(self, node: ast.Global) -> None:
        self.getScope(ast.FunctionDef, ast.AsyncFunctionDef).nonlocals.update(node.names)

    def visit_If(self, node: ast.If) -> None:
        self.adaptLineNrString(node)
        self.emit('if (')
        self.emitBeginTruthy()
        global inIf
        inIf = True
        self.visit(node.test)
        inIf = False
        self.emitEndTruthy()
        self.emit(') {{\n')
        self.indent()
        self.emitBody(node.body)
        self.dedent()
        self.emit('}}\n')
        if node.orelse:
            if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                self.emit('else ')
                self.visit(node.orelse[0])
            else:
                self.adaptLineNrString(node.orelse, 1)
                self.emit('else {{\n')
                self.indent()
                self.emitBody(node.orelse)
                self.dedent()
                self.emit('}}\n')

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.emit('(')
        self.emitBeginTruthy()
        self.visit(node.test)
        self.emitEndTruthy()
        self.emit(' ? ')
        self.visit(node.body)
        self.emit(' : ')
        self.visit(node.orelse)
        self.emit(')')

    def visit_Import(self, node: ast.Import) -> None:
        self.importHoistMemos.append(utils.Any(node=node, lineNr=self.lineNr))

    def revisit_Import(self, importHoistMemo: Any) -> None:
        self.lineNr = importHoistMemo.lineNr
        node: ast.Import = importHoistMemo.node
        self.adaptLineNrString(node)
        names = [alias for alias in node.names if not alias.name.startswith(self.stubsName)]
        if not names:
            return
        for index, alias in enumerate(names):
            try:
                module: Module = self.useModule(alias.name)
            except Exception as exception:
                utils.enhanceException(exception, lineNr=self.lineNr, message="\n\tCan't import module '{}'".format(alias.name))
            if alias.asname and (alias.asname not in self.allOwnNames and alias.asname not in self.allImportedNames):
                self.allImportedNames.add(alias.asname)
                self.emit("import * as {} from '{}';\n".format(
                    self.filterId(alias.asname), 
                    module.importRelPath
                ))
            else:
                self.emit("import * as __module_{}__ from '{}';\n".format(
                    self.filterId(module.name).replace('.', '_'), 
                    module.importRelPath
                ))
                aliasSplit = alias.name.split('.', 1)
                head: str = aliasSplit[0]
                tail: str = aliasSplit[1] if len(aliasSplit) > 1 else ''
                self.importHeads.add(head)
                self.emit("__nest__ ({}, '{}', __module_{}__);\n".format(
                    self.filterId(head), 
                    self.filterId(tail), 
                    self.filterId(module.name.replace('.', '_'))
                ))
            if index < len(names) - 1:
                self.emit(';\n')

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.importHoistMemos.append(utils.Any(node=node, lineNr=self.lineNr))

    def revisit_ImportFrom(self, importHoistMemo: Any) -> None:
        self.lineNr = importHoistMemo.lineNr
        node: ast.ImportFrom = importHoistMemo.node
        self.adaptLineNrString(node)
        if node.module.startswith(self.stubsName):
            return
        try:
            namePairs: List[Any] = []
            facilityImported: bool = False
            for index, alias in enumerate(node.names):
                if alias.name == '*':
                    if len(node.names) > 1:
                        raise utils.Error(lineNr=self.lineNr, message="\n\tCan't import module '{}'".format(alias.name))
                    module: Module = self.useModule(node.module)
                    for aName in module.exportedNames:
                        namePairs.append(utils.Any(name=aName, asName=None))
                else:
                    try:
                        module: Module = self.useModule('{}.{}'.format(node.module, alias.name))
                        self.emit("import * as {} from '{}';\n".format(
                            self.filterId(alias.asname) if alias.asname else self.filterId(alias.name), 
                            module.importRelPath
                        ))
                        self.allImportedNames.add(alias.asname or alias.name)
                    except:
                        module: Module = self.useModule(node.module)
                        namePairs.append(utils.Any(name=alias.name, asName=alias.asname))
                        facilityImported = True
            if facilityImported:
                module = self.useModule(node.module)
                namePairs.append(utils.Any(name=alias.name, asName=alias.asname))
            if namePairs:
                try:
                    self.emit('import {{')
                    for index, namePair in enumerate(sorted(namePairs, key=lambda namePair: namePair.asName if namePair.asName else namePair.name)):
                        if not (namePair.asName if namePair.asName else namePair.name) in self.allOwnNames and not (namePair.asName if namePair.asName else namePair.name) in self.allImportedNames:
                            self.emitComma(index)
                            self.emit(self.filterId(namePair.name))
                            if namePair.asName:
                                self.emit(' as {}'.format(self.filterId(namePair.asName)))
                                self.allImportedNames.add(namePair.asName)
                            else:
                                self.allImportedNames.add(namePair.name)
                    self.emit("}} from '{}';\n".format(module.importRelPath))
                except:
                    print('Unexpected import error:', traceback.format_exc())
        except Exception as exception:
            utils.enhanceException(exception, lineNr=self.lineNr, message="\n\tCan't import from module '{}'".format(node.module))

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        self.emit(repr(''.join([value.value if isinstance(value, ast.Constant) else '{{}}' for value in node.values])))
        self.emit('.format (')
        index: int = 0
        for value in node.values:
            if isinstance(value, ast.FormattedValue):
                self.emitComma(index)
                self.visit(value)
                index += 1
        self.emit(')')

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.emit('(function __lambda__ ')
        self.visit(node.args)
        self.emit('return ')
        self.visit(node.body)
        self.dedent()
        self.emit(';\n}})')

    def visit_List(self, node: ast.List) -> None:
        self.emit('[')
        for index, elt in enumerate(node.elts):
            self.emitComma(index)
            self.visit(elt)
        self.emit(']')

    def visit_ListComp(self, node: ast.ListComp, isSet: bool = False, isDict: bool = False, isGenExp: bool = False) -> None:
        elts: List[Any] = []
        bodies: List[List[ast.stmt]] = [[]]

        def nestLoops(generators: List[ast.comprehension]) -> None:
            for comprehension in generators:
                target: ast.AST = comprehension.target
                iter: ast.AST = comprehension.iter
                bodies.append([])
                bodies[-2].append(ast.For(target, iter, bodies[-1], []))
                for expr in comprehension.ifs:
                    test: ast.AST = expr
                    bodies.append([])
                    bodies[-2].append(ast.If(test=test, body=bodies[-1], orelse=[]))
            bodies[-1].append(ast.Call(
                func=ast.Attribute(value=ast.Name(id=self.getTemp('accu'), ctx=ast.Load()), attr='append', ctx=ast.Load()), 
                args=[ast.List(elts=[node.key, node.value], ctx=ast.Load())] if isDict else [node.elt], 
                keywords=[]
            ))
            self.visit(bodies[0][0])

        self.emit('(function () {{\n')
        self.inscope(ast.FunctionDef())
        self.indent()
        self.emit('var {} = [];\n'.format(self.nextTemp('accu')))
        nestLoops(node.generators[:])
        self.emit('return {}{}{};\n'.format(
            'set (' if isSet else 'dict (' if isDict else '{} ('.format(self.filterId('iter')) if isGenExp else '', 
            self.getTemp('accu'), 
            ')' if isSet or isDict or isGenExp else ''
        ))
        self.prevTemp('accu')
        self.dedent()
        self.descope()
        self.emit('}}) ()')

    def visit_Module(self, node: ast.Module) -> None:
        self.adaptLineNrString()
        self.emit("// {}'ed from Python, {}\n".format(
            self.module.program.envir.transpiler_name.capitalize(), 
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ))
        self.adaptLineNrString(node)
        self.inscope(node)
        self.importHoistFragmentIndex: int = self.fragmentIndex
        self.emit("var __name__ = '{}';\n".format(self.module.__name__))
        self.allOwnNames.add('__name__')
        for statement in node.body:
            if self.isCommentString(statement):
                pass
            else:
                self.visit(statement)
                self.emit(';\n')
        if self.allowDocAttribs:
            docString: Optional[str] = ast.get_docstring(node)
            if docString:
                self.emit("export var __doc__ = '{}';\n".format(docString.replace('\n', '\\n')))
        "\n        Make the globals () function work as well as possible in conjunction with JavaScript 6 modules rather than closures\n\n        JavaScript 6 module-level variables normally cannot be accessed directly by their name as a string\n        They aren't attributes of any global object, certainly not in strict mode, which is the default for modules\n        By making getters and setters by the same name members of __all__, we can approach globals () as a dictionary\n\n        Limitations:\n        - We can access (read/write) but not create module-level globals this way\n        - If there are a lot of globals (bad style) this mechanism becomes expensive, so it must be under a pragma\n\n        It's possible that future versions of JavaScript facilitate better solutions to this minor problem\n        "
        if self.allowGlobals:
            self.emit('var __all__ = dict ({{' + ', '.join([
                f'get {name} () {{{{return {name};}}}}, set {name} (value) {{{{{name} = value;}}}}' for name in sorted(self.allOwnNames)
            ]) + '}});\n')
        self.fragmentIndex = self.importHoistFragmentIndex
        for importHoistMemo in reversed(self.importHoistMemos):
            if isinstance(importHoistMemo.node, ast.Import):
                self.revisit_Import(importHoistMemo)
            else:
                self.revisit_ImportFrom(importHoistMemo)
        if utils.commandArgs.xreex or self.module.sourcePrename == '__init__':
            if self.allImportedNames:
                self.emit('export {{{}}};\n'.format(', '.join([self.filterId(importedName) for importedName in self.allImportedNames])))
        self.fragmentIndex = self.importHoistFragmentIndex
        if self.module.name != self.module.program.runtimeModuleName:
            runtimeModule: Module = self.module.program.moduleDict[self.module.program.runtimeModuleName]
            importedNamesFromRuntime: str = ', '.join(sorted([
                exportedNameFromRuntime for exportedNameFromRuntime in runtimeModule.exportedNames 
                if exportedNameFromRuntime not in self.allOwnNames and exportedNameFromRuntime not in self.allImportedNames
            ]))
            self.emit("import {{{}}} from '{}';\n".format(importedNamesFromRuntime, runtimeModule.importRelPath))
        self.fragmentIndex = self.importHoistFragmentIndex
        for importHead in sorted(self.importHeads):
            self.emit('var {} = {{}};\n'.format(self.filterId(importHead)))
        self.descope()

    def visit_Name(self, node: ast.Name) -> None:
        if node.id == '__file__':
            self.visit(ast.Constant(value=self.module.sourcePath))
            return
        elif node.id == '__filename__':
            path: Tuple[str, str] = os.path.split(self.module.sourcePath)
            fileName: str = path[1]
            if fileName.startswith('__init__'):
                subDir: Tuple[str, str] = os.path.split(path[0])
                fileName = os.path.join(subDir[1], fileName)
            self.visit(ast.Constant(value=fileName))
            return
        elif node.id == '__line__':
            self.visit(ast.Constant(value=self.lineNr))
            return
        elif isinstance(node.ctx, ast.Store):
            if isinstance(self.getScope().node, ast.Module):
                self.allOwnNames.add(node.id)
        self.emit(self.filterId(node.id))

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.getScope(ast.FunctionDef, ast.AsyncFunctionDef).nonlocals.update(node.names)

    def visit_Pass(self, node: ast.Pass) -> None:
        self.adaptLineNrString(node)
        self.emit('// pass')

    def visit_Raise(self, node: ast.Raise) -> None:
        self.adaptLineNrString(node)
        if node.exc:
            self.emit('var {} = '.format(self.nextTemp('except')))
            self.visit(node.exc)
            self.emit(';\n')
        else:
            pass
        self.emit('{}.__cause__ = '.format(self.getTemp('except')))
        if node.cause:
            self.visit(node.cause)
        else:
            self.emit('null')
        self.emit(';\n')
        self.emit('throw {}', self.getTemp('except'))
        if node.exc:
            self.prevTemp('except')

    def visit_Return(self, node: ast.Return) -> None:
        self.adaptLineNrString(node)
        self.emit('return ')
        if node.value:
            self.visit(node.value)

    def visit_Set(self, node: ast.Set) -> None:
        self.emit('new set ([')
        for index, elt in enumerate(node.elts):
            self.emitComma(index)
            self.visit(elt)
        self.emit('])')

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self.visit_ListComp(node, isSet=True)

    def visit_Slice(self, node: ast.Slice) -> None:
        self.emit('tuple ([')
        if node.lower is None:
            self.emit('0')
        else:
            self.visit(node.lower)
        self.emit(', ')
        if node.upper is None:
            self.emit('null')
        else:
            self.visit(node.upper)
        self.emit(', ')
        if node.step is None:
            self.emit('1')
        else:
            self.visit(node.step)
        self.emit('])')

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.slice, ast.Index):
            if isinstance(node.slice.value, ast.Tuple):
                self.visit(node.value)
                self.emit('.__getitem__ (')
                self.stripTuple = True
                self.visit(node.slice.value)
                self.emit(')')
            elif self.allowOperatorOverloading:
                self.emit('__getitem__ (')
                self.visit(node.value)
                self.emit(', ')
                self.visit(node.slice.value)
                self.emit(')')
            else:
                try:
                    isRhsIndex: bool = not self.expectingNonOverloadedLhsIndex
                    self.expectingNonOverloadedLhsIndex = False
                    if isRhsIndex and self.allowKeyCheck:
                        self.emit('__k__ (')
                        self.visit(node.value)
                        self.emit(', ')
                        self.visit(node.slice.value)
                        self.emit(')')
                    else:
                        self.visit(node.value)
                        self.emit(' [')
                        self.visit(node.slice.value)
                        self.emit(']')
                except:
                    print(traceback.format_exc())
        elif isinstance(node.slice, ast.Slice):
            if self.allowOperatorOverloading:
                self.emit('__getslice__ (')
                self.visit(node.value)
                self.emit(', ')
            else:
                self.visit(node.value)
                self.emit('.__getslice__ (')
            if node.slice.lower is None:
                self.emit('0')
            else:
                self.visit(node.slice.lower)
            self.emit(', ')
            if node.slice.upper is None:
                self.emit('null')
            else:
                self.visit(node.slice.upper)
            self.emit(', ')
            if node.slice.step:
                self.visit(node.slice.step)
            else:
                self.emit('1')
            self.emit(')')
        elif isinstance(node.slice, ast.ExtSlice):
            self.visit(node.value)
            self.emit('.__getitem__ (')
            self.emit('[')
            for index, dim in enumerate(node.slice.dims):
                self.emitComma(index)
                self.visit(dim)
            self.emit(']')
            self.emit(')')

    def visit_Try(self, node: ast.Try) -> None:
        self.adaptLineNrString(node)
        self.emit('try {{\n')
        self.indent()
        self.emitBody(node.body)
        if node.orelse:
            self.emit('try {{\n')
            self.indent()
            self.emitBody(node.orelse)
            self.dedent()
            self.emit('}}\n')
            self.emit('catch ({}) {{\n'.format(self.nextTemp('except')))
            self.indent()
            self.emit('}}\n')
            self.prevTemp('except')
        self.dedent()
        self.emit('}}\n')
        if node.handlers:
            self.emit('catch ({}) {{\n'.format(self.nextTemp('except')))
            self.indent()
            for index, exceptionHandler in enumerate(node.handlers):
                if index:
                    self.emit('else ')
                if exceptionHandler.type:
                    self.emit('if (isinstance ({}, '.format(self.getTemp('except')))
                    self.visit(exceptionHandler.type)
                    self.emit(')) {{\n')
                    self.indent()
                    if exceptionHandler.name:
                        self.emit('var {} = {};\n'.format(exceptionHandler.name, self.getTemp('except')))
                    self.emitBody(exceptionHandler.body)
                    self.dedent()
                    self.emit('}}\n')
                else:
                    self.emitBody(exceptionHandler.body)
                    break
            else:
                self.emit('else {{\n')
                self.indent()
                self.emit('throw {};\n'.format(self.getTemp('except')))
                self.dedent()
                self.emit('}}\n')
            self.dedent()
            self.prevTemp('except')
            self.emit('}}\n')
        if node.finalbody:
            self.emit('finally {{\n')
            self.indent()
            self.emitBody(node.finalbody)
            self.dedent()
            self.emit('}}\n')

    def visit_Tuple(self, node: ast.Tuple) -> None:
        keepTuple: bool = not (self.stripTuple or self.stripTuples)
        self.stripTuple = False
        if keepTuple:
            self.emit('tuple (')
        self.emit('[')
        for index, elt in enumerate(node.elts):
            self.emitComma(index)
            self.visit(elt)
        self.emit(']')
        if keepTuple:
            self.emit(')')

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if self.allowOperatorOverloading and isinstance(node.op, ast.USub):
            self.emit('{} ('.format(self.filterId('__neg__')))
            self.visit(node.operand)
            self.emit(')')
        else:
            self.emit(self.operators[type(node.op)][0])
            self.emitBeginTruthy()
            self.visitSubExpr(node, node.operand)
            self.emitEndTruthy()

    def visit_While(self, node: ast.While) -> None:
        self.adaptLineNrString(node)
        if node.orelse:
            self.emit('var {} = false;\n'.format(self.nextTemp('break')))
        else:
            self.skipTemp('break')
        self.emit('while (')
        self.emitBeginTruthy()
        self.visit(node.test)
        self.emitEndTruthy()
        self.emit(') {{\n')
        self.indent()
        self.emitBody(node.body)
        self.dedent()
        self.emit('}}\n')
        if node.orelse:
            self.adaptLineNrString(node.orelse, 1)
            self.emit('if (!{}) {{\n'.format(self.getTemp('break')))
            self.prevTemp('break')
            self.indent()
            self.emitBody(node.orelse)
            self.dedent()
            self.emit('}}\n')
        else:
            self.prevTemp('break')

    def visit_With(self, node: ast.With) -> None:
        from contextlib import contextmanager, ExitStack
        self.adaptLineNrString(node)

        @contextmanager
        def itemContext(item: ast.withitem):
            if not self.noskipCodeGeneration:
                yield
                return
            self.emit('var ')
            if item.optional_vars:
                self.visit(item.optional_vars)
                withId: str = item.optional_vars.id  # type: ignore
            else:
                withId: str = self.nextTemp('withid')
                self.emit(withId)
            self.emit(' = ')
            self.visit(item.context_expr)
            self.emit(';\n')
            self.emit('try {{\n')
            self.indent()
            self.emit('{}.__enter__ ();\n'.format(withId))
            yield
            self.emit('{}.__exit__ ();\n'.format(withId))
            self.dedent()
            self.emit('}}\n')
            self.emit('catch ({}) {{\n'.format(self.nextTemp('except')))
            self.indent()
            self.emit('if (! ({0}.__exit__ ({1}.name, {1}, {1}.stack))) {{\n'.format(withId, self.getTemp('except')))
            self.indent()
            self.emit('throw {};\n'.format(self.getTemp('except')))
            self.dedent()
            self.emit('}}\n')
            self.dedent()
            self.emit('}}\n')
            self.prevTemp('except')
            if withId == self.getTemp('withid'):
                self.prevTemp('withid')

        @contextmanager
        def pragmaContext(item: ast.withitem):
            expr: ast.AST = item.context_expr
            name: str = expr.args[0].s
            if name.startswith('no'):
                revName: str = name[2:]
            else:
                revName: str = 'no' + name
            self.visit(expr)
            yield
            self.visit(ast.Call(
                func=ast.Name(id='__pragma__', ctx=ast.Load()), 
                args=[ast.Constant(value=revName)] + expr.args[1:], 
                keywords=[]
            ))

        @contextmanager
        def skipContext(item: ast.withitem):
            self.noskipCodeGeneration = False
            yield
            self.noskipCodeGeneration = True

        with ExitStack() as stack:
            for item in node.items:
                expr: ast.AST = item.context_expr
                if self.isCall(expr, '__pragma__'):
                    if expr.args[0].s == 'skip':
                        stack.enter_context(skipContext(item))
                    else:
                        stack.enter_context(pragmaContext(item))
                else:
                    stack.enter_context(itemContext(item))
            self.emitBody(node.body)

    def visit_Yield(self, node: ast.Yield) -> None:
        self.getScope(ast.FunctionDef, ast.AsyncFunctionDef).containsYield = True
        self.emit('yield')
        if node.value is not None:
            self.emit(' ')
            self.visit(node.value)

    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
        self.getScope(ast.FunctionDef, ast.AsyncFunctionDef).containsYield = True
        self.emit('yield* ')
        self.visit(node.value)

    # Additional methods like visit_FunctionDef, etc., would continue here with appropriate type annotations.
