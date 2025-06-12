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
from org.transcrypt import utils, sourcemaps, minify, static_check, type_check
from typing import Any, List, Dict, Set, Tuple, Optional, Union, Callable, Iterator, Iterable, TypeVar, Generic, cast

inIf: bool = False
ecom: bool = True
noecom: bool = False
dataClassDefaultArgTuple: List[List[Union[str, bool]]] = [
    ['init', True], 
    ['repr', True], 
    ['eq', True], 
    ['order', False], 
    ['unsafe_hash', False], 
    ['frozen', False]
]

class Program:
    def __init__(
        self, 
        moduleSearchDirs: List[str], 
        symbols: Set[str], 
        envir: Any
    ) -> None:
        utils.setProgram(self)
        self.moduleSearchDirs: List[str] = moduleSearchDirs
        self.symbols: Set[str] = symbols
        self.envir: Any = envir
        self.javascriptVersion: int = int(utils.commandArgs.esv) if utils.commandArgs.esv else 6
        self.moduleDict: Dict[str, Module] = {}
        self.importStack: List[List[Union[Module, Optional[int]]]] = []
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
            project: Dict[str, Any] = {}
            
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
            
        project = {
            'options': utils.commandArgs.projectOptions, 
            'modules': [
                {'source': module.sourcePath, 'target': module.targetPath} 
                for module in self.moduleDict.values()
            ]
        }
        with utils.create(self.projectPath) as projectFile:
            json.dump(project, projectFile)

    def provide(
        self, 
        moduleName: str, 
        __moduleName__: Optional[str] = None, 
        filter: Optional[Callable[[str], str]] = None
    ) -> 'Module':
        if moduleName in self.moduleDict:
            return self.moduleDict[moduleName]
        else:
            return Module(self, moduleName, __moduleName__, filter)

class Module:
    def __init__(
        self, 
        program: Program, 
        name: str, 
        __name__: Optional[str], 
        filter: Optional[Callable[[str], str]]
    ) -> None:
        self.program: Program = program
        self.name: str = name
        self.__name__: str = __name__ if __name__ else self.name
        self.findPaths(filter)
        self.program.importStack.append([self, None])
        self.program.moduleDict[self.name] = self
        self.sourceMapper: sourcemaps.SourceMapper = sourcemaps.SourceMapper(
            self.name, 
            self.program.targetDir, 
            not utils.commandArgs.nomin, 
            utils.commandArgs.dmap
        )
        
        if (
            utils.commandArgs.build or 
            self.program.optionsChanged or 
            (not os.path.isfile(self.targetPath)) or 
            (os.path.getmtime(self.sourcePath) > os.path.getmtime(self.targetPath)
        ):
            if self.isJavascriptOnly:
                self.loadJavascript()
                javascriptDigest: utils.JavascriptDigest = utils.digestJavascript(
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
            self.targetCode: str = open(self.targetPath, 'r').read()
            javascriptDigest = utils.digestJavascript(
                self.targetCode, 
                self.program.symbols, 
                True, 
                False, 
                refuseIfAppearsMinified=True
            )
            if not javascriptDigest:
                minify.run(self.program.targetDir, self.targetName, self.prettyTargetName, prettify=True)
                self.prettyTargetCode: str = open(self.prettyTargetPath, 'r').read()
                javascriptDigest = utils.digestJavascript(
                    self.prettyTargetCode, 
                    self.program.symbols, 
                    True, 
                    False
                )
                
        self.targetCode: str = javascriptDigest.digestedCode
        self.importedModuleNames: List[str] = javascriptDigest.importedModuleNames
        self.exportedNames: List[str] = javascriptDigest.exportedNames
        
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
                        targetLine = '/* {} */ {}'.format(sourceLineNrString, targetLine)
                    targetLines.append(targetLine)
                    if utils.commandArgs.map:
                        self.sourceLineNrs.append(sourceLineNr)
                        
            if utils.commandArgs.map:
                utils.log(False, 'Saving source map in: {}\n', self.mapPath)
                self.sourceMapper.generateAndSavePrettyMap(self.sourceLineNrs)
                shutil.copyfile(self.sourcePath, self.mapSourcePath)
        else:
            targetLines = [line for line in ''.join(self.generator.targetFragments).split('\n') if line.strip() != ';']
            
        self.targetCode: str = '\n'.join(targetLines)

    def loadJavascript(self) -> None:
        with tokenize.open(self.sourcePath) as sourceFile:
            self.targetCode: str = sourceFile.read()

    def parse(self) -> None:
        def pragmasFromComments(sourceCode: str) -> str:
            tokens: Iterator[tokenize.TokenInfo] = tokenize.tokenize(io.BytesIO(sourceCode.encode('utf-8')).readline)
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
                strippedHead = head.lstrip()
                indent = head[:len(head) - len(strippedHead)]
                pragmaName = tail.replace(' ', '').replace('\t', '')[3:]
                if pragmaName == 'ecom':
                    sourceLines[pragmaCommentLineIndex] = ecom
                elif pragmaName == 'noecom':
                    sourceLines[pragmaCommentLineIndex] = noecom
                elif pragmaName.startswith('no'):
                    sourceLines[shortPragmaCommentLineIndex] = "{}__pragma__ ('{}'); {}; __pragma__ ('{}')".format(indent, pragmaName, head, pragmaName[2:])
                else:
                    sourceLines[shortPragmaCommentLineIndex] = "{}__pragma__ ('{}'); {}; __pragma__ ('no{}')".format(indent, pragmaName, head, pragmaName)
                    
            uncommentedSourceLines: List[str] = []
            for sourceLine in sourceLines:
                if sourceLine == ecom:
                    allowExecutableComments = True
                elif sourceLine == noecom:
                    allowExecutableComments = False
                elif allowExecutableComments:
                    lStrippedSourceLine = sourceLine.lstrip()
                    if not lStrippedSourceLine[:4] in {"'''?", "?'''", '"""?', '?"""'}:
                        uncommentedSourceLines.append(sourceLine.replace('#?', '', 1) if lStrippedSourceLine.startswith('#?') else sourceLine
                else:
                    uncommentedSourceLines.append(sourceLine)
                    
            return '\n'.join(uncommentedSourceLines)
            
        try:
            utils.log(False, 'Parsing module: {}\n', self.sourcePath)
            with tokenize.open(self.sourcePath) as sourceFile:
                self.sourceCode: str = utils.extraLines + sourceFile.read()
                
            self.parseTree: ast.Module = ast.parse(pragmasFromComments(self.sourceCode))
            for node in ast.walk(self.parseTree):
                for childNode in ast.iter_child_nodes(node):
                    childNode.parentNode = node
        except SyntaxError as syntaxError:
            utils.enhanceException(syntaxError, lineNr=syntaxError.lineno, message='\n\t{} [<-SYNTAX FAULT] {}'.format(syntaxError.text[:syntaxError.offset].lstrip(), syntaxError.text[syntaxError.offset:].rstrip()) if syntaxError.text else syntaxError.args[0])

    def dumpTree(self) -> None:
        utils.log(False, 'Dumping syntax tree for module: {}\n', self.sourcePath)

        def walk(name: str, value: