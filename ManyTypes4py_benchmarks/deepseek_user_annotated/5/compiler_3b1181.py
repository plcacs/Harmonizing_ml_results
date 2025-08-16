# ====== Legal notices
#
#
# Copyright 2014 - 2018 Jacques de Hooge, GEATEC engineering, www.geatec.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
from typing import Any, Dict, List, Set, Tuple, Optional, Union, Callable

from org.transcrypt import utils, sourcemaps, minify, static_check, type_check

inIf: bool = False
ecom: bool = True
noecom: bool = False

dataClassDefaultArgTuple: List[List[Union[str, bool]] = [
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
        moduleSearchDirs: List[str],   # All possible roots of the module path
        symbols: Set[str],            # Set of symbols from command line
        envir: Dict[str, Any]               # Data about run/compilation environment
    ) -> None:
        utils.setProgram(self)

        self.moduleSearchDirs: List[str] = moduleSearchDirs
        self.symbols: Set[str] = symbols
        self.envir: Dict[str, Any] = envir
        self.javascriptVersion: int = int(utils.commandArgs.esv) if utils.commandArgs.esv else 6

        self.moduleDict: Dict[str, 'Module'] = {}    # Administration of all modules
        self.importStack: List[List[Union['Module', Optional[int]]]] = []   # Pending imports

        # Set paths
        self.sourcePrepath: str = os.path.abspath(utils.commandArgs.source).replace('\\', '/')
        
        self.sourceDir: str = '/'.join(self.sourcePrepath.split('/')[: -1])
        self.mainModuleName: str = self.sourcePrepath.split('/')[-1]
        
        if utils.commandArgs.outdir:
            if os.path.isabs(utils.commandArgs.outdir):
                self.targetDir: str = utils.commandArgs.outdir.replace('\\', '/')
            else:
                self.targetDir: str = f'{self.sourceDir}/{utils.commandArgs.outdir}'.replace('\\', '/')
        else:
            self.targetDir: str = f'{self.sourceDir}/__target__'.replace('\\', '/')
        
        self.projectPath: str = f'{self.targetDir}/{self.mainModuleName}.project'

        # Load the most recent project metadata
        try:
            with open(self.projectPath, 'r') as projectFile:
                project: Dict[str, Any] = json.load(projectFile)
        except:
            project: Dict[str, Any] = {}

        # Reset everything in case of a build or a command args change
        self.optionsChanged: bool = utils.commandArgs.projectOptions != project.get('options')
        if utils.commandArgs.build or self.optionsChanged:
            shutil.rmtree(self.targetDir, ignore_errors=True)

        try:
            # Provide runtime module
            self.runtimeModuleName: str = 'org.transcrypt.__runtime__'
            self.searchedModulePaths: List[str] = []   # Report only failure of searching runtime
            self.provide(self.runtimeModuleName)

            # Provide main module and all other modules recursively
            self.searchedModulePaths = []   # Report only failure of searching for main
            self.provide(self.mainModuleName, '__main__')
        except Exception as exception:
            utils.enhanceException(
                exception,
                message=f'\n\t{exception}'
            )

        # Finally, save the run info
        project = {
            'options': utils.commandArgs.projectOptions,
            'modules': [{'source': module.sourcePath, 'target': module.targetPath} for module in self.moduleDict.values()],
        }
        with utils.create(self.projectPath) as projectFile:
            json.dump(project, projectFile)

    def provide(self, moduleName: str, __moduleName__: Optional[str] = None, filter: Optional[Callable[[str], str]] = None) -> 'Module':
        if moduleName in self.moduleDict:
            return self.moduleDict[moduleName]
        else:
            return Module(self, moduleName, __moduleName__, filter)

class Module:
    def __init__(self, program: Program, name: str, __name__: Optional[str], filter: Optional[Callable[[str], str]]) -> None:
        self.program: Program = program
        self.name: str = name
        self.__name__: str = __name__ if __name__ else self.name

        # Try to find module
        self.findPaths(filter)

        # Remember names of module being under compilation
        self.program.importStack.append([self, None])

        # Register that module is found
        self.program.moduleDict[self.name] = self

        # Create sourcemapper
        self.sourceMapper: sourcemaps.SourceMapper = sourcemaps.SourceMapper(
            self.name,
            self.program.targetDir,
            not utils.commandArgs.nomin,
            utils.commandArgs.dmap
        )

        # Generate, copy or load target code and symbols
        if (
            utils.commandArgs.build or self.program.optionsChanged
            or
            not os.path.isfile(self.targetPath) or os.path.getmtime(self.sourcePath) > os.path.getmtime(self.targetPath)
        ):
            if self.isJavascriptOnly:
                self.loadJavascript()
                javascriptDigest = utils.digestJavascript(self.targetCode, self.program.symbols, not utils.commandArgs.dnostrip, False)
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
                javascriptDigest = utils.digestJavascript(self.targetCode, self.program.symbols, False, self.generator.allowDebugMap)

            utils.log(True, 'Saving target code in: {}\n', self.targetPath)
            filePath = self.targetPath if utils.commandArgs.nomin else self.prettyTargetPath
            with utils.create(filePath) as aFile:
                aFile.write(self.targetCode)

            if not utils.commandArgs.nomin:
                utils.log(True, 'Saving minified target code in: {}\n', self.targetPath)
                minify.run(
                    self.program.targetDir,
                    self.prettyTargetName,
                    self.targetName,
                    mapFileName=self.shrinkMapName if utils.commandArgs.map else None,
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
            javascriptDigest = utils.digestJavascript(self.targetCode, self.program.symbols, True, False, refuseIfAppearsMinified=True)

            if not javascriptDigest:
                minify.run(
                    self.program.targetDir,
                    self.targetName,
                    self.prettyTargetName,
                    prettify=True,
                )
                self.prettyTargetCode = open(self.prettyTargetPath, 'r').read()
                javascriptDigest = utils.digestJavascript(self.prettyTargetCode, self.program.symbols, True, False)

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
                self.isJavascriptOnly: bool = os.path.isfile(self.javascriptSourcePath) and not os.path.isfile(self.pythonSourcePath)
                self.sourcePath: str = self.javascriptSourcePath if self.isJavascriptOnly else self.pythonSourcePath
                break

            self.program.searchedModulePaths.extend([self.pythonSourcePath, self.javascriptSourcePath])
        else:
            raise utils.Error(
                message='\n\tImport error, can\'t find any of:\n\t\t{}\n'.format('\n\t\t'.join(self.program.searchedModulePaths))
            )

    def generateJavascriptAndPrettyMap(self) -> None:
        utils.log(False, 'Generating code for module: {}\n', self.targetPath)
        self.generator = Generator(self)

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

        self.targetCode = '\n'.join(targetLines)

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
                    strippedComment = tokenString[1:].lstrip()
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
                    sourceLines[shortPragmaCommentLineIndex] = '{}__pragma__ (\'{}\'); {}; __pragma__ (\'{}\')'.format(indent, pragmaName, head, pragmaName[2:])
                else:
                    sourceLines[shortPragmaCommentLineIndex] = '{}__pragma__ (\'{}\'); {}; __pragma__ (\'no{}\')'.format(indent, pragmaName, head, pragmaName)
                    
            uncommentedSourceLines: List[str] = []
            allowExecutableComments: bool = utils.commandArgs.ecom
            for sourceLine in sourceLines:
                if sourceLine == ecom:
                    allowExecutableComments = True
                elif sourceLine == noecom:
                    allowExecutableComments = False
                elif allowExecutableComments:
                    lStrippedSourceLine = sourceLine.lstrip()
                    if not lStrippedSourceLine[:4] in {"'''?", "?'''", '"""?', '?"""'}:
                        uncommentedSourceLines.append(sourceLine.replace('#?', '', 1) if lStrippedSourceLine.startswith('#?') else sourceLine)
                else:
                    uncommentedSourceLines.append(sourceLine)
                    
            return '\n'.join(uncommentedSourceLines)

        try:
            utils.log(False, 'Parsing module: {}\n', self.sourcePath)

            with tokenize.open(self