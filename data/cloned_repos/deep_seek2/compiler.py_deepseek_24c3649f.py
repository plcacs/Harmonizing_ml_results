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
from typing import Any, List, Dict, Set, Tuple, Optional, Union

from org.transcrypt import utils, sourcemaps, minify, static_check, type_check

inIf: bool = False
ecom: bool = True
noecom: bool = False

dataClassDefaultArgTuple: List[Tuple[str, bool]] = [
    ['init', True], ['repr', True], ['eq', True], ['order', False], ['unsafe_hash', False], ['frozen', False]
]

class Program:
    def __init__(
        self,
        moduleSearchDirs: List[str],  # All possible roots of the module path, the latter uniquely denoted by the dotted module name
        symbols: Set[str],            # Set of symbols either passed on the command line, deduced from command line switches etc.
        envir: Dict[str, Any]         # Data about run / compilation environment
    ) -> None:
        utils.setProgram(self)

        self.moduleSearchDirs: List[str] = moduleSearchDirs
        self.symbols: Set[str] = symbols
        self.envir: Dict[str, Any] = envir
        self.javascriptVersion: int = int(utils.commandArgs.esv) if utils.commandArgs.esv else 6

        self.moduleDict: Dict[str, Module] = {}    # Administration of all modules that play a role in this program
        self.importStack: List[List[Union[Module, None]]] = []   # Pending imports, enables showing load sequence in case a module cannot be loaded

        # Set paths
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
            # Provide runtime module since it's always needed but never imported explicitly
            self.runtimeModuleName: str = 'org.transcrypt.__runtime__'
            self.searchedModulePaths: List[str] = []   # Report only failure of searching runtime, so clear any history
            self.provide(self.runtimeModuleName)

            # Provide main module and, with that, all other modules recursively
            self.searchedModulePaths: List[str] = []   # Report only failure of searching for main, so clear any history
            self.provide(self.mainModuleName, '__main__')
        except Exception as exception:
            utils.enhanceException(    # If it was an Error, don't change it, otherwise make it one (??? Just to be sure?)
                exception,
                message=f'\n\t{exception}'
            )

        # Finally, save the run info
        project: Dict[str, Any] = {
            'options': utils.commandArgs.projectOptions,
            'modules': [{'source': module.sourcePath, 'target': module.targetPath} for module in self.moduleDict.values()],
        }
        with utils.create(self.projectPath) as projectFile:
            json.dump(project, projectFile)

    def provide(self, moduleName: str, __moduleName__: Optional[str] = None, filter: Optional[Any] = None) -> 'Module':
        # moduleName may contain dots if it's imported, but it'll have the same name in every import

        if moduleName in self.moduleDict:  # Find out if module is already provided
            return self.moduleDict[moduleName]
        else:                              # If not, provide by loading or compiling
            # This may fail legally if filteredModuleName ends on a name of something in a module, rather than of the module itself
            return Module(self, moduleName, __moduleName__, filter)

class Module:
    def __init__(self, program: Program, name: str, __name__: Optional[str], filter: Optional[Any]) -> None:
        self.program: Program = program
        self.name: str = name
        self.__name__: str = __name__ if __name__ else self.name

        # Try to find module, exception if fails
        self.findPaths(filter)

        # Remember names of module being under compilation and line nrs of current import
        # Used for error reports
        # Note that JavaScript-only modules will leave lineNr None if they import something
        # This is since there's no explicit import location in such modules
        # Only add a module to the importStack if it's at least found by findPaths, otherwise it has no sourcePath to report
        self.program.importStack.append([self, None])

        # Register that module is found
        self.program.moduleDict[self.name] = self

        # Create sourcemapper, if only for cleaning dir after previous run
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
            # If it's a build rather than a make, or the target doesn't exist or the target is obsolete

            if self.isJavascriptOnly:
                # Digest source JavaScript and copy to target location
                self.loadJavascript()

                # JavaScript-only, so annotations are pointless, so it's ok to strip
                javascriptDigest: utils.JavascriptDigest = utils.digestJavascript(self.targetCode, self.program.symbols, not utils.commandArgs.dnostrip, False)

            else:
                # Perform static typecheck on source code
                if utils.commandArgs.dstat:
                    try:
                        type_check.run(self.sourcePath)
                    except Exception as exception:
                        utils.log(True, 'Validating: {} and dependencies\n\tInternal error in static typing validator\n', self.sourcePath)

                # Construct parse tree
                self.parse()
                if utils.commandArgs.dtree:
                    self.dumpTree()

                # Perform lightweight static check on parse tree
                if utils.commandArgs.dcheck:
                    try:
                        static_check.run(self.sourcePath, self.parseTree)
                    except Exception as exception:
                        utils.log(True, 'Checking: {}\n\tInternal error in lightweight consistency checker, remainder of module skipped\n', self.sourcePath)

                # Generate JavaScript code and sourcemap from parse tree
                self.generateJavascriptAndPrettyMap()

                # Generated code, may have annotations so don't strip comments, and there are no non-annotation comments to strip anyhow, neither are there any strip pragma's
                javascriptDigest: utils.JavascriptDigest = utils.digestJavascript(self.targetCode, self.program.symbols, False, self.generator.allowDebugMap)

            # Write target code
            utils.log(True, 'Saving target code in: {}\n', self.targetPath)
            filePath: str = self.targetPath if utils.commandArgs.nomin else self.prettyTargetPath
            with utils.create(filePath) as aFile:
                aFile.write(self.targetCode)

            # Minify target code
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

            # Append map reference to target file, which may be minified or not
            with open(self.targetPath, 'a') as targetFile:
                targetFile.write(self.mapRef)

        else:
            # If it's a make rather than a build and the target exists, load it, beautify it if needed and run through digestJavascript for obtaining symbols
            self.targetCode: str = open(self.targetPath, 'r').read()
            javascriptDigest: utils.JavascriptDigest = utils.digestJavascript(self.targetCode, self.program.symbols, True, False, refuseIfAppearsMinified=True)

            if not javascriptDigest:
                minify.run(
                    self.program.targetDir,
                    self.targetName,
                    self.prettyTargetName,
                    prettify=True,
                )
                self.prettyTargetCode: str = open(self.prettyTargetPath, 'r').read()
                javascriptDigest: utils.JavascriptDigest = utils.digestJavascript(self.prettyTargetCode, self.program.symbols, True, False)    # Prettified, so doesn't start with '/'!

        self.targetCode: str = javascriptDigest.digestedCode
        self.importedModuleNames: List[str] = javascriptDigest.importedModuleNames
        self.exportedNames: List[str] = javascriptDigest.exportedNames

        for importedModuleName in self.importedModuleNames:
            # Unfiltered hyphens allowed, since we may be in a JavaScript-only part of the module hierarchy
            # Also these imports cannot legally fail, since the digested JavaScript code already has unambiguous imports
            # If the JavaScript module was just generated from a Python module, it will already be in the module dictionary
            self.program.searchedModulePaths = []
            self.program.provide(importedModuleName)

        # Remove eventual intermediate files
        utils.tryRemove(self.prettyTargetPath)
        utils.tryRemove(self.shrinkMapPath)
        utils.tryRemove(self.prettyMapPath)

        # Module not under compilation anymore, so pop it
        self.program.importStack.pop()

    def findPaths(self, filter: Optional[Any]) -> None:
        # Filter to get hyphens and/or dots in name if a suitable alias is defined
        # The filter function, and with it the active aliases, are passed by the importing module
        rawRelSourceSlug: str = self.name.replace('.', '/')
        relSourceSlug: str = filter(rawRelSourceSlug) if filter and utils.commandArgs.alimod else rawRelSourceSlug

        for searchDir in self.program.moduleSearchDirs:
            # Find source slugs
            sourceSlug: str = f'{searchDir}/{relSourceSlug}'
            if os.path.isdir(sourceSlug):
                self.sourceDir: str = sourceSlug
                self.sourcePrename: str = '__init__'
            else:
                self.sourceDir, self.sourcePrename = sourceSlug.rsplit('/', 1)
            self.sourcePrepath: str = f'{self.sourceDir}/{self.sourcePrename}'
            self.pythonSourcePath: str = f'{self.sourcePrepath}.py'
            self.javascriptSourcePath: str = f'{self.sourcePrepath}.js'

            # Find target slugs
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

            # If module exists
            if os.path.isfile(self.pythonSourcePath) or os.path.isfile(self.javascriptSourcePath):
                # Check if it's a JavaScript-only module
                self.isJavascriptOnly: bool = os.path.isfile(self.javascriptSourcePath) and not os.path.isfile(self.pythonSourcePath)
                # Set more paths (tree, sourcemap, ...)
                # (To do)
                self.sourcePath: str = self.javascriptSourcePath if self.isJavascriptOnly else self.pythonSourcePath
                break

            # Remember all fruitless paths to give a decent error report if module isn't found
            # Note that these aren't all searched paths for a particular module,
            # since the difference between an module and a facility inside a module isn't always known a priori
            self.program.searchedModulePaths.extend([self.pythonSourcePath, self.javascriptSourcePath])
        else:
            # If even the target can't be loaded then there's a problem with this module, root or not
            # However, loading a module is allowed to fail (see self.revisit_ImportFrom)
            # In that case this error is swallowed, but searchedModulePath is retained,
            # because searching in the swallowing except branch may also fail and should mention ALL searched paths
            raise utils.Error(
                message='\n\tImport error, can\'t find any of:\n\t\t{}\n'.format('\n\t\t'.join(self.program.searchedModulePaths))
            )

    def generateJavascriptAndPrettyMap(self) -> None:
        utils.log(False, 'Generating code for module: {}\n', self.targetPath)

        # Generate target fragments
        self.generator: Generator = Generator(self)

        # Fabricate target lines from target fragments
        if utils.commandArgs.map or utils.commandArgs.anno:
            # In both cases the generator will have instrumented the target fragments by appending line numbers
            # N.B. __pragma__ ('noanno') will be too late to prevent instrumenting of the main module's first line
            # In that case if no source maps are required either, the appended line numbers simply won't be used

            # Split joined fragments into (instrumented) lines
            instrumentedTargetLines: List[str] = ''.join(self.generator.targetFragments).split('\n')

            # Only remember source line nrs if a map is to be generated (so not if only annotated JavaScript is needed)
            if utils.commandArgs.map:
                self.sourceLineNrs: List[int] = []

            # Split instrumentedTargetLines in (bare) targetLines and sourceLineNrs, skipping empty statements
            targetLines: List[str] = []
            for targetLine in instrumentedTargetLines:
                # The actual splitting
                sourceLineNrString: str = targetLine[-sourcemaps.lineNrLength:]                           # Take the appended line number, e.g. the string '000014'
                sourceLineNr: int = int('1' + sourceLineNrString) - sourcemaps.maxNrOfSourceLinesPerModule  # Turn it into an integer, e.g. 14
                targetLine: str = targetLine[:-sourcemaps.lineNrLength]                                   # Obtain non-instrumented line by removing the appended line number

                # Only append non-emptpy statements and their number info
                if targetLine.strip() != ';':                                                          # If the non-instrumented line isn't empty
                    if self.generator.allowDebugMap:                                                    # If annotations comments have to be prepended
                        targetLine: str = '/* {} */ {}'.format(sourceLineNrString, targetLine)              # Prepend them
                    targetLines.append(targetLine)                                                     # Add the target line, with or without prepended annotation comment

                    # Store line nrs for source map
                    if utils.commandArgs.map:
                        self.sourceLineNrs.append(sourceLineNr)                                        # Remember its line number to be able to generate a sourcemap

            # Generate per module sourcemap and copy sourcefile to target location
            if utils.commandArgs.map:
                utils.log(False, 'Saving source map in: {}\n', self.mapPath)
                self.sourceMapper.generateAndSavePrettyMap(self.sourceLineNrs)
                shutil.copyfile(self.sourcePath, self.mapSourcePath)
        else:
            # No maps or annotations needed, so this 'no stripping' shortcut for speed
            targetLines: List[str] = [line for line in ''.join(self.generator.targetFragments).split('\n') if line.strip() != ';']

        self.targetCode: str = '\n'.join(targetLines)

    def loadJavascript(self) -> None:
        with tokenize.open(self.sourcePath) as sourceFile:
            self.targetCode: str = sourceFile.read()

    def parse(self) -> None:
        def pragmasFromComments(sourceCode: str) -> str:
            # This function turns comment-like pragma's into regular ones, both for multi-line and single-line pragma's
            # It changes rather than regenerates the sourcecode, since tokenize/untokenize will mess up formatting
            # Single line pragma's are always comment-like and will be turned into multi-line function-like pragma's
            # Also in this function executable comments are converted to normal code 

            # Tokenize the source code, to be able to recognize comments easily
            tokens: tokenize.TokenInfo = tokenize.tokenize(io.BytesIO(sourceCode.encode('utf-8')).readline)

            # Store all line indices of comment-like pragma's, multi-line and single-line in separate lists
            pragmaCommentLineIndices: List[int] = []
            shortPragmaCommentLineIndices: List[int] = []
            ecomPragmaLineIndices: List[int] = []
            noecomPragmaLineIndices: List[int] = []
            pragmaIndex: int = -1000
            for tokenIndex, (tokenType, tokenString, startRowColumn, endRowColumn, logicalLine) in enumerate(tokens):
                if tokenType == tokenize.COMMENT:
                    strippedComment: str = tokenString[1:].lstrip()
                    if strippedComment.startswith('__pragma__'):
                        # Remember line index of multi-line pragma, like: # __pragma__ (...
                        pragmaCommentLineIndices.append(startRowColumn[0] - 1)
                    elif strippedComment.replace(' ', '').replace('\t', '').startswith('__:'):
                        # Remember line index of single-line pragma, like: <some code> # __: ...
                        shortPragmaCommentLineIndices.append(startRowColumn[0] - 1)
                if tokenType == tokenize.NAME and tokenString == '__pragma__':
                    pragmaIndex = tokenIndex

                if tokenIndex - pragmaIndex == 2:
                    pragmaKind: str = tokenString[1:-1]
                    if pragmaKind == 'ecom':
                        ecomPragmaLineIndices.append(startRowColumn[0] - 1)
                    elif pragmaKind == 'noecom':
                        noecomPragmaLineIndices.append(startRowColumn[0] - 1)

            # Convert original, non-tokenized sourcecode to a list of lines
            sourceLines: List[str] = sourceCode.split('\n')

            # Use line indices of multi-line function-like ecom / noecom pragma's to transform these lines into executable comment switches
            for ecomPragmaLineIndex in ecomPragmaLineIndices:
                sourceLines[ecomPragmaLineIndex] = ecom
            for noecomPragmaLineIndex in noecomPragmaLineIndices:
                sourceLines[noecomPragmaLineIndex] = noecom

            # Use line indices of multi-line comment-like pragma singles to transform these into function-like pragma singles (which often turn out te be part of a matching pair)
            allowExecutableComments: bool = utils.commandArgs.ecom
            for pragmaCommentLineIndex in pragmaCommentLineIndices:
                indentation, separator, tail = sourceLines[pragmaCommentLineIndex].partition('#')
                pragma, separator, comment = tail.partition('#')
                pragma = pragma.replace(' ', '').replace('\t', '')

                # Turn appropriate lines into executable comment switches
                if "('ecom')" in pragma or '("ecom")' in pragma:
                    allowExecutableComments = True
                    sourceLines[pragmaCommentLineIndex] = ecom
                elif "('noecom')" in pragma or '("noecom")' in pragma:
                    allowExecutableComments = False
                    sourceLines[pragmaCommentLineIndex] = noecom
                else:
                    sourceLines[pragmaCommentLineIndex] = indentation + tail.lstrip()

            # Use line indices of single-line comment-like pragma's to transform these into function-like pragma pairs
            for shortPragmaCommentLineIndex in shortPragmaCommentLineIndices:
                head, tail = sourceLines[shortPragmaCommentLineIndex].rsplit('#', 1)
                strippedHead = head.lstrip()
                indent = head[:len(head) - len(strippedHead)]
                pragmaName = tail.replace(' ', '').replace('\t', '')[3:]

                # Turn appropriate lines into executable comment switches
                if pragmaName == 'ecom':
                    sourceLines[pragmaCommentLineIndex] = ecom
                elif pragmaName == 'noecom':
                    sourceLines[pragmaCommentLineIndex] = noecom
                elif pragmaName.startswith('no'):
                    sourceLines[shortPragmaCommentLineIndex] = '{}__pragma__ (\'{}\'); {}; __pragma__ (\'{}\')'.format(indent, pragmaName, head, pragmaName[2:])    # Correct!
                else:
                    sourceLines[shortPragmaCommentLineIndex] = '{}__pragma__ (\'{}\'); {}; __pragma__ (\'no{}\')'.format(indent, pragmaName, head, pragmaName)

            # Switch executable comments on c.q. off and turn executable comments into normal code lines for Transcrypt (as opposed to CPython)
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

            # Return joined lines, to be used for parsing
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
            utils.enhanceException(
                syntaxError,
                lineNr=syntaxError.lineno,
                message=(
                    '\n\t{} [<-SYNTAX FAULT] {}'.format(
                        syntaxError.text[:syntaxError.offset].lstrip(),
                        syntaxError.text[syntaxError.offset:].rstrip()
                    )
                    if syntaxError.text else
                    syntaxError.args[0]
                )
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
                self.treeFragments.append('= {0}'.format(value))

        self.treeFragments: List[str] = []
        walk('file', self.parseTree, 0)
        self.textTree: str = ''.join(self.treeFragments)[1:]

        with utils.create(self.treePath) as treeFile:
            treeFile.write(self.textTree)

class Generator(ast.NodeVisitor):
    # Terms like parent, child, ancestor and descendant refer to the parse tree here, not to inheritance

    def __init__(self, module: Module) -> None:
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

        self.aliases: List[Tuple[str, str]] = [
            # Format: ('<Python source identifier>', '<JavaScript target identifier>')

            ('js_and', 'and'),
            ('arguments', 'py_arguments'), ('js_arguments', 'arguments'),
            ('case', 'py_case'),
            ('clear', 'py_clear'), ('js_clear', 'clear'),
            ('js_conjugate', 'conjugate'),
            ('default', 'py_default'),
            ('del', 'py_del'), ('js_del', 'del'),
            ('false', 'py_false'),
            ('js_from', 'from'),
            ('get', 'py_get'), ('js_get', 'get'),
            ('js_global', 'global'),
            ('Infinity', 'py_Infinity'), ('js_Infinity', 'Infinity'),
            ('is', 'py_is'), ('js_is', 'is'),
            ('isNaN', 'py_isNaN'), ('js_isNaN', 'isNaN'),
            ('iter', 'py_iter'), ('js_iter', 'iter'),
            ('items', 'py_items'), ('js_items', 'items'),
            ('keys', 'py_keys'), ('js_keys', 'keys'),
            ('name', 'py_name'), ('js_name', 'name'),
            ('NaN', 'py_NaN'), ('js_NaN', 'NaN'),
            ('new', 'py_new'),
            ('next', 'py_next'), ('js_next', 'next'),
            ('js_not', 'not'),
            ('js_or', 'or'),
            ('pop', 'py_pop'), ('js_pop', 'pop'),
            ('popitem', 'py_popitem'), ('js_popitem', 'popitem'),
            ('replace', 'py_replace'), ('js_replace', 'replace'),
            ('selector', 'py_selector'), ('js_selector', 'selector'),
            ('sort', 'py_sort'), ('js_sort', 'sort'),
            ('split', 'py_split'), ('js_split', 'split'),
            ('switch', 'py_switch'),
            ('type', 'py_metatype'), ('js_type', 'type'),    # Only for the type metaclass, the type operator is dealt with separately in visit_Call
            ('TypeError', 'py_TypeError'), ('js_TypeError', 'TypeError'),
            ('update', 'py_update'), ('js_update', 'update'),
            ('values', 'py_values'), ('js_values', 'values'),
            ('reversed', 'py_reversed'), ('js_reversed', 'reversed'),
            ('setdefault', 'py_setdefault'), ('js_setdefault', 'setdefault'),
            ('js_super', 'super'),
            ('true', 'py_true'),
            ('undefined', 'py_undefined'), ('js_undefined', 'undefined'),
        ]

        self.idFiltering: bool = True

        self.tempIndices: Dict[str, int] = {}
        self.skippedTemps: Set[str] = set()
        self.stubsName: str = 'org.{}.stubs.'.format(self.module.program.envir.transpiler_name)

        self.nameConsts: Dict[Optional[bool], str] = {
            None: 'null',
            True: 'true',
            False: 'false'
        }

        self.operators: Dict[type, Tuple[Optional[str], int]] = {
            # Highest precendence

            ast.Not: ('!', 16),
            ast.Invert: ('~', 16),
            ast.UAdd: ('+', 16),
            ast.USub: ('-', 16),

            ast.Pow: (None, 15),        # Dealt with separately

            ast.Mult: ('*', 14),
            ast.MatMult: (None, 14),    # Dealt with separately
            ast.Div: ('/', 14),
            ast.FloorDiv: (None, 14),   # Dealt with separately
            ast.Mod: ('%', 14),         # Used only for JavaScript %, which differs from Python %

            ast.Add: ('+', 13),
            ast.Sub: ('-', 13),

            ast.LShift: ('<<', 12),
            ast.RShift: ('>>', 12),

            ast.Lt: ('<', 11),
            ast.LtE: ('<=', 11),
            ast.Gt: ('>', 11),
            ast.GtE: ('>=', 11),
            ast.In: (None, 11),         # Dealt with separately
            ast.NotIn: (None, 11),      # Dealt with separately

            ast.Eq: ('==', 10),
            ast.NotEq: ('!=', 10),
            ast.Is: ('===', 10),        # Not really, but closest for now
            ast.IsNot: ('!==', 10),     # Not really, but closest for now

            ast.BitAnd: ('&', 9),
            ast.BitOr: ('|', 8),
            ast.BitXor: ('^', 7),
            ast.And: ('&&', 6),
            ast.Or: ('||', 5)

            # Lowest precedence
        }

        self.allowKeywordArgs: bool = utils.commandArgs.kwargs
        self.allowOperatorOverloading: bool = utils.commandArgs.opov
        self.allowConversionToIterable: bool = utils.commandArgs.iconv
        self.allowConversionToTruthValue: bool = utils.commandArgs.tconv
        self.allowKeyCheck: bool = utils.commandArgs.keycheck
        self.allowDebugMap: bool = utils.commandArgs.anno and not self.module.sourcePath.endswith('.js')
        self.allowDocAttribs: bool = utils.commandArgs.docat
        self.allowGlobals: bool = utils.commandArgs.xglobs
        self.allowJavaScriptIter: bool = False
        self.allowJavaScriptCall: bool = utils.commandArgs.jscall
        self.allowJavaScriptKeys: bool = utils.commandArgs.jskeys
        self.allowJavaScriptMod: bool = utils.commandArgs.jsmod
        self.allowMemoizeCalls: bool = utils.commandArgs.fcall

        self.noskipCodeGeneration: bool = True
        self.conditionalCodeGeneration: bool = True
        self.stripTuple: bool = False     # For speed, tuples are translated to bare JavaScript arrays if they're just indices. Will autoreset.
        self.stripTuples: bool = False    # For correctness, tuples are translated to bare JavaScript arrays if they are assignment target in a JavaScript 6 for-loop. Will not autoreset.
        self.replaceSend: bool = False

        try:
            self.visit(module.parseTree)
            self.targetFragments.append(self.lineNrString) # Last target fragment doesn't have a '\n' to replace in the emit method

        except Exception as exception:
            # print (traceback.format_exc ()) # Keep at hand for debugging purposes

            utils.enhanceException(
                exception,
                lineNr=self.lineNr
            )

        if self.tempIndices:
            raise utils.Error(
                message='\n\tTemporary variables leak in code generator: {}'.format(self.tempIndices)
            )

    def visitSubExpr(self, node: ast.AST, child: ast.AST) -> None:
        def getPriority(exprNode: ast.AST) -> int:
            if type(exprNode) in (ast.BinOp, ast.BoolOp):
                return self.operators[type(exprNode.op)][1]
            elif type(exprNode) == ast.Compare:
                return self.operators[type(exprNode.ops[0])][1]  # All ops have same priority
            elif type(exprNode) == ast.Yield:
                return -1000000
            else:
                return 1000000  # No need for parenthesis

        if getPriority(child) <= getPriority(node):
            self.emit('(')
            self.visit(child)
            self.emit(')')
        else:
            self.visit(child)

    def filterId(self, qualifiedId: str) -> str:   # Convention: only called at emission time or file name fabrication time

        if not self.idFiltering or (qualifiedId.startswith('__') and qualifiedId.endswith('__')):
            # Leave system dunder names unchanged
            return qualifiedId
        else:
            # Filter the rest, trying all aliases sucessively
            for alias in self.aliases:

                # Replace non-adjacent and odd adjacent matches, turning __<alias [0]>__ into =<alias [1]>=
                qualifiedId = re.sub(
                    fr'(^|(?P<preDunder