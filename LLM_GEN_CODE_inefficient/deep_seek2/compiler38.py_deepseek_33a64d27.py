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
from typing import Any, List, Dict, Set, Tuple, Optional, Union, Callable

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

        self.moduleDict: Dict[str, Any] = {}    # Administration of all modules that play a role in this program
        self.importStack: List[List[Any]] = []   # Pending imports, enables showing load sequence in case a module cannot be loaded

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

    def provide(self, moduleName: str, __moduleName__: Optional[str] = None, filter: Optional[Callable[[str], str] = None) -> Any:
        # moduleName may contain dots if it's imported, but it'll have the same name in every import

        if moduleName in self.moduleDict:  # Find out if module is already provided
            return self.moduleDict[moduleName]
        else:                              # If not, provide by loading or compiling
            # This may fail legally if filteredModuleName ends on a name of something in a module, rather than of the module itself
            return Module(self, moduleName, __moduleName__, filter)

class Module:
    def __init__(self, program: Program, name: str, __name__: Optional[str], filter: Optional[Callable[[str], str]]) -> None:
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
            self.program.searchedModulePaths: List[str] = []
            self.program.provide(importedModuleName)

        # Remove eventual intermediate files
        utils.tryRemove(self.prettyTargetPath)
        utils.tryRemove(self.shrinkMapPath)
        utils.tryRemove(self.prettyMapPath)

        # Module not under compilation anymore, so pop it
        self.program.importStack.pop()

    def findPaths(self, filter: Optional[Callable[[str], str]]) -> None:
        # Filter to get hyphens and/or dots in name if a suitable alias is defined
        # The filter function, and with it the active aliases, are passed by the importing module
        rawRelSourceSlug: str = self.name.replace('.', '/')
        relSourceSlug: str = filter(rawRelSourceSlug) if filter and utils.commandArgs.alimod else rawRelSourceSlug

        '''
        # BEGIN DEBUGGING CODE
        print()
        print('Raw slug   :', rawRelSourceSlug)
        print('Cooked slug:', relSourceSlug)
        print()
        # END DEBUGGING CODE
        '''

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
