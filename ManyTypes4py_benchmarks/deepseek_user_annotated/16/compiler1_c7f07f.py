from typing import (
    Any, Dict, List, Set, Tuple, Optional, Union, Callable, Iterator, 
    Sequence, Iterable, Mapping, MutableMapping, TypeVar, Generic, 
    cast, overload, ContextManager, NoReturn
)
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
        moduleSearchDirs: List[str],
        symbols: Set[str],
        envir: Any
    ) -> None:
        utils.setProgram(self)
        self.moduleSearchDirs: List[str] = moduleSearchDirs
        self.symbols: Set[str] = symbols
        self.envir: Any = envir
        self.javascriptVersion: int = int(utils.commandArgs.esv) if utils.commandArgs.esv else 6
        self.moduleDict: Dict[str, 'Module'] = {}
        self.importStack: List[List[Union['Module', Optional[int]]] = []
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
            utils.enhanceException(
                exception,
                message=f'\n\t{exception}'
            )

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

        self.program.importStack.pop()
        utils.tryRemove(self.prettyTargetPath)
        utils.tryRemove(self.shrinkMapPath)
        utils.tryRemove(self.prettyMapPath)

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
