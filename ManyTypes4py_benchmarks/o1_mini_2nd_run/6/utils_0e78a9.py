import os
import sys
import argparse
import inspect
import tokenize
import re
import time
import traceback
from typing import Any as TypingAny, Optional, List, IO, Union, NoReturn

class Any:

    def __init__(self, **attribs: TypingAny) -> None:
        for attrib in attribs:
            setattr(self, attrib, attribs[attrib])

defaultJavaScriptVersion: int = 5

class CommandArgsError(BaseException):
    pass

class CommandArgsExit(BaseException):
    pass

class ArgumentParser(argparse.ArgumentParser):

    def error(self, message: str) -> None:
        self.print_help(sys.stdout)
        if message:
            log(True, '\nError: {}\n', message)
        raise CommandArgsError()

    def exit(self, status: int = 0, message: Optional[str] = None) -> None:
        if message:
            log(True, 'Exit: {}', message)
        raise CommandArgsExit()

class CommandArgs:

    def parse(self) -> None:
        self.argParser = ArgumentParser()
        self.argParser.add_argument('source', nargs='?', help='.py file containing source code of main module')
        self.argParser.add_argument('-a', '--anno', help='annotate target files that were compiled from Python with source file names and source line numbers', action='store_true')
        self.argParser.add_argument('-am', '--alimod', help='use aliasing for module paths', action='store_true')
        self.argParser.add_argument('-b', '--build', help='rebuild all target files from scratch', action='store_true')
        self.argParser.add_argument('-c', '--complex', help='enable complex number support, locally requires operator overloading', action='store_true')
        self.argParser.add_argument('-d', '--docat', help='enable __doc__ attributes. Apply sparsely, since it will make docstrings part of the generated code', action='store_true')
        self.argParser.add_argument('-da', '--dassert', help='debug: activate assertions', action='store_true')
        self.argParser.add_argument('-dc', '--dcheck', help='debug: perform lightweight consistency check', action='store_true')
        self.argParser.add_argument('-de', '--dextex', help='debug: show extended exception reports', action='store_true')
        self.argParser.add_argument('-dl', '--dlog', help='debug: log compiler messages to disk', action='store_true')
        self.argParser.add_argument('-dm', '--dmap', help='debug: dump human readable source map', action='store_true')
        self.argParser.add_argument('-dn', '--dnostrip', help='debug: no comment stripping of __core__ and __builtin__ in-line modules', action='store_true')
        self.argParser.add_argument('-ds', '--dstat', help='debug: validate static typing using annotations', action='store_true')
        self.argParser.add_argument('-dt', '--dtree', help='debug: dump syntax tree', action='store_true')
        self.argParser.add_argument('-e', '--esv', nargs='?', help='ecma script version of generated code, default = 6. The symbol __esv<versionnr>__ is added to the global symbol list, e.g. __esv7__.')
        self.argParser.add_argument('-ec', '--ecom', help='enable executable comments, seen as comments by CPython but as executable statements by Transcrypt', action='store_true')
        self.argParser.add_argument('-f', '--fcall', help="enable fastcall mechanism by default. You can also use __pragma__ ('fcal') and __pragma__ ('nofcall')", action='store_true')
        self.argParser.add_argument('-g', '--gen', help="enable generators and iterators. Disadvised, since it will result in a function call for each loop iteration. Preferably use __pragma__ ('gen') and __pragma__ ('nogen')", action='store_true')
        self.argParser.add_argument('-i', '--iconv', help="enable automatic conversion to iterable by default. Disadvised, since it will result in a type check for each for-loop. Preferably use __pragma__ ('iconv') and __pragma__ ('noiconv') to enable automatic conversion locally", action='store_true')
        self.argParser.add_argument('-jc', '--jscall', help="enable native JavaScript calls for Python methods. This is fast, but doesn't support bound method assignment, decorators and non-instance methods. Preferably use __pragma__ ('jscall') and __pragma__ ('nojscall') to enable native JavaScript calls locally", action='store_true')
        self.argParser.add_argument('-jk', '--jskeys', help="interpret {key: 'value'} as {'key': 'value'} and forbid {key (): 'value'}, as JavaScript does. Disadvised, since it's less flexible than the Python interpretation. Either follow Python semantics by using {'key': 'value'} explicitly if you want literal keys or use __pragma__ ('jskeys') and __pragma__ ('nojskeys') locally instead to make clear local deviation from Python semantics", action='store_true')
        self.argParser.add_argument('-jm', '--jsmod', help="give %% and %%= JavaScript rather than Python behaviour. Disadvised, since it deviates from the mathematical 'modulo' operator. Either follow Python semantics or use __pragma__ ('jsmod') and __pragma__ ('nojsmod') locally instead to make clear local deviation.", action='store_true')
        self.argParser.add_argument('-k', '--kwargs', help="enable keyword arguments by default. In general this is disadvised, use __pragma__ ('kwargs') and __pragma__('nokwargs') locally instead to prevent bloated code", action='store_true')
        self.argParser.add_argument('-kc', '--keycheck', help="enable checking for existence of dictionary keys. In general this is disadvised, use __pragma__ ('keycheck') and __pragma__('nokeycheck') locally instead to prevent bloated code", action='store_true')
        self.argParser.add_argument('-l', '--license', help='show license', action='store_true')
        self.argParser.add_argument('-m', '--map', help='generate source map', action='store_true')
        self.argParser.add_argument('-n', '--nomin', help='no minification', action='store_true')
        self.argParser.add_argument('-o', '--opov', help="enable operator overloading by default. In general this is disadvised, use __pragma__ ('opov') and __pragma__('noopov') locally instead to prevent slow code", action='store_true')
        self.argParser.add_argument('-od', '--outdir', help='override output directory (default = __target__)')
        self.argParser.add_argument('-p', '--parent', nargs='?', help='object that will hold application, default is window. Use -p .none to generate orphan application, e.g. for use in node.js')
        self.argParser.add_argument('-r', '--run', help='run source file rather than compiling it', action='store_true')
        self.argParser.add_argument('-s', '--symbols', nargs='?', help='names, joined by $, separately passed to main module in __symbols__ variable')
        self.argParser.add_argument('-sf', '--sform', help='enable support for string formatting mini language', action='store_true')
        self.argParser.add_argument('-t', '--tconv', help="enable automatic conversion to truth value by default. Disadvised, since it will result in a conversion for each boolean. Preferably use __pragma__ ('tconv') and __pragma__ ('notconv') to enable automatic conversion locally", action='store_true')
        self.argParser.add_argument('-u', '--unit', nargs='?', help='compile to units rather than to monolithic application. Use -u .auto to autogenerate dynamically loadable native JavaScript modules, one per Python module. Use -u .run to generate the loader and the staticcally loadable runtime unit. Use -u .com to generate a statically loadable component unit.')
        self.argParser.add_argument('-v', '--verbose', help='show all messages', action='store_true')
        self.argParser.add_argument('-x', '--x', help='reserved for extended options')
        self.argParser.add_argument('-xr', '--xreex', help='re-export all imported names', action='store_true')
        self.argParser.add_argument('-xg', '--xglobs', help="allow use of the 'globals' function", action='store_true')
        self.argParser.add_argument('-xp', '--xpath', nargs='?', help="additional module search paths, joined by $, #'s will be replaced by spaces")
        self.argParser.add_argument('-xt', '--xtiny', help='generate tiny version of runtime, a.o. lacking support for implicit and explicit operator overloading. Use only if generated code can be validated, since it will introduce semantic alterations in edge cases', action='store_true')
        self.argParser.add_argument('-*', '--star', help='Like it? Grow it! Go to GitHub and then click [* Star]', action='store_true')
        self.projectOptions: dict = self.argParser.parse_args().__dict__
        self.__dict__.update(self.projectOptions)

        def logAndExit(message: str) -> None:
            log(True, message)
            sys.exit(1)
        invalidCombi: str = 'Invalid combination of options'
        if not (self.license or self.star or self.source):
            logAndExit(self.argParser.format_usage().capitalize())
        elif self.map and self.unit:
            logAndExit('{}: -m / --map and -u / --unit'.format(invalidCombi))
        elif self.parent and self.unit == '.com':
            logAndExit('{}: -p / --parent and -u / --unit .com'.format(invalidCombi))
        elif self.parent == '.export' and self.esv and (int(self.esv) < 6):
            logAndExit('{}: -p / --parent .export and -e / --esv < 6'.format(invalidCombi))
        elif self.unit == '.auto' and self.esv and (int(self.esv) < 6):
            logAndExit('{}: -u / --unit .auto and -e / --esv < 6'.format(invalidCombi))
        global extraLines
        extraLines = ['def __pragma__ (): pass', "__pragma__ ('skip')", '__new__ = __include__ = 0', "__pragma__ ('noskip')", ''] if commandArgs.dcheck else []
        global nrOfExtraLines
        nrOfExtraLines = max(len(extraLines) - 1, 0)
        extraLines = '\n'.join(extraLines)

commandArgs: CommandArgs = CommandArgs()

def create(path: str, binary: bool = False) -> IO[Union[bytes, str]]:
    for i in range(10):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if binary:
                return open(path, 'wb')
            else:
                return open(path, 'w', encoding='utf-8')
            if i > 0:
                log(True, f'Created {path} at attempt {i + 1}')
        except Exception:
            time.sleep(0.5)
    else:
        raise Error(f'Failed to create {path}')

def tryRemove(filePath: str) -> None:
    try:
        os.remove(filePath)
    except Exception:
        pass

def formatted(*args: TypingAny) -> str:
    try:
        return str(args[0]).format(*args[1:])
    except IndexError:
        return ' '.join(str(arg) for arg in args)

logFileName: str = 'transcrypt.log'
try:
    os.remove(logFileName)
except Exception:
    pass

def log(always: bool, *args: TypingAny) -> None:
    if always or commandArgs.verbose:
        print(formatted(*args), end='')
        try:
            if commandArgs.dlog:
                with open(logFileName, 'a', encoding='utf-8') as logFile:
                    logFile.write(formatted(*args))
        except Exception:
            pass

program: Optional[TypingAny] = None

def setProgram(aProgram: TypingAny) -> None:
    global program
    program = aProgram

class Error(Exception):

    def __init__(self, lineNr: int = 0, message: str = '') -> None:
        self.lineNr: int = lineNr - nrOfExtraLines
        self.message: str = message

    def set(self, lineNr: int = 0, message: str = '') -> None:
        if not self.lineNr:
            self.lineNr = lineNr - nrOfExtraLines
        if not self.message:
            self.message = message

    def __str__(self) -> str:
        result: str = 'Error while compiling (offending file last):'
        for importRecord in program.importStack[:-1]:
            try:
                sourcePath: str = importRecord[0].sourcePath
            except AttributeError:
                sourcePath = '<unknown>'
            result += "\n\tFile '{}', line {}, at import of:".format(sourcePath, importRecord[1])
        result += "\n\tFile '{}', line {}, namely:".format(str(program.importStack[-1][0].name), self.lineNr)
        result += '\n\t{}'.format(self.message)
        return result

def enhanceException(exception: Exception, **kwargs: TypingAny) -> NoReturn:
    if isinstance(exception, Error):
        exception.set(**kwargs)
        result: Error = exception
    else:
        result: Error = Error(**kwargs)
    if commandArgs.dextex:
        stack = inspect.stack()
        result_str = '\n    Exception of class {0} enhanced at:\n        file: {1}\n        function: {3}\n        line: {2}\n        context: {4}\n        kwargs: {5}\n        result: {6}\n    '.format(
            exception.__class__, 
            stack[1].filename if len(stack) > 1 else '<unknown>', 
            stack[1].lineno if len(stack) > 1 else 0, 
            stack[1].function if len(stack) > 1 else '<unknown>', 
            stack[1].code_context[0].strip() if len(stack) > 1 and stack[1].code_context else '<no context>', 
            kwargs, 
            result
        )
        print(result_str)
    raise result from None

def digestJavascript(code: str, symbols: List[str], mayStripComments: bool, mayRemoveAnnotations: bool, refuseIfAppearsMinified: bool = False) -> Any:
    """
    - Honor ifdefs
    - Strip comments if allowed by command line switch AND indicated by pragma
    - Harvest import and export info
    """
    if refuseIfAppearsMinified and code[0] != '/':
        return None
    stripComments: bool = False

    def stripSingleLineComments(line: str) -> str:
        pos: int = line.find('//')
        return (line if pos < 0 else line[:pos]).rstrip()

    passStack: List[bool] = []

    def passable(targetLine: str) -> bool:
        nonlocal stripComments

        def __pragma__(name: str, *args: str) -> None:
            nonlocal stripComments
            if name == 'stripcomments':
                stripComments = mayStripComments
            if name == 'ifdef':
                passStack.append(args[0] in symbols)
            elif name == 'ifndef':
                passStack.append(not args[0] in symbols)
            elif name == 'else':
                if passStack:
                    passStack[-1] = not passStack[-1]
            elif name == 'endif':
                if passStack:
                    passStack.pop()

        strippedLine: str = targetLine.lstrip()
        if stripComments and strippedLine.startswith('/*'):
            passStack.append(False)
            return all(passStack)
        elif stripComments and strippedLine.endswith('*/'):
            if passStack:
                passStack.pop()
        elif strippedLine.startswith('__pragma__') and (
            'stripcomments' in strippedLine or 
            'ifdef' in strippedLine or 
            'ifndef' in strippedLine or 
            'else' in strippedLine or 
            'endif' in strippedLine
        ):
            exec(strippedLine)
            return False
        return all(passStack)

    passableLines: List[str] = [line for line in code.split('\n') if passable(line)]
    if stripComments:
        passableLines = [commentlessLine for commentlessLine in [stripSingleLineComments(line) for line in passableLines] if commentlessLine]
    result: Any = Any(digestedCode='\n'.join(passableLines), nrOfLines=len(passableLines), exportedNames=[], importedModuleNames=[])
    namesPattern: re.Pattern = re.compile('({.*})')
    pathPattern: re.Pattern = re.compile('([\'|"].*[\'|"])')
    wordPattern: re.Pattern = re.compile('[\\w/*$]+')
    for line in passableLines:
        words: List[str] = wordPattern.findall(line)
        if words:
            if mayRemoveAnnotations and words[0] == '/*':
                words = words[3:]
            if words:
                if words[0] == 'export':
                    if words[1] in {'var', 'function'}:
                        result.exportedNames.append(words[2])
                    else:
                        match: Optional[re.Match] = namesPattern.search(line)
                        if match:
                            extracted = wordPattern.findall(match.group(1))
                            result.exportedNames.extend([name for name in extracted])
                elif words[0] == 'import':
                    match: Optional[re.Match] = pathPattern.search(line)
                    if match:
                        module_path: str = eval(match.group(1))[2:-3]
                        result.importedModuleNames.append(module_path)
    return result

