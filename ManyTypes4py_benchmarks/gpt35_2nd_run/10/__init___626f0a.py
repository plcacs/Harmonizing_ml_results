import re
from typing import Set, Dict, Any, Callable, Union, Tuple

class Actions:
    error: str = 'error'
    ignore: str = 'ignore'
    always: str = 'always'
    defaultact: str = 'default'
    module: str = 'module'
    once: str = 'once'

ActionSet: Set[str] = set([x for x in dir(Actions) if not x.startswith('_')])
CategoryMap: Dict[str, Any] = {'UserWarning': UserWarning, 'DeprecationWarning': DeprecationWarning, 'RuntimeWarning': RuntimeWarning}

_warnings_defaults: bool = False
filters: list = []
defaultaction: str = Actions.defaultact
onceregistry: Dict[str, int] = {}
_filters_version: int = 1

def _filters_mutated() -> None:
    global _filters_version
    _filters_version += 1

def showwarning(message: str, category: Any, filename: str, lineno: int, file=None, line=None) -> None:
    ...

def formatwarning(message: str, category: Any, filename: str, lineno: int, line=None) -> str:
    ...

def _showwarnmsg_impl(msg: Any) -> None:
    ...

def _formatwarnmsg_impl(msg: Any) -> str:
    ...

def setShowWarning(func: Callable) -> None:
    ...

def _showwarnmsg(msg: Any) -> None:
    ...

def _formatwarnmsg(msg: Any) -> str:
    ...

def addWarningCategory(cat: Any) -> None:
    ...

def filterwarnings(action: str, message: str = '', category: Any = Warning, module: str = '', lineno: int = 0, append: bool = False) -> None:
    ...

def simplefilter(action: str, category: Any = Warning, lineno: int = 0, append: bool = False) -> None:
    ...

def _add_filter(*item: Any, append: bool) -> None:
    ...

def resetwarnings() -> None:
    ...

__warningregistry__: Dict[str, Any] = {}

def _checkCatMatch(msgCat: Any, filtCat: Any) -> bool:
    ...

def warn_explicit(message: Union[str, Warning], category: Any, filename: str, lineno: int, module=None, registry=None, module_globals=None) -> None:
    ...

class WarningMessage:
    def __init__(self, message: str, category: Any, filename: str, lineno: int, file=None, line=None) -> None:
        ...

    def __str__(self) -> str:
        ...

class catch_warnings:
    def __init__(self, *, record: bool = False, module=None) -> None:
        ...

def setWarningOptions(opts: list) -> None:
    ...

class _OptionError(Exception):
    ...

def _processoptions(args: list) -> None:
    ...

def _setoption(arg: str) -> None:
    ...

def _getaction(action: str) -> str:
    ...

def _getcategory(category: str) -> Any:
    ...

if not _warnings_defaults:
    silence: list = [DeprecationWarning]
    for cls in silence:
        simplefilter(Actions.ignore, category=cls)
