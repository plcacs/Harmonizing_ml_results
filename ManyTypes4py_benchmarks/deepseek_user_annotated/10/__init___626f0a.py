# File: transcrypt/modules/warnings/__init__.py
# Author: Carl Allendorph
# Date: 20NOV2016
#
# Description:
#   This file contains the implementation of a module to
# provide features that can be used by developer to flag
# runtime issues that would not be caught by the static
# type checking system. These warnings are for conditions
# that aren't necessarily fatal but denote differences in
# the transcrypt implementation from the normal cpython
# implementation
#

from typing import (
    Any, Dict, List, Optional, Pattern, Set, Tuple, Type, TypeVar, Union,
    Callable, cast
)
import re

T = TypeVar('T', bound='Warning')

class Actions:
    error: str = "error"
    ignore: str = "ignore"
    always: str = "always"
    defaultact: str = "default"
    module: str = "module"
    once: str = "once"

ActionSet: Set[str] = set([x for x in dir(Actions) if not x.startswith("_")])

CategoryMap: Dict[str, Type[Warning]] = {
    "UserWarning": UserWarning,
    "DeprecationWarning": DeprecationWarning,
    "RuntimeWarning": RuntimeWarning,
}

_warnings_defaults: bool = False

filters: List[Tuple[str, Optional[Pattern[str]], Type[Warning], Optional[Pattern[str]], int]] = []
defaultaction: str = Actions.defaultact
onceregistry: Dict[Tuple[str, Type[Warning]], int] = {}

_filters_version: int = 1

def _filters_mutated() -> None:
    global _filters_version
    _filters_version += 1

class WarningMessage:
    def __init__(
        self,
        message: Union[str, Warning],
        category: Union[str, Type[Warning]],
        filename: str,
        lineno: int,
        file: Optional[Any] = None,
        line: Optional[str] = None
    ) -> None:
        self.message: Union[str, Warning] = message
        self.category: Union[str, Type[Warning]] = category
        self.filename: str = filename
        self.lineno: int = lineno
        self.file: Optional[Any] = file
        self.line: Optional[str] = line
        self._category_name: Optional[str] = category.__name__ if hasattr(category, '__name__') else str(category)

    def __str__(self) -> str:
        return ("{{message : {}, category : {}, filename : {}, lineno : {}, "
                "line : {} }}".format(self.message, self._category_name,
                                     self.filename, self.lineno, self.line))

def showwarning(
    message: Union[str, Warning],
    category: Type[Warning],
    filename: str,
    lineno: int,
    file: Optional[Any] = None,
    line: Optional[str] = None
) -> None:
    msg = WarningMessage(message, category, filename, lineno, file, line)
    _showwarnmsg_impl(msg)

def formatwarning(
    message: Union[str, Warning],
    category: Type[Warning],
    filename: str,
    lineno: int,
    line: Optional[str] = None
) -> str:
    msg = WarningMessage(message, category, filename, lineno, None, line)
    return _formatwarnmsg_impl(msg)

def _showwarnmsg_impl(msg: WarningMessage) -> None:
    f = msg.file
    text = _formatwarnmsg(msg)
    if f is None:
        text = text.rstrip('\r\n')
        console.log(text)
    else:
        try:
            f.write(text)
        except Exception:
            pass

def _formatwarnmsg_impl(msg: WarningMessage) -> str:
    s = "{}:{}: {}: {}\n".format(
        msg.filename, msg.lineno, msg.category, str(msg.message)
    )
    if msg.line:
        line = msg.line.strip()
        s += "  {}\n".format(line)
    return s

_showwarning: Callable[..., None] = showwarning

def setShowWarning(func: Callable[..., None]) -> None:
    if not callable(func):
        raise TypeError("showwarning method must be callable")
    global showwarning
    showwarning = func

def _showwarnmsg(msg: WarningMessage) -> None:
    if not callable(showwarning):
        raise TypeError("warnings.showwarning() must be set to a function or method")
    showwarning(msg.message, cast(Type[Warning], msg.category), msg.filename, msg.lineno,
                msg.file, msg.line)

_formatwarning: Callable[..., str] = formatwarning

def _formatwarnmsg(msg: WarningMessage) -> str:
    global formatwarning
    if formatwarning is not _formatwarning:
        return formatwarning(msg.message, cast(Type[Warning], msg.category),
                           msg.filename, msg.lineno, line=msg.line)
    return _formatwarnmsg_impl(msg)

def addWarningCategory(cat: Type[Warning]) -> None:
    name = cat.__name__
    if name not in CategoryMap:
        CategoryMap[name] = cat
    else:
        raise Exception("Warning Category {} already exists".format(name))

def filterwarnings(
    action: str,
    message: str = "",
    category: Type[Warning] = Warning,
    module: str = "",
    lineno: int = 0,
    append: bool = False
) -> None:
    assert action in ActionSet, "invalid action: {}".format(action)
    assert isinstance(message, str), "message must be a string"
    assert isinstance(module, str), "module must be a string"
    assert isinstance(lineno, int) and lineno >= 0, "lineno must be an int >= 0"
    _add_filter(
        action,
        re.compile(message, re.I) if message else None,
        category,
        re.compile(module) if module else None,
        lineno,
        append=append
    )

def simplefilter(
    action: str,
    category: Type[Warning] = Warning,
    lineno: int = 0,
    append: bool = False
) -> None:
    assert action in ActionSet, "invalid action: {}".format(action)
    assert isinstance(lineno, int) and lineno >= 0, "lineno must be an int >= 0"
    _add_filter(action, None, category, None, lineno, append=append)

def _add_filter(*item: Any, append: bool) -> None:
    if not append:
        try:
            filters.remove(item)
        except ValueError:
            pass
        filters.insert(0, item)
    else:
        if item not in filters:
            filters.append(item)
    _filters_mutated()

def resetwarnings() -> None:
    global filters
    filters = []
    _filters_mutated()

__warningregistry__: Dict[Any, Any] = {}

def _checkCatMatch(msgCat: Type[Warning], filtCat: Type[Warning]) -> bool:
    return msgCat.__name__ == filtCat.__name__

def warn_explicit(
    message: Union[str, Warning],
    category: Type[Warning],
    filename: str,
    lineno: int,
    module: Optional[str] = None,
    registry: Optional[Dict[Any, Any]] = None,
    module_globals: Optional[Dict[str, Any]] = None
) -> None:
    lineno = int(lineno)
    if module is None:
        module = filename or "<unknown>"
        if module[-3:].lower() == ".py":
            module = module[:-3]
    if registry is None:
        registry = __warningregistry__

    try:
        currVersion = registry["version"]
    except KeyError:
        currVersion = 0

    if currVersion != _filters_version:
        registry.clear()
        registry['version'] = _filters_version

    if isinstance(message, Warning):
        text = str(message)
        category = message.__class__
    else:
        text = message
        message = category(message)
    key = (text, category, lineno)
    if key in registry:
        return

    for item in filters:
        action, msg, cat, mod, ln = item
        if ((msg is None or msg.match(text)) and
            _checkCatMatch(category, cat) and
            (mod is None or mod.match(module)) and
            (ln == 0 or lineno == ln)):
            break
    else:
        action = defaultaction

    if action == Actions.ignore:
        registry[key] = 1
        return

    if action == Actions.error:
        raise message

    if action == Actions.once:
        registry[key] = 1
        oncekey = (text, category)
        if oncekey in onceregistry:
            return
        onceregistry[oncekey] = 1
    elif action == Actions.always:
        pass
    elif action == Actions.module:
        registry[key] = 1
        altkey = (text, category, 0)
        if altkey in registry:
            return
        registry[altkey] = 1
    elif action == Actions.defaultact:
        registry[key] = 1
    else:
        raise RuntimeError(f"Unrecognized action ({action}) in warnings.filters:\n {item}")

    msg = WarningMessage(message, category.__name__, filename, lineno)
    _showwarnmsg(msg)

class catch_warnings:
    def __init__(self, *, record: bool = False, module: Optional[Any] = None) -> None:
        self._record = record
        self._entered = False
        raise NotImplementedError("with/as not well supported in transcrypt")

class _OptionError(Exception):
    pass

def setWarningOptions(opts: List[str]) -> None:
    _processoptions(opts)

def _processoptions(args: List[str]) -> None:
    for arg in args:
        try:
            _setoption(arg)
        except _OptionError as msg:
            console.log(f"WARNING: Invalid -W option ignored: {msg}")

def _setoption(arg: str) -> None:
    parts = arg.split(':')
    if len(parts) > 5:
        raise _OptionError(f"too many fields (max 5): {arg}")
    while len(parts) < 5:
        parts.append('')
    action, message, category, module, lineno = [s.strip() for s in parts]
    action = _getaction(action)
    message = re.escape(message)
    category = _getcategory(category)
    module = re.escape(module)
    if module:
        module = module + '$'
    if lineno:
        try:
            lineno = int(lineno)
            if lineno < 0:
                raise ValueError
        except (ValueError, OverflowError):
            raise _OptionError(f"invalid lineno {lineno}")
    else:
        lineno = 0
    filterwarnings(action, message, category, module, lineno)

def _getaction(action: str) -> str:
    if not action:
        return Actions.defaultact
    if action == "all":
        return Actions.always
    for a in ActionSet:
        if a.startswith(action):
            return a
    raise _OptionError(f"invalid action: {action}")

def _getcategory(category: str) -> Type[Warning]:
    if not category:
        return Warning
    if category in CategoryMap:
        try:
            cat = CategoryMap[category]
        except NameError:
            raise _OptionError(f"unknown warning category: {category}")
    else:
        raise Exception(f"Unable to import category: {category}, use `addWarningCategory`")
    return cat

if not _warnings_defaults:
    silence = [DeprecationWarning]
    for cls in silence:
        simplefilter(Actions.ignore, category=cls)
