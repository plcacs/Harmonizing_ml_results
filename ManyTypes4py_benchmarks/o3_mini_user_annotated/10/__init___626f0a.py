from typing import Any, Callable, Dict, List, Optional, Pattern, Tuple, Type, Union
import re

###########################################
# Transcrypt differences with Python `warnings` module
#
# 1) We don't have import/eval so this makes passing
#       in custom "Warning" types hard. To help with this,
#       I've added a new method "addWarningCategory" that
#       allows the user to add a new warning type via a
#       programmatic interface.
# 2) There is a limited subset of warnings right now to keep
#       things small.
# 3) catch_warnings is not implemented currently because the
#       transcrypt framework currently does not support "with/as"
#       clauses very well.
# 4) This module does not implement the `warn` method. The user
#       is suggested to use the `warn_explicit` method instead.
#       In order to prevent cases where a NonImplementedError is
#       thrown instead of a useful warning, the `warn` method is
#       not present - forcing the developer to pick a better
#       solution (like use `warn_explicit`).

###########################################
# Module initialization

class Actions:
    error: str = "error"
    ignore: str = "ignore"
    always: str = "always"
    defaultact: str = "default"
    module: str = "module"
    once: str = "once"

ActionSet = set([x for x in dir(Actions) if not x.startswith("_")])

# Map for warning category names to types
CategoryMap: Dict[str, Type[Warning]] = {
    "UserWarning": UserWarning,
    "DeprecationWarning": DeprecationWarning,
    "RuntimeWarning": RuntimeWarning,
}

_warnings_defaults: bool = False

# filters contains a sequence of filter 5-tuples
# Tuple elements: (action: str, message_regex: Optional[Pattern[str]], category: Type[Warning],
# module_regex: Optional[Pattern[str]], lineno: int)
filters: List[Tuple[str, Optional[Pattern[str]], Type[Warning], Optional[Pattern[str]], int]] = []
defaultaction: str = Actions.defaultact
onceregistry: Dict[Tuple[str, Type[Warning]], int] = {}

_filters_version: int = 1

def _filters_mutated() -> None:
    global _filters_version
    _filters_version += 1

def showwarning(message: Any, category: Any, filename: str, lineno: int,
                file: Optional[Any] = None, line: Optional[str] = None) -> None:
    """Hook to write a warning to a file; replace if you like."""
    msg = WarningMessage(message, category, filename, lineno, file, line)
    _showwarnmsg_impl(msg)

def formatwarning(message: Any, category: Any, filename: str, lineno: int,
                  line: Optional[str] = None) -> str:
    """Function to format a warning the standard way."""
    msg = WarningMessage(message, category, filename, lineno, None, line)
    return _formatwarnmsg_impl(msg)

def _showwarnmsg_impl(msg: "WarningMessage") -> None:
    """ Default Show Message Implementation """
    f = msg.file
    text: str = _formatwarnmsg(msg)
    if f is None:
        text = text.rstrip('\r\n')
        console.log(text)
    else:
        try:
            f.write(text)
        except Exception:
            pass

def _formatwarnmsg_impl(msg: "WarningMessage") -> str:
    """ @note - we've removed the msg.source handling and the associated
    tracemalloc code as this isn't available in the js runtime.
    """
    s: str = "{}:{}: {}: {}\n".format(
        msg.filename, msg.lineno, msg.category, str(msg.message)
    )
    if msg.line:
        line_str: str = msg.line.strip()
        s += "  {}\n".format(line_str)
    return s

_showwarning: Callable[..., None] = showwarning

def setShowWarning(func: Callable[[Any, Any, str, int, Optional[Any], Optional[str]], None]) -> None:
    if not callable(func):
        raise TypeError("showwarning method must be callable")
    global showwarning
    showwarning = func

def _showwarnmsg(msg: "WarningMessage") -> None:
    if not callable(showwarning):
        raise TypeError("warnings.showwarning() must be set to a function or method")
    showwarning(msg.message, msg.category, msg.filename, msg.lineno, msg.file, msg.line)

_formatwarning: Callable[..., str] = formatwarning

def _formatwarnmsg(msg: "WarningMessage") -> str:
    global formatwarning
    if formatwarning is not _formatwarning:
        return formatwarning(msg.message, msg.category, msg.filename, msg.lineno, line=msg.line)
    return _formatwarnmsg_impl(msg)

def addWarningCategory(cat: Type[Warning]) -> None:
    name: str = cat.__name__
    if name not in CategoryMap:
        CategoryMap[name] = cat
    else:
        raise Exception("Warning Category {} already exists".format(name))

def filterwarnings(action: str, message: str = "", category: Type[Warning] = Warning,
                   module: str = "", lineno: int = 0, append: bool = False) -> None:
    assert action in ActionSet, "invalid action: {}".format(action)
    assert isinstance(message, str), "message must be a string"
    assert isinstance(module, str), "module must be a string"
    assert isinstance(lineno, int) and lineno >= 0, "lineno must be an int >= 0"
    _add_filter(action, re.compile(message, re.I), category,
                re.compile(module) if module else None, lineno, append=append)

def simplefilter(action: str, category: Type[Warning] = Warning,
                 lineno: int = 0, append: bool = False) -> None:
    assert action in ActionSet, "invalid action: {}".format(action)
    assert isinstance(lineno, int) and lineno >= 0, "lineno must be an int >= 0"
    _add_filter(action, None, category, None, lineno, append=append)

def _add_filter(*item: Any, append: bool) -> None:
    if not append:
        try:
            filters.remove(item)
        except Exception:
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

def warn_explicit(message: Union[Warning, Any], category: Type[Warning], filename: str,
                  lineno: int, module: Optional[str] = None,
                  registry: Optional[Dict[Any, Any]] = None, module_globals: Any = None) -> None:
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
        registry["version"] = _filters_version
    if isinstance(message, Warning):
        text: str = str(message)
        category = message.__class__
    else:
        text = message
        message = category(message)
    key: Tuple[str, Type[Warning], int] = (text, category, lineno)
    if key in registry:
        return
    for item in filters:
        action_filt, msg_regex, cat, mod_regex, ln in item
        if ((msg_regex is None or msg_regex.match(text)) and
            _checkCatMatch(category, cat) and
            (mod_regex is None or mod_regex.match(module)) and
            (ln == 0 or lineno == ln)):
            break
    else:
        action_filt = defaultaction
    if action_filt == Actions.ignore:
        registry[key] = 1
        return
    if action_filt == Actions.error:
        raise message
    if action_filt == Actions.once:
        registry[key] = 1
        oncekey: Tuple[str, Type[Warning]] = (text, category)
        if oncekey in onceregistry:
            return
        onceregistry[oncekey] = 1
    elif action_filt == Actions.always:
        pass
    elif action_filt == Actions.module:
        registry[key] = 1
        altkey: Tuple[str, Type[Warning], int] = (text, category, 0)
        if altkey in registry:
            return
        registry[altkey] = 1
    elif action_filt == Actions.defaultact:
        registry[key] = 1
    else:
        raise RuntimeError("Unrecognized action ({}) in warnings.filters:\n {}".format(action_filt, item))
    msg = WarningMessage(message, category.__name__, filename, lineno)
    _showwarnmsg(msg)

class WarningMessage(object):
    def __init__(self, message: Any, category: Union[Type[Warning], str], filename: str,
                 lineno: int, file: Optional[Any] = None, line: Optional[str] = None) -> None:
        self.message: Any = message
        self.category: Union[Type[Warning], str] = category
        self.filename: str = filename
        self.lineno: int = lineno
        self.file: Optional[Any] = file
        self.line: Optional[str] = line
        self._category_name: Optional[str] = category.__name__ if hasattr(category, "__name__") else None

    def __str__(self) -> str:
        return ("{{message : {}, category : {}, filename : {}, lineno : {}, line : {} }}".format(
            self.message, self._category_name, self.filename, self.lineno, self.line))

def _checkCatMatch(msgCat: Type[Warning], filtCat: Type[Warning]) -> bool:
    return msgCat.__name__ == filtCat.__name__

class catch_warnings(object):
    def __init__(self, *, record: bool = False, module: Optional[Any] = None) -> None:
        self._record: bool = record
        self._entered: bool = False
        raise NotImplementedError("with/as not well supported in transcrypt")

def setWarningOptions(opts: List[str]) -> None:
    _processoptions(opts)

class _OptionError(Exception):
    pass

def _processoptions(args: List[str]) -> None:
    for arg in args:
        try:
            _setoption(arg)
        except _OptionError as msg:
            console.log("WARNING: Invalid -W option ignored: {}".format(msg))

def _setoption(arg: str) -> None:
    parts: List[str] = arg.split(':')
    if len(parts) > 5:
        raise _OptionError("too many fields (max 5): {}".format(arg))
    while len(parts) < 5:
        parts.append('')
    action_str, message_str, category_str, module_str, lineno_str = [s.strip() for s in parts]
    action_final: str = _getaction(action_str)
    message_escaped: str = re.escape(message_str)
    category_final: Type[Warning] = _getcategory(category_str)
    module_escaped: str = re.escape(module_str)
    if module_escaped:
        module_escaped = module_escaped + '$'
    if lineno_str:
        try:
            lineno_int = int(lineno_str)
            if lineno_int < 0:
                raise ValueError
        except (ValueError, OverflowError):
            raise _OptionError("invalid lineno {}".format(lineno_str))
    else:
        lineno_int = 0
    filterwarnings(action_final, message_escaped, category_final, module_escaped, lineno_int)

def _getaction(action: str) -> str:
    if not action:
        return Actions.defaultact
    if action == "all":
        return Actions.always
    for a in ActionSet:
        if a.startswith(action):
            return a
    raise _OptionError("invalid action: {}".format(action))

def _getcategory(category: str) -> Type[Warning]:
    if not category:
        return Warning
    if category in CategoryMap.keys():
        try:
            cat: Type[Warning] = CategoryMap[category]
        except NameError:
            raise _OptionError("unknown warning category: {}".format(category))
    else:
        raise Exception("Unable to import category: {}, use `addWarningCategory`".format(category))
    return cat

if not _warnings_defaults:
    silence: List[Type[Warning]] = [DeprecationWarning]
    for cls in silence:
        simplefilter(Actions.ignore, category=cls)

__warningregistry__: Dict[Any, Any] = {}