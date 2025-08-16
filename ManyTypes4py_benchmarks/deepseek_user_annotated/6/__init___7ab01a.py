from org.transcrypt.stubs.browser import __pragma__
from re.translate import translate
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union, Iterator, Match as MatchType, Pattern as PatternType

# Flags
T = (1 << 0)
TEMPLATE = T

I = (1 << 1)
IGNORECASE = I

# Deprecated
L = (1 << 2)
LOCALE = L

M = (1 << 3)
MULTILINE = M

S = (1 << 4)
DOTALL = S
# Legacy - Unicode by default in Python 3
U = (1 << 5)
UNICODE = U
X = (1 << 6)
VERBOSE = X
DEBUG = (1 << 7)

A = (1 << 8)
ASCII = A

# This is a javascript specific flag
Y = (1 << 16)
STICKY = Y
G = (1 << 17)
GLOBAL = G
# This flag is used to indicate that re module should use
# the javascript regex engine directly and not attempt to
# translate the regex string into a python regex
J = (1 << 19)
JSSTRICT = J

class error(Exception):
    """ Regular Expression Exception Class """
    def __init__(self, msg: str, error: Exception, pattern: Optional[str] = None, flags: int = 0, pos: Optional[int] = None) -> None:
        Exception.__init__(self, msg, error=error)
        self.pattern: Optional[str] = pattern
        self.flags: int = flags
        self.pos: Optional[int] = pos

class ReIndexError(IndexError):
    """ Index Error variant for the re module - primarily used for
    the group method in the Match Object.
    """
    def __init__(self) -> None:
        IndexError.__init__(self, "no such group")

class Match:
    """ Resulting Match from a Regex Operation """
    def __init__(self, mObj: List[Optional[str]], string: str, pos: int, endpos: int, rObj: 'Regex', namedGroups: Optional[Dict[str, int]] = None) -> None:
        for index, match in enumerate(mObj):
            mObj[index] = None if mObj[index] == js_undefined else mObj[index]
        self._obj: List[Optional[str]] = mObj
        self._pos: int = pos
        self._endpos: int = endpos
        self._re: 'Regex' = rObj
        self._string: str = string
        self._namedGroups: Optional[Dict[str, int]] = namedGroups
        self._lastindex: Optional[int] = self._lastMatchGroup()
        self._lastgroup: Optional[str] = self._namedGroups[self._lastindex] if (self._namedGroups is not None and self._lastindex is not None) else None

    # Read-only Properties
    @property
    def pos(self) -> int:
        return self._pos

    @pos.setter
    def pos(self, val: Any) -> None:
        raise AttributeError("readonly attribute")

    @property
    def endpos(self) -> int:
        return self._endpos

    @endpos.setter
    def endpos(self, val: Any) -> None:
        raise AttributeError("readonly attribute")

    @property
    def re(self) -> 'Regex':
        return self._re

    @re.setter
    def re(self, val: Any) -> None:
        raise AttributeError("readonly attribute")

    @property
    def string(self) -> str:
        return self._string

    @string.setter
    def string(self, val: Any) -> None:
        raise AttributeError("readonly attribute")

    @property
    def lastgroup(self) -> Optional[str]:
        return self._lastgroup

    @lastgroup.setter
    def lastgroup(self, val: Any) -> None:
        raise AttributeError("readonly attribute")

    @property
    def lastindex(self) -> Optional[int]:
        return self._lastindex

    @lastindex.setter
    def lastindex(self, val: Any) -> None:
        raise AttributeError("readonly attribute")

    def _lastMatchGroup(self) -> Optional[int]:
        if len(self._obj) > 1:
            for i in range(len(self._obj)-1, 0, -1):
                if self._obj[i] is not None:
                    return i
            return None
        else:
            return None

    def expand(self, template: str) -> str:
        raise NotImplementedError()

    def group(self, *args: Union[int, str]) -> Union[str, Tuple[str, ...]]:
        ret: List[str] = []
        if len(args) > 0:
            for index in args:
                if isinstance(index, str):
                    if self._namedGroups is not None:
                        if index not in self._namedGroups.keys():
                            raise ReIndexError()
                        ret.append(self._obj[self._namedGroups[index]])
                    else:
                        raise NotImplementedError("No NamedGroups Available")
                else:
                    if index >= len(self._obj):
                        raise ReIndexError()
                    val = self._obj[index]
                    if val is None:
                        raise ReIndexError()
                    ret.append(val)
        else:
            val = self._obj[0]
            if val is None:
                raise ReIndexError()
            ret.append(val)

        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)

    def groups(self, default: Any = None) -> Tuple[Any, ...]:
        if len(self._obj) > 1:
            return tuple(x if x is not None else default for x in self._obj[1:])
        else:
            return tuple()

    def groupdict(self, default: Any = None) -> Dict[str, Any]:
        if self._namedGroups is not None:
            ret: Dict[str, Any] = {}
            for gName, gId in self._namedGroups.items():
                value = self._obj[gId]
                ret[gName] = value if value is not None else default
            return ret
        else:
            raise NotImplementedError("No NamedGroups Available")

    def start(self, group: Union[int, str] = 0) -> int:
        gId = 0
        if isinstance(group, str):
            if self._namedGroups is not None:
                if group not in self._namedGroups.keys():
                    raise ReIndexError()
                gId = self._namedGroups[group]
            else:
                raise NotImplementedError("No NamedGroups Available")
        else:
            gId = group

        if gId >= len(self._obj):
            raise ReIndexError()

        if gId == 0:
            return self._obj.index
        else:
            if self._obj[gId] is not None:
                r = compile(escape(self._obj[gId]), self._re.flags)
                m = r.search(self._obj[0])
                if m:
                    return self._obj.index + m.start()
                else:
                    raise Exception("Failed to find capture group")
            else:
                return -1

    def end(self, group: Union[int, str] = 0) -> int:
        gId = 0
        if isinstance(group, str):
            if self._namedGroups is not None:
                if group not in self._namedGroups.keys():
                    raise ReIndexError()
                gId = self._namedGroups[group]
            else:
                raise NotImplementedError("No NamedGroups Available")
        else:
            gId = group

        if gId >= len(self._obj):
            raise ReIndexError()

        if gId == 0:
            return self._obj.index + len(self._obj[0])
        else:
            if self._obj[gId] is not None:
                r = compile(escape(self._obj[gId]), self._re.flags)
                m = r.search(self._obj[0])
                if m:
                    return self._obj.index + m.end()
                else:
                    raise Exception("Failed to find capture group")
            else:
                return -1

    def span(self, group: Union[int, str] = 0) -> Tuple[int, int]:
        return (self.start(group), self.end(group))

class Regex:
    """ Regular Expression Object """
    def __init__(self, pattern: str, flags: int) -> None:
        if not ((flags & ASCII) > 0):
            flags |= UNICODE

        self._flags: int = flags
        self._jsFlags: str
        self._obj: Any
        self._jsFlags, self._obj = self._compileWrapper(pattern, flags)
        self._jspattern: str = pattern
        self._pypattern: str = pattern

        _, groupCounterRegex = self._compileWrapper(pattern + '|', flags)
        self._groups: int = groupCounterRegex.exec('').length-1
        self._groupindex: Optional[Dict[str, int]] = None

    # Read-only Properties
    @property
    def pattern(self) -> str:
        return self._pypattern.replace('\\', '\\\\')

    @pattern.setter
    def pattern(self, val: Any) -> None:
        raise AttributeError("readonly attribute")

    @property
    def flags(self) -> int:
        return self._flags

    @flags.setter
    def flags(self, val: Any) -> None:
        raise AttributeError("readonly attribute")

    @property
    def groups(self) -> int:
        return self._groups

    @groups.setter
    def groups(self, val: Any) -> None:
        raise AttributeError("readonly attribute")

    @property
    def groupindex(self) -> Dict[str, int]:
        return self._groupindex if self._groupindex is not None else {}

    @groupindex.setter
    def groupindex(self, val: Any) -> None:
        raise AttributeError("readonly attribute")

    def _compileWrapper(self, pattern: str, flags: int = 0) -> Tuple[str, Any]:
        jsFlags = self._convertFlags(flags)

        rObj: Any = None
        errObj: Optional[Exception] = None
        __pragma__('js', '{}',
                   '''
                   try {
                     rObj = new RegExp(pattern, jsFlags)
                   } catch( err ) {
                     errObj = err
                   }
                   ''')

        if errObj is not None:
            raise error(errObj.message, errObj, pattern, flags)

        return (jsFlags, rObj)

    def _convertFlags(self, flags: int) -> str:
        bitmaps = [
            (DEBUG, ""),
            (IGNORECASE, "i"),
            (MULTILINE, "m"),
            (STICKY, "y"),
            (GLOBAL, "g"),
            (UNICODE, "u"),
        ]
        return "".join([x[1] for x in bitmaps if (x[0] & flags) > 0])

    def _getTargetStr(self, string: str, pos: int, endpos: Optional[int]) -> str:
        endPtr = len(string)
        if endpos is not None:
            if endpos < endPtr:
                endPtr = endpos
        if endPtr < 0:
            endPtr = 0
        return string[pos:endPtr]

    def _patternHasCaptures(self) -> bool:
        return self._groups > 0

    def search(self, string: str, pos: int = 0, endpos: Optional[int] = None) -> Optional[Match]:
        if endpos is None:
            endpos = len(string)
        rObj = self._obj
        m = rObj.exec(string)
        if m:
            if m.index < pos or m.index > endpos:
                return None
            else:
                return Match(m, string, pos, endpos, self, self._groupindex)
        else:
            return None

    def match(self, string: str, pos: int = 0, endpos: Optional[int] = None) -> Optional[Match]:
        target = string
        if endpos is not None:
            target = target[:endpos]
        else:
            endpos = len(string)

        rObj = self._obj
        m = rObj.exec(target)
        if m:
            if m.index == pos:
                return Match(m, string, pos, endpos, self, self._groupindex)
            else:
                return None
        else:
            return None

    def fullmatch(self, string: str, pos: int = 0, endpos: Optional[int] = None) -> Optional[Match]:
        target = string
        strEndPos = len(string)
        if endpos is not None:
            target = target[:endpos]
            strEndPos = endpos

        rObj = self._obj
        m = rObj.exec(target)
        if m:
            obsEndPos = (m.index + len(m[0]))
            if m.index == pos and obsEndPos == strEndPos:
                return Match(m, string, pos, strEndPos, self, self._groupindex)
            else:
                return None
        else:
            return None

    def split(self, string: str, maxsplit: int = 0) -> List[str]:
        if maxsplit < 0:
            return [string]

        mObj: Optional[List[str]] = None
        rObj = self._obj
        if maxsplit == 0:
            mObj = string.split(rObj)
            return mObj
        else:
            flags = self._flags
            flags |= GLOBAL

            _, rObj = self._compileWrapper(self._jspattern, flags)
            ret: List[str] = []
            lastM: Optional[Any] = None
            cnt = 0
            for i in range(0, maxsplit):
                m = rObj.exec(string)
                if m:
                    cnt += 1
                    if lastM is not None:
                        start = lastM.index + len(lastM[0])
                        head = string[start:m.index]
                        ret.append(head)
                        if len(m) > 1:
                            ret.extend(m[1:])
                    else:
                        head = string[:m.index]
                        ret.append(head)
                        if len(m) > 1:
                            ret.extend(m[1:])
                    lastM = m
                else:
                    break

            if lastM is not None:
                endPos = lastM.index + len(lastM[0])
                end = string[endPos:]
                ret.append(end)

            return ret

    def _findAllMatches(self, string: str, pos: int = 0, endpos: Optional[int] = None) -> List[Any]:
        target = self._getTargetStr(string, pos, endpos)

        flags = self._flags
        flags |= GLOBAL

        _, rObj = self._compileWrapper(self._jspattern, flags)
        ret: List[Any] = []
        while True:
            m = rObj.exec(target)
            if m:
                ret.append(m)
            else:
                break
        return ret

    def findall(self, string: str, pos: int = 0, endpos: Optional[int] = None) -> List[Union[str, Tuple[str, ...]]]:
        mlist = self._findAllMatches(string, pos, endpos)

        def mSelect(m: Any) -> Union[str, Tuple[str, ...]]:
            if len(m) > 2:
                return tuple(m[1:])
            elif len(m) == 2:
                return m[1]
            else:
                return m[0]

        return list(map(mSelect, mlist))

    def finditer(self, string: str, pos: int = 0, endpos: Optional[int] = None) -> Iterator[Match]:
        mlist = self._findAllMatches(string, pos, endpos)
        return iter(map(lambda m: Match(m, string, 0, len(string), self, self._groupindex), mlist))

    def sub(self, repl: Union[str, Callable[[Match], str]], string: str, count: int = 0) -> str:
        ret, _ = self.subn(repl, string, count)
        return ret

    def subn(self, repl: Union[str, Callable[[Match], str]], string: str, count: int = 0) -> Tuple[str, int]:
        flags = self._flags
        flags |= GLOBAL

        _, rObj = self._compileWrapper(self._jspattern, flags)
        ret = ""
        totalMatch = 0
        lastEnd = -1
        while True:
            if count > 0:
                if totalMatch >= count:
                    if lastEnd < 0:
                        return (ret, totalMatch)
                    else:
                        ret += string[lastEnd:m.index]
                        return (ret, totalMatch)

            m = rObj.exec(string)
            if m:
                if lastEnd < 0:
                    ret += string[:m.index]
                else:
                    ret += string[lastEnd:m.index]

                if callable(repl):
                    content = repl(Match(m, string, 0, len(string), self, self._groupindex))
                    ret += content
                else:
                    ret += repl

                totalMatch += 1
                lastEnd = m.index + len(m[0])
            else:
                if lastEnd < 0:
                    return (string, 0)
                else:
                    ret += string[lastEnd:]
                    return (ret, totalMatch)

class PyRegExp(Regex):
    """ Python Regular Expression object which translates a python
    regex syntax string into a format that can be passed to the
    js regex engine.
    """
    def __init__(self, pyPattern: str, flags: int) -> None:
        jsTokens, inlineFlags, namedGroups, nCapGroups, n_splits = translate(pyPattern)
        flags |= inlineFlags

        jsPattern = ''.join(jsTokens)
        Regex.__init__(self, jsPattern, flags)
        self._pypattern: str = pyPattern

        self._nsplits: int = n_splits
        self._jsTokens: List[str] = jsTokens
        self._capgroups: int = nCapGroups
        self._groupindex: Optional[Dict[str, int]] = namedGroups

def compile(pattern: str, flags: int = 0) -> Union[Regex, PyRegExp]:
    if flags & JSSTRICT:
        return Regex(pattern, flags)
    else:
        return PyRegExp(pattern, flags)

def search(pattern: str, string: