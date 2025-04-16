from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Iterator
from org.transcrypt.stubs.browser import __pragma__

# Flags
T: int = (1 << 0)
TEMPLATE: int = T

I: int = (1 << 1)
IGNORECASE: int = I

# Deprecated
L: int = (1 << 2)
LOCALE: int = L

M: int = (1 << 3)
MULTILINE: int = M

S: int = (1 << 4)
DOTALL: int = S
# Legacy - Unicode by default in Python 3
U: int = (1 << 5)
UNICODE: int = U
X: int = (1 << 6)
VERBOSE: int = X
DEBUG: int = (1 << 7)

A: int = (1 << 8)
ASCII: int = A

# This is a javascript specific flag
Y: int = (1 << 16)
STICKY: int = Y
G: int = (1 << 17)
GLOBAL: int = G
# This flag is used to indicate that re module should use
# the javascript regex engine directly and not attempt to
# translate the regex string into a python regex
J: int = (1 << 19)
JSSTRICT: int = J

class error(Exception):
    """ Regular Expression Exception Class """
    def __init__(self, msg: str, error: Any, pattern: Optional[str] = None, flags: int = 0, pos: Optional[int] = None) -> None:
        Exception.__init__(self, msg, error=error)
        self.pattern: Optional[str] = pattern
        self.flags: int = flags
        self.pos: Optional[int] = pos

class ReIndexError(IndexError):
    """ Index Error variant for the re module - primarily used for the group method in the Match Object. """
    def __init__(self) -> None:
        IndexError.__init__(self, "no such group")

class Match:
    """ Resulting Match from a Regex Operation """
    def __init__(self, mObj: List[Optional[str]], string: str, pos: int, endpos: int, rObj: 'Regex', namedGroups: Optional[Dict[str, int]] = None) -> None:
        self._obj: List[Optional[str]] = [None if x == js_undefined else x for x in mObj]
        self._pos: int = pos
        self._endpos: int = endpos
        self._re: 'Regex' = rObj
        self._string: str = string
        self._namedGroups: Optional[Dict[str, int]] = namedGroups
        self._lastindex: Optional[int] = self._lastMatchGroup()
        self._lastgroup: Optional[str] = self._namedGroups[self._lastindex] if self._namedGroups is not None else None

    # Read-only Properties
    @property
    def pos(self) -> int:
        return self._pos

    @property
    def endpos(self) -> int:
        return self._endpos

    @property
    def re(self) -> 'Regex':
        return self._re

    @property
    def string(self) -> str:
        return self._string

    @property
    def lastgroup(self) -> Optional[str]:
        return self._lastgroup

    @property
    def lastindex(self) -> Optional[int]:
        return self._lastindex

    def _lastMatchGroup(self) -> Optional[int]:
        if len(self._obj) > 1:
            for i in range(len(self._obj) - 1, 0, -1):
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
                    ret.append(self._obj[index])
        else:
            ret.append(self._obj[0])

        return ret[0] if len(ret) == 1 else tuple(ret)

    def groups(self, default: Optional[Any] = None) -> Tuple[Optional[Any], ...]:
        if len(self._obj) > 1:
            return tuple(x if x is not None else default for x in self._obj[1:])
        else:
            return tuple()

    def groupdict(self, default: Optional[Any] = None) -> Dict[str, Optional[Any]]:
        if self._namedGroups is not None:
            return {gName: self._obj[gId] if self._obj[gId] is not None else default for gName, gId in self._namedGroups.items()}
        else:
            raise NotImplementedError("No NamedGroups Available")

    def start(self, group: Union[int, str] = 0) -> int:
        gId: int = 0
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
        gId: int = 0
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
        self._jsFlags, self._obj = self._compileWrapper(pattern, flags)
        self._jspattern: str = pattern
        self._pypattern: str = pattern

        _, groupCounterRegex = self._compileWrapper(pattern + '|', flags)
        self._groups: int = groupCounterRegex.exec('').length - 1
        self._groupindex: Optional[Dict[str, int]] = None

    # Read-only Properties
    @property
    def pattern(self) -> str:
        return self._pypattern.replace('\\', '\\\\')

    @property
    def flags(self) -> int:
        return self._flags

    @property
    def groups(self) -> int:
        return self._groups

    @property
    def groupindex(self) -> Dict[str, int]:
        return self._groupindex if self._groupindex is not None else {}

    def _compileWrapper(self, pattern: str, flags: int = 0) -> Tuple[str, Any]:
        jsFlags: str = self._convertFlags(flags)
        rObj: Optional[Any] = None
        errObj: Optional[Any] = None

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

        return jsFlags, rObj

    def _convertFlags(self, flags: int) -> str:
        bitmaps: List[Tuple[int, str]] = [
            (DEBUG, ""),
            (IGNORECASE, "i"),
            (MULTILINE, "m"),
            (STICKY, "y"),
            (GLOBAL, "g"),
            (UNICODE, "u"),
        ]
        return "".join(x[1] for x in bitmaps if (x[0] & flags) > 0)

    def _getTargetStr(self, string: str, pos: int, endpos: Optional[int]) -> str:
        endPtr: int = len(string)
        if endpos is not None:
            if endpos < endPtr:
                endPtr = endpos
        if endPtr < 0:
            endPtr = 0
        return string[pos:endPtr]

    def _patternHasCaptures(self) -> bool:
        return self._groups > 0

    def search(self, string: str, pos: int = 0, endpos: Optional[int] = None) -> Optional['Match']:
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

    def match(self, string: str, pos: int = 0, endpos: Optional[int] = None) -> Optional['Match']:
        target: str = string
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

    def fullmatch(self, string: str, pos: int = 0, endpos: Optional[int] = None) -> Optional['Match']:
        target: str = string
        strEndPos: int = len(string)
        if endpos is not None:
            target = target[:endpos]
            strEndPos = endpos

        rObj = self._obj
        m = rObj.exec(target)
        if m:
            obsEndPos: int = m.index + len(m[0])
            if m.index == pos and obsEndPos == strEndPos:
                return Match(m, string, pos, strEndPos, self, self._groupindex)
            else:
                return None
        else:
            return None

    def split(self, string: str, maxsplit: int = 0) -> List[str]:
        if maxsplit < 0:
            return [string]

        rObj = self._obj
        if maxsplit == 0:
            return string.split(rObj)
        else:
            flags: int = self._flags | GLOBAL
            _, rObj = self._compileWrapper(self._jspattern, flags)
            ret: List[str] = []
            lastM: Optional[Any] = None
            cnt: int = 0
            for _ in range(maxsplit):
                m = rObj.exec(string)
                if m:
                    cnt += 1
                    if lastM is not None:
                        start: int = lastM.index + len(lastM[0])
                        head: str = string[start:m.index]
                        ret.append(head)
                        if len(m) > 1:
                            ret.extend(m[1:])
                    else:
                        head: str = string[:m.index]
                        ret.append(head)
                        if len(m) > 1:
                            ret.extend(m[1:])
                    lastM = m
                else:
                    break

            if lastM is not None:
                endPos: int = lastM.index + len(lastM[0])
                end: str = string[endPos:]
                ret.append(end)

            return ret

    def _findAllMatches(self, string: str, pos: int = 0, endpos: Optional[int] = None) -> List[Any]:
        target: str = self._getTargetStr(string, pos, endpos)
        flags: int = self._flags | GLOBAL
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
        mlist: List[Any] = self._findAllMatches(string, pos, endpos)

        def mSelect(m: Any) -> Union[str, Tuple[str, ...]]:
            if len(m) > 2:
                return tuple(m[1:])
            elif len(m) == 2:
                return m[1]
            else:
                return m[0]

        return list(map(mSelect, mlist))

    def finditer(self, string: str, pos: int, endpos: Optional[int] = None) -> Iterator['Match']:
        mlist: List[Any] = self._findAllMatches(string, pos, endpos)
        return iter(map(lambda m: Match(m, string, 0, len(string), self, self._groupindex), mlist))

    def sub(self, repl: Union[str, Callable[['Match'], str]], string: str, count: int = 0) -> str:
        ret, _ = self.subn(repl, string, count)
        return ret

    def subn(self, repl: Union[str, Callable[['Match'], str]], string: str, count: int = 0) -> Tuple[str, int]:
        flags: int = self._flags | GLOBAL
        _, rObj = self._compileWrapper(self._jspattern, flags)
        ret: str = ""
        totalMatch: int = 0
        lastEnd: int = -1
        while True:
            if count > 0:
                if totalMatch >= count:
                    if lastEnd < 0:
                        return ret, totalMatch
                    else:
                        ret += string[lastEnd:m.index]
                        return ret, totalMatch

            m = rObj.exec(string)
            if m:
                if lastEnd < 0:
                    ret += string[:m.index]
                else:
                    ret += string[lastEnd:m.index]

                if callable(repl):
                    content: str = repl(Match(m, string, 0, len(string), self, self._groupindex))
                    ret += content
                else:
                    ret += repl

                totalMatch += 1
                lastEnd = m.index + len(m[0])
            else:
                if lastEnd < 0:
                    return string, 0
                else:
                    ret += string[lastEnd:]
                    return ret, totalMatch

class PyRegExp(Regex):
    """ Python Regular Expression object which translates a python regex syntax string into a format that can be passed to the js regex engine. """
    def __init__(self, pyPattern: str, flags: int) -> None:
        jsTokens, inlineFlags, namedGroups, nCapGroups, n_splits = translate(pyPattern)
        flags |= inlineFlags
        jsPattern: str = ''.join(jsTokens)
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

def search(pattern: str, string: str, flags: int = 0) -> Optional[Match]:
    p = compile(pattern, flags)
    return p.search(string)

def match(pattern: str, string: str, flags: int = 0) -> Optional[Match]:
    p = compile(pattern, flags)
    return p.match(string)

def fullmatch(pattern: str, string: str, flags: int = 0) -> Optional[Match]:
    p = compile(pattern, flags)
    return p.fullmatch(string)

def split(pattern: str, string: str, maxsplit: int = 0, flags: int = 0) -> List[str]:
    p = compile(pattern, flags)
    return p.split(string, maxsplit)

def findall(pattern: str, string: str, flags: int = 0) -> List[Union[str, Tuple[str, ...]]]:
    p = compile(pattern, flags)
    return p.findall(string)

def finditer(pattern: str, string: str, flags: int = 0) -> Iterator[Match]:
    p = compile(pattern, flags)
    return p.finditer(string)

def sub(pattern: str, repl: Union[str, Callable[['Match'], str]], string: str, count: int = 0, flags: int = 0) -> str:
    p = compile(pattern, flags)
    return p.sub(repl, string, count)

def subn(pattern: str, repl: Union[str, Callable[['Match'], str]], string: str, count: int = 0, flags: int = 0) -> Tuple[str, int]:
    p = compile(pattern, flags)
    return p.subn(repl, string, count)

def escape(string: str) -> str:
    ret: Optional[str] = None

    def replfunc(m: Any) -> str:
        if m[0] == "\\":
            return "\\\\\\\\"
        else:
            return "\\\\" + m[0]

    __pragma__(
        'js', '{}',
        '''
        var r = /[^A-Za-z:;\d]/g;
        ret = string.replace(r, replfunc);
        ''')
    if ret is not None:
        return ret
    else:
        raise Exception("Failed to escape the passed string")

def purge() -> None:
    pass
