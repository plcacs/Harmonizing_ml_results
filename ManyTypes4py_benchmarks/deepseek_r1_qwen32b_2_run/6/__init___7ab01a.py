from org.transcrypt.stubs.browser import __pragma__
from re.translate import translate
from typing import Optional, Union, Tuple, List, Dict, Iterator

T = 1 << 0
TEMPLATE = T
I = 1 << 1
IGNORECASE = I
L = 1 << 2
LOCALE = L
M = 1 << 3
MULTILINE = M
S = 1 << 4
DOTALL = S
U = 1 << 5
UNICODE = U
X = 1 << 6
VERBOSE = X
DEBUG = 1 << 7
A = 1 << 8
ASCII = A
Y = 1 << 16
STICKY = Y
G = 1 << 17
GLOBAL = G
J = 1 << 19
JSSTRICT = J

class error(Exception):
    def __init__(self, msg: str, error: Exception, pattern: Optional[str] = None, flags: int = 0, pos: Optional[int] = None) -> None:
        super().__init__(msg, error=error)
        self.pattern = pattern
        self.flags = flags
        self.pos = pos

class ReIndexError(IndexError):
    def __init__(self) -> None:
        super().__init__('no such group')

class Match(object):
    def __init__(self, mObj: List[Optional[str]], string: str, pos: int, endpos: int, rObj: 'Regex', namedGroups: Optional[Dict[str, int]] = None) -> None:
        for index, match in enumerate(mObj):
            mObj[index] = None if mObj[index] == js_undefined else mObj[index]
        self._obj = mObj
        self._pos = pos
        self._endpos = endpos
        self._re = rObj
        self._string = string
        self._namedGroups = namedGroups
        self._lastindex = self._lastMatchGroup()
        if self._namedGroups is not None:
            self._lastgroup = self._namedGroups[self._lastindex]
        else:
            self._lastgroup = None

    def _getPos(self) -> int:
        return self._pos

    def _setPos(self, val: int) -> None:
        raise AttributeError('readonly attribute')
    pos = property(_getPos, _setPos)

    def _getEndPos(self) -> int:
        return self._endpos

    def _setEndPos(self, val: int) -> None:
        raise AttributeError('readonly attribute')
    endpos = property(_getEndPos, _setEndPos)

    def _getRe(self) -> 'Regex':
        return self._re

    def _setRe(self, val: 'Regex') -> None:
        raise AttributeError('readonly attribute')
    re = property(_getRe, _setRe)

    def _getString(self) -> str:
        return self._string

    def _setString(self, val: str) -> None:
        raise AttributeError('readonly attribute')
    string = property(_getString, _setString)

    def _getLastGroup(self) -> Optional[str]:
        return self._lastgroup

    def _setLastGroup(self, val: str) -> None:
        raise AttributeError('readonly attribute')
    lastgroup = property(_getLastGroup, _setLastGroup)

    def _getLastIndex(self) -> Optional[int]:
        return self._lastindex

    def _setLastIndex(self, val: int) -> None:
        raise AttributeError('readonly attribute')
    lastindex = property(_getLastIndex, _setLastIndex)

    def _lastMatchGroup(self) -> Optional[int]:
        if len(self._obj) > 1:
            for i in range(len(self._obj) - 1, 0, -1):
                if self._obj[i] is not None:
                    return i
            return None
        else:
            return None

    def expand(self, template: str) -> None:
        raise NotImplementedError()

    def group(self, *args: Union[int, str]) -> Union[str, Tuple[str, ...]]:
        ret = []
        if len(args) > 0:
            for index in args:
                if isinstance(index, str):
                    if self._namedGroups is not None:
                        if index not in self._namedGroups.keys():
                            raise ReIndexError()
                        ret.append(self._obj[self._namedGroups[index]])
                    else:
                        raise NotImplementedError('No NamedGroups Available')
                else:
                    if index >= len(self._obj):
                        raise ReIndexError()
                    ret.append(self._obj[index])
        else:
            ret.append(self._obj[0])
        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)

    def groups(self, default: Optional[str] = None) -> Tuple[Optional[str], ...]:
        if len(self._obj) > 1:
            ret = self._obj[1:]
            return tuple([x if x is not None else default for x in ret])
        else:
            return tuple()

    def groupdict(self, default: Optional[str] = None) -> Dict[str, Optional[str]]:
        if self._namedGroups is not None:
            ret = {}
            for gName, gId in self._namedGroups.items():
                value = self._obj[gId]
                ret[gName] = value if value is not None else default
            return ret
        else:
            raise NotImplementedError('No NamedGroups Available')

    def start(self, group: Union[int, str] = 0) -> int:
        gId = 0
        if isinstance(group, str):
            if self._namedGroups is not None:
                if group not in self._namedGroups.keys():
                    raise ReIndexError()
                gId = self._namedGroups[group]
            else:
                raise NotImplementedError('No NamedGroups Available')
        else:
            gId = group
        if gId >= len(self._obj):
            raise ReIndexError()
        if gId == 0:
            return self._obj.index
        elif self._obj[gId] is not None:
            r = compile(escape(self._obj[gId]), self._re.flags)
            m = r.search(self._obj[0])
            if m:
                return self._obj.index + m.start()
            else:
                raise Exception('Failed to find capture group')
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
                raise NotImplementedError('No NamedGroups Available')
        else:
            gId = group
        if gId >= len(self._obj):
            raise ReIndexError()
        if gId == 0:
            return self._obj.index + len(self._obj[0])
        elif self._obj[gId] is not None:
            r = compile(escape(self._obj[gId]), self._re.flags)
            m = r.search(self._obj[0])
            if m:
                return self._obj.index + m.end()
            else:
                raise Exception('Failed to find capture group')
        else:
            return -1

    def span(self, group: Union[int, str] = 0) -> Tuple[int, int]:
        return (self.start(group), self.end(group))

class Regex(object):
    def __init__(self, pattern: str, flags: int) -> None:
        if not flags & ASCII > 0:
            flags |= UNICODE
        self._flags = flags
        self._jsFlags, self._obj = self._compileWrapper(pattern, flags)
        self._jspattern = pattern
        self._pypattern = pattern
        _, groupCounterRegex = self._compileWrapper(pattern + '|', flags)
        self._groups = groupCounterRegex.exec('').length - 1
        self._groupindex = None

    def _getPattern(self) -> str:
        ret = self._pypattern.replace('\\', '\\\\')
        return ret

    def _setPattern(self, val: str) -> None:
        raise AttributeError('readonly attribute')
    pattern = property(_getPattern, _setPattern)

    def _getFlags(self) -> int:
        return self._flags

    def _setFlags(self, val: int) -> None:
        raise AttributeError('readonly attribute')
    flags = property(_getFlags, _setFlags)

    def _getGroups(self) -> int:
        return self._groups

    def _setGroups(self, val: int) -> None:
        raise AttributeError('readonly attribute')
    groups = property(_getGroups, _setGroups)

    def _getGroupIndex(self) -> Optional[Dict[str, int]]:
        if self._groupindex is None:
            return {}
        else:
            return self._groupindex

    def _setGroupIndex(self, val: Dict[str, int]) -> None:
        raise AttributeError('readonly attribute')
    groupindex = property(_getGroupIndex, _setGroupIndex)

    def _compileWrapper(self, pattern: str, flags: int = 0) -> Tuple[str, object]:
        jsFlags = self._convertFlags(flags)
        rObj = None
        errObj = None
        __pragma__('js', '{}', '\n                   try {\n                     rObj = new RegExp(pattern, jsFlags)\n                   } catch( err ) {\n                     errObj = err\n                   }\n                   ')
        if errObj is not None:
            raise error(errObj.message, errObj, pattern, flags)
        return (jsFlags, rObj)

    def _convertFlags(self, flags: int) -> str:
        bitmaps = [(DEBUG, ''), (IGNORECASE, 'i'), (MULTILINE, 'm'), (STICKY, 'y'), (GLOBAL, 'g'), (UNICODE, 'u')]
        ret = ''.join([x[1] for x in bitmaps if x[0] & flags > 0])
        return ret

    def _getTargetStr(self, string: str, pos: int, endpos: int) -> str:
        endPtr = len(string)
        if endpos < endPtr:
            endPtr = endpos
        if endPtr < 0:
            endPtr = 0
        ret = string[pos:endPtr]
        return ret

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

    def fullmatch(self, string: str, pos: int = 0, endpos: Optional[int] = None) -> Optional['Match']:
        target = string
        strEndPos = len(string)
        if endpos is not None:
            target = target[:endpos]
            strEndPos = endpos
        rObj = self._obj
        m = rObj.exec(target)
        if m:
            obsEndPos = m.index + len(m[0])
            if m.index == pos and obsEndPos == strEndPos:
                return Match(m, string, pos, strEndPos, self, self._groupindex)
            else:
                return None
        else:
            return None

    def split(self, string: str, maxsplit: int = 0) -> List[str]:
        if maxsplit < 0:
            return [string]
        mObj = None
        rObj = self._obj
        if maxsplit == 0:
            mObj = string.split(rObj)
            return mObj
        else:
            flags = self._flags
            flags |= GLOBAL
            _, rObj = self._compileWrapper(self._jspattern, flags)
            ret = []
            lastM = None
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

    def _findAllMatches(self, string: str, pos: int = 0, endpos: Optional[int] = None) -> List[List[Optional[str]]]:
        target = self._getTargetStr(string, pos, endpos)
        flags = self._flags
        flags |= GLOBAL
        _, rObj = self._compileWrapper(self._jspattern, flags)
        ret = []
        while True:
            m = rObj.exec(target)
            if m:
                ret.append(m)
            else:
                break
        return ret

    def findall(self, string: str, pos: int = 0, endpos: Optional[int] = None) -> Union[List[str], List[Tuple[str, ...]]]:
        mlist = self._findAllMatches(string, pos, endpos)

        def mSelect(m: List[Optional[str]]) -> Union[str, Tuple[str, ...]]:
            if len(m) > 2:
                return tuple(m[1:])
            elif len(m) == 2:
                return m[1]
            else:
                return m[0]
        ret = list(map(mSelect, mlist))
        return ret

    def finditer(self, string: str, pos: int, endpos: Optional[int] = None) -> Iterator['Match']:
        mlist = self._findAllMatches(string, pos, endpos)
        ret = map(lambda m: Match(m, string, 0, len(string), self, self._groupindex), mlist)
        return iter(ret)

    def sub(self, repl: Union[str, callable], string: str, count: int = 0) -> str:
        ret, _ = self.subn(repl, string, count)
        return ret

    def subn(self, repl: Union[str, callable], string: str, count: int = 0) -> Tuple[str, int]:
        flags = self._flags
        flags |= GLOBAL
        _, rObj = self._compileWrapper(self._jspattern, flags)
        ret = ''
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
            elif lastEnd < 0:
                return (string, 0)
            else:
                ret += string[lastEnd:]
                return (ret, totalMatch)

class PyRegExp(Regex):
    def __init__(self, pyPattern: str, flags: int) -> None:
        jsTokens, inlineFlags, namedGroups, nCapGroups, n_splits = translate(pyPattern)
        flags |= inlineFlags
        jsPattern = ''.join(jsTokens)
        super().__init__(jsPattern, flags)
        self._pypattern = pyPattern
        self._nsplits = n_splits
        self._jsTokens = jsTokens
        self._capgroups = nCapGroups
        self._groupindex = namedGroups

def compile(pattern: str, flags: int = 0) -> 'Regex':
    if flags & JSSTRICT:
        p = Regex(pattern, flags)
    else:
        p = PyRegExp(pattern, flags)
    return p

def search(pattern: str, string: str, flags: int = 0) -> Optional['Match']:
    p = compile(pattern, flags)
    return p.search(string)

def match(pattern: str, string: str, flags: int = 0) -> Optional['Match']:
    p = compile(pattern, flags)
    return p.match(string)

def fullmatch(pattern: str, string: str, flags: int = 0) -> Optional['Match']:
    p = compile(pattern, flags)
    return p.fullmatch(string)

def split(pattern: str, string: str, maxsplit: int = 0, flags: int = 0) -> List[str]:
    p = compile(pattern, flags)
    return p.split(string, maxsplit)

def findall(pattern: str, string: str, flags: int = 0) -> Union[List[str], List[Tuple[str, ...]]]:
    p = compile(pattern, flags)
    return p.findall(string)

def finditer(pattern: str, string: str, flags: int = 0) -> Iterator['Match']:
    p = compile(pattern, flags)
    return p.finditer(string)

def sub(pattern: str, repl: Union[str, callable], string: str, count: int = 0, flags: int = 0) -> str:
    p = compile(pattern, flags)
    return p.sub(repl, string, count)

def subn(pattern: str, repl: Union[str, callable], string: str, count: int = 0, flags: int = 0) -> Tuple[str, int]:
    p = compile(pattern, flags)
    return p.subn(repl, string, count)

def escape(string: str) -> str:
    ret = None

    def replfunc(m: Optional[Match]) -> str:
        if m[0] == '\\':
            return '\\\\\\\\'
        else:
            return '\\\\' + m[0]
    __pragma__('js', '{}', '\n        var r = /[^A-Za-z:;\\d]/g;\n        ret = string.replace(r, replfunc);\n        ')
    if ret is not None:
        return ret
    else:
        raise Exception('Failed to escape the passed string')

def purge() -> None:
    pass