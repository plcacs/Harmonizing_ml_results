from org.transcrypt.stubs.browser import __pragma__
from re.translate import translate
from typing import Any, Callable, Dict, Iterable, List, Optional, Pattern, Tuple, Union, Iterator, Match as TypingMatch
import sys

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
    """ Regular Expression Exception Class
    """
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
        IndexError.__init__(self, 'no such group')

class Match(object):
    """ Resulting Match from a Regex Operation
    """
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
        if self._namedGroups is not None and self._lastindex is not None:
            self._lastgroup: Optional[str] = self._namedGroups.get(self._lastindex, None)
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

    def _setLastGroup(self, val: Optional[str]) -> None:
        raise AttributeError('readonly attribute')
    lastgroup = property(_getLastGroup, _setLastGroup)

    def _getLastIndex(self) -> Optional[int]:
        return self._lastindex

    def _setLastIndex(self, val: Optional[int]) -> None:
        raise AttributeError('readonly attribute')
    lastindex = property(_getLastIndex, _setLastIndex)

    def _lastMatchGroup(self) -> Optional[int]:
        """ Determine the last matching group in the object
        """
        if len(self._obj) > 1:
            for i in range(len(self._obj) - 1, 0, -1):
                if self._obj[i] is not None:
                    return i
            return None
        else:
            return None

    def expand(self, template: str) -> str:
        """
        """
        raise NotImplementedError()

    def group(self, *args: Union[int, str]) -> Union[str, Tuple[str, ...]]:
        """ Return the string[s] for a group[s]
        if only one group is provided, a string is returned
        if multiple groups are provided, a tuple of strings is returned
        """
        ret: List[str] = []
        if len(args) > 0:
            for index in args:
                if isinstance(index, str):
                    if self._namedGroups is not None:
                        if index not in self._namedGroups.keys():
                            raise ReIndexError()
                        ret.append(str(self._obj[self._namedGroups[index]]))
                    else:
                        raise NotImplementedError('No NamedGroups Available')
                else:
                    if index >= len(self._obj):
                        raise ReIndexError()
                    ret.append(str(self._obj[index]))
        else:
            ret.append(str(self._obj[0]))
        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)

    def groups(self, default: Any = None) -> Tuple[Any, ...]:
        """ Get all the groups in this match. Replace any
        groups that did not contribute to the match with default
        value.
        """
        if len(self._obj) > 1:
            ret = self._obj[1:]
            return tuple([x if x is not None else default for x in ret])
        else:
            return tuple()

    def groupdict(self, default: Any = None) -> Dict[str, Any]:
        """ The concept of named captures doesn't exist
        in javascript so this will likely never be implemented.
        For the python translated re we will have a group dict where
        possible.
        """
        if self._namedGroups is not None:
            ret: Dict[str, Any] = {}
            for gName, gId in self._namedGroups.items():
                value = self._obj[gId]
                ret[gName] = value if value is not None else default
            return ret
        else:
            raise NotImplementedError('No NamedGroups Available')

    def start(self, group: Union[int, str] = 0) -> int:
        """ Find the starting index in the string for the passed
        group id or named group string.
        @param group
          if the type of group is a str, then the named groups dict
            is searched for a matching string.
          if the type of group is an int, then the groups are
            indexed starting with 0 = entire match, and 1,... are
            the indices of the matched sub-groups
        """
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
            r = compile(escape(str(self._obj[gId])), self._re.flags)
            m = r.search(self._obj[0])
            if m:
                return self._obj.index + m.start()
            else:
                raise Exception('Failed to find capture group')
        else:
            return -1

    def end(self, group: Union[int, str] = 0) -> int:
        """ Find the ending index in the string for the passed
        group id or named group string.
        @param group
          if the type of group is a str, then the named groups dict
            is searched for a matching string.
          if the type of group is an int, then the groups are
            indexed starting with 0 = entire match, and 1,... are
            the indices of the matched sub-groups
        """
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
            return self._obj.index + len(str(self._obj[0]))
        elif self._obj[gId] is not None:
            r = compile(escape(str(self._obj[gId])), self._re.flags)
            m = r.search(self._obj[0])
            if m:
                return self._obj.index + m.end()
            else:
                raise Exception('Failed to find capture group')
        else:
            return -1

    def span(self, group: Union[int, str] = 0) -> Tuple[int, int]:
        """ Find the start and end index in the string for the passed
        group id or named group string.
        @param group
          if the type of group is a str, then the named groups dict
            is searched for a matching string.
          if the type of group is an int, then the groups are
            indexed starting with 0 = entire match, and 1,... are
            the indices of the matched sub-groups
        @return tuple of (start, end)
        """
        return (self.start(group), self.end(group))

class Regex(object):
    """ Regular Expression Object
    """
    def __init__(self, pattern: str, flags: int) -> None:
        """ Initial the Regex Object
        @param pattern - javascript regular expression pattern as a string
        @param flags - string of javascript flags for the subsequently
           created RegExp object.
        """
        if not flags & ASCII > 0:
            flags |= UNICODE
        self._flags: int = flags
        self._jsFlags: str
        self._obj: Any
        self._jsFlags, self._obj = self._compileWrapper(pattern, flags)
        self._jspattern: str = pattern
        self._pypattern: str = pattern
        _, groupCounterRegex = self._compileWrapper(pattern + '|', flags)
        self._groups: int = groupCounterRegex.exec('').length - 1
        self._groupindex: Optional[Dict[str, int]] = None

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

    def _getGroupIndex(self) -> Dict[str, int]:
        if self._groupindex is None:
            return {}
        else:
            return self._groupindex

    def _setGroupIndex(self, val: Dict[str, int]) -> None:
        raise AttributeError('readonly attribute')
    groupindex = property(_getGroupIndex, _setGroupIndex)

    def _compileWrapper(self, pattern: str, flags: int = 0) -> Tuple[str, Any]:
        """ This function wraps the creation of the the
        regular expresion so that we can catch the
        Syntax Error exception and turn it into a
        Python Exception
        """
        jsFlags = self._convertFlags(flags)
        rObj = None
        errObj = None
        __pragma__('js', '{}', '\n                   try {\n                     rObj = new RegExp(pattern, jsFlags)\n                   } catch( err ) {\n                     errObj = err\n                   }\n                   ')
        if errObj is not None:
            raise error(errObj.message, errObj, pattern, flags)
        return (jsFlags, rObj)

    def _convertFlags(self, flags: int) -> str:
        """ Convert the Integer map based flags to a
        string list of flags for js
        """
        bitmaps = [(DEBUG, ''), (IGNORECASE, 'i'), (MULTILINE, 'm'), (STICKY, 'y'), (GLOBAL, 'g'), (UNICODE, 'u')]
        ret = ''.join([x[1] for x in bitmaps if x[0] & flags > 0])
        return ret

    def _getTargetStr(self, string: str, pos: int, endpos: Optional[int]) -> str:
        """ Given an start and endpos, slice out a target string.
        """
        endPtr = len(string)
        if endpos is not None:
            if endpos < endPtr:
                endPtr = endpos
        if endPtr < 0:
            endPtr = 0
        ret = string[pos:endPtr]
        return ret

    def _patternHasCaptures(self) -> bool:
        """ Check if the regex pattern contains a capture
        necessary to make split behave correctly
        """
        return self._groups > 0

    def search(self, string: str, pos: int = 0, endpos: Optional[int] = None) -> Optional[Match]:
        """ Search through a string for matches to
        this regex object. @see the python docs
        """
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
        """ Match this regex at the beginning of the passed
        string. @see the python docs
        """
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
        """ Match the entirety of the passed string to this regex
        object. @see the python docs
        """
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
        """ Split the passed string on each match of this regex
        object. If the regex contains captures, then the match
        content is included as a separate item. If no captures are
        in the regex, then only the non-matching split content is
        returned. @see the python docs
        @param maxsplit max number of times to split the string
          at a matching substring.
        @return list of sub-strings
        """
        if maxsplit < 0:
            return [string]
        mObj: List[str] = None
        rObj = self._obj
        if maxsplit == 0:
            mObj = string.split(rObj)
            return mObj
        else:
            flags = self._flags
            flags |= GLOBAL
            _, rObj = self._compileWrapper(self._jspattern, flags)
            ret: List[str] = []
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
        """ Find All the