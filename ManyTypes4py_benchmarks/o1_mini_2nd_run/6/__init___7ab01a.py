from org.transcrypt.stubs.browser import __pragma__
from re.translate import translate
from typing import Any, Optional, Dict, List, Tuple, Callable, Iterator

T: int = 1 << 0
TEMPLATE: int = T
I: int = 1 << 1
IGNORECASE: int = I
L: int = 1 << 2
LOCALE: int = L
M: int = 1 << 3
MULTILINE: int = M
S: int = 1 << 4
DOTALL: int = S
U: int = 1 << 5
UNICODE: int = U
X: int = 1 << 6
VERBOSE: int = X
DEBUG: int = 1 << 7
A: int = 1 << 8
ASCII: int = A
Y: int = 1 << 16
STICKY: int = Y
G: int = 1 << 17
GLOBAL: int = G
J: int = 1 << 19
JSSTRICT: int = J


class error(Exception):
    """ Regular Expression Exception Class
    """

    def __init__(self, msg: str, error: Any, pattern: Optional[str] = None, flags: int = 0, pos: Optional[int] = None) -> None:
        """
        """
        super().__init__(msg, error=error)
        self.pattern: Optional[str] = pattern
        self.flags: int = flags
        self.pos: Optional[int] = pos


class ReIndexError(IndexError):
    """ Index Error variant for the re module - primarily used for
    the group method in the Match Object.
    """

    def __init__(self) -> None:
        super().__init__('no such group')


class Match:
    """ Resulting Match from a Regex Operation
    """

    def __init__(
        self,
        mObj: List[Any],
        string: str,
        pos: int,
        endpos: Optional[int],
        rObj: 'Regex',
        namedGroups: Optional[Dict[str, int]] = None
    ) -> None:
        """
        """
        for index, match in enumerate(mObj):
            mObj[index] = None if mObj[index] == js_undefined else mObj[index]
        self._obj: List[Any] = mObj
        self._pos: int = pos
        self._endpos: Optional[int] = endpos
        self._re: 'Regex' = rObj
        self._string: str = string
        self._namedGroups: Optional[Dict[str, int]] = namedGroups
        self._lastindex: Optional[int] = self._lastMatchGroup()
        if self._namedGroups is not None and self._lastindex is not None:
            self._lastgroup: Optional[str] = self._namedGroups.get(self._lastindex)
        else:
            self._lastgroup: Optional[str] = None

    def _getPos(self) -> int:
        return self._pos

    def _setPos(self, val: Any) -> None:
        raise AttributeError('readonly attribute')

    pos: property = property(_getPos, _setPos)

    def _getEndPos(self) -> Optional[int]:
        return self._endpos

    def _setEndPos(self, val: Any) -> None:
        raise AttributeError('readonly attribute')

    endpos: property = property(_getEndPos, _setEndPos)

    def _getRe(self) -> 'Regex':
        return self._re

    def _setRe(self, val: Any) -> None:
        raise AttributeError('readonly attribute')

    re: property = property(_getRe, _setRe)

    def _getString(self) -> str:
        return self._string

    def _setString(self, val: Any) -> None:
        raise AttributeError('readonly attribute')

    string: property = property(_getString, _setString)

    def _getLastGroup(self) -> Optional[str]:
        return self._lastgroup

    def _setLastGroup(self, val: Any) -> None:
        raise AttributeError('readonly attribute')

    lastgroup: property = property(_getLastGroup, _setLastGroup)

    def _getLastIndex(self) -> Optional[int]:
        return self._lastindex

    def _setLastIndex(self, val: Any) -> None:
        raise AttributeError('readonly attribute')

    lastindex: property = property(_getLastIndex, _setLastIndex)

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

    def group(self, *args: Any) -> Any:
        """ Return the string[s] for a group[s]
        if only one group is provided, a string is returned
        if multiple groups are provided, a tuple of strings is returned
        """
        ret: List[Any] = []
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

    def start(self, group: Any = 0) -> int:
        """ Find the starting index in the string for the passed
        group id or named group string.
        @param group
          if the type of group is a str, then the named groups dict
            is searched for a matching string.
          if the type of group is an int, then the groups are
            indexed starting with 0 = entire match, and 1,... are
            the indices of the matched sub-groups
        """
        gId: int = 0
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

    def end(self, group: Any = 0) -> int:
        """ Find the ending index in the string for the passed
        group id or named group string.
        @param group
          if the type of group is a str, then the named groups dict
            is searched for a matching string.
          if the type of group is an int, then the groups are
            indexed starting with 0 = entire match, and 1,... are
            the indices of the matched sub-groups
        """
        gId: int = 0
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

    def span(self, group: Any = 0) -> Tuple[int, int]:
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


class Regex:
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

    def _setPattern(self, val: Any) -> None:
        raise AttributeError('readonly attribute')

    pattern: property = property(_getPattern, _setPattern)

    def _getFlags(self) -> int:
        return self._flags

    def _setFlags(self, val: Any) -> None:
        raise AttributeError('readonly attribute')

    flags: property = property(_getFlags, _setFlags)

    def _getGroups(self) -> int:
        return self._groups

    def _setGroups(self, val: Any) -> None:
        raise AttributeError('readonly attribute')

    groups: property = property(_getGroups, _setGroups)

    def _getGroupIndex(self) -> Dict[str, int]:
        if self._groupindex is None:
            return {}
        else:
            return self._groupindex

    def _setGroupIndex(self, val: Any) -> None:
        raise AttributeError('readonly attribute')

    groupindex: property = property(_getGroupIndex, _setGroupIndex)

    def _compileWrapper(self, pattern: str, flags: int = 0) -> Tuple[str, Any]:
        """ This function wraps the creation of the the
        regular expresion so that we can catch the
        Syntax Error exception and turn it into a
        Python Exception
        """
        jsFlags = self._convertFlags(flags)
        rObj: Any = None
        errObj: Any = None
        __pragma__('js', '{}', '''
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
        """ Convert the Integer map based flags to a
        string list of flags for js
        """
        bitmaps: List[Tuple[int, str]] = [
            (DEBUG, ''),
            (IGNORECASE, 'i'),
            (MULTILINE, 'm'),
            (STICKY, 'y'),
            (GLOBAL, 'g'),
            (UNICODE, 'u')
        ]
        ret: str = ''.join([x[1] for x in bitmaps if x[0] & flags > 0])
        return ret

    def _getTargetStr(self, string: str, pos: int, endpos: Optional[int]) -> str:
        """ Given an start and endpos, slice out a target string.
        """
        endPtr: int = len(string)
        if endpos is not None:
            if endpos < endPtr:
                endPtr = endpos
        if endPtr < 0:
            endPtr = 0
        ret: str = string[pos:endPtr]
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
        m: Any = rObj.exec(string)
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
        target: str = string
        if endpos is not None:
            target = target[:endpos]
        else:
            endpos = len(string)
        rObj = self._obj
        m: Any = rObj.exec(target)
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
        target: str = string
        strEndPos: int = len(string)
        if endpos is not None:
            target = target[:endpos]
            strEndPos = endpos
        rObj = self._obj
        m: Any = rObj.exec(target)
        if m:
            obsEndPos: int = m.index + len(m[0])
            if m.index == pos and obsEndPos == strEndPos:
                return Match(m, string, pos, strEndPos, self, self._groupindex)
            else:
                return None
        else:
            return None

    def split(self, string: str, maxsplit: int = 0) -> List[Any]:
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
        mObj: Any = None
        rObj = self._obj
        if maxsplit == 0:
            mObj = string.split(rObj)
            return mObj
        else:
            flags = self._flags
            flags |= GLOBAL
            _, rObj = self._compileWrapper(self._jspattern, flags)
            ret: List[Any] = []
            lastM: Optional[Any] = None
            cnt: int = 0
            for i in range(0, maxsplit):
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
                        head = string[:m.index]
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
        flags: int = self._flags
        flags |= GLOBAL
        _, rObj = self._compileWrapper(self._jspattern, flags)
        ret: List[Any] = []
        while True:
            m: Any = rObj.exec(target)
            if m:
                ret.append(m)
            else:
                break
        return ret

    def findall(self, string: str, pos: int = 0, endpos: Optional[int] = None) -> List[Any]:
        """ Find All the matches to this regex in the passed string
        @return either:
          List of strings of the matched regex has 1 or 0 capture
            groups
          List of elements that are each a list of the groups matched
            at each location in the string.
        @see the python docs
        """
        mlist: List[Any] = self._findAllMatches(string, pos, endpos)

        def mSelect(m: Any) -> Any:
            if len(m) > 2:
                return tuple(m[1:])
            elif len(m) == 2:
                return m[1]
            else:
                return m[0]

        ret = list(map(mSelect, mlist))
        return ret

    def finditer(self, string: str, pos: int, endpos: Optional[int] = None) -> Iterator[Match]:
        """ Like findall but returns an iterator instead of
        a list.
        @see the python docs
        """
        mlist: List[Any] = self._findAllMatches(string, pos, endpos)
        ret = map(lambda m: Match(m, string, 0, len(string), self, self._groupindex), mlist)
        return iter(ret)

    def sub(self, repl: Any, string: str, count: int = 0) -> str:
        """ Substitute each match of this regex in the passed string
        with either:
          if repl is of type string,
             replace with repl
          if repl is a callable object, then the returned value
            from repl(m) where m is the match object at a particular
            point in the string.
        @see the python docs
        @return the augmented string with substitutions
        """
        ret, _ = self.subn(repl, string, count)
        return ret

    def subn(self, repl: Any, string: str, count: int = 0) -> Tuple[str, int]:
        """ Similar to sub except that instead of just returning the
        augmented string, it returns a tuple of the augmented string
        and the number of times that the replacement op occured.
        (augstr, numreplacements)
        @see the python docs
        """
        flags: int = self._flags
        flags |= GLOBAL
        _, rObj = self._compileWrapper(self._jspattern, flags)
        ret: str = ''
        totalMatch: int = 0
        lastEnd: int = -1
        while True:
            if count > 0:
                if totalMatch >= count:
                    if lastEnd < 0:
                        return (ret, totalMatch)
                    else:
                        ret += string[lastEnd:m.index]
                        return (ret, totalMatch)
            m: Any = rObj.exec(string)
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
    """ Python Regular Expression object which translates a python
    regex syntax string into a format that can be passed to the
    js regex engine.
    """

    def __init__(self, pyPattern: str, flags: int) -> None:
        """
        @pattern Python Regex String
        @pattern flags bit flags passed by the user.
        """
        jsTokens: List[str]
        inlineFlags: int
        namedGroups: Optional[Dict[str, int]]
        nCapGroups: int
        n_splits: int
        jsTokens, inlineFlags, namedGroups, nCapGroups, n_splits = translate(pyPattern)
        flags |= inlineFlags
        jsPattern: str = ''.join(jsTokens)
        super().__init__(jsPattern, flags)
        self._pypattern: str = pyPattern
        self._nsplits: int = n_splits
        self._jsTokens: List[str] = jsTokens
        self._capgroups: int = nCapGroups
        self._groupindex: Optional[Dict[str, int]] = namedGroups


def compile(pattern: str, flags: int = 0) -> Regex:
    """ Compile a regex object and return
    an object that can be used for further processing.
    """
    if flags & JSSTRICT:
        p: Regex = Regex(pattern, flags)
    else:
        p = PyRegExp(pattern, flags)
    return p


def search(pattern: str, string: str, flags: int = 0) -> Optional[Match]:
    """ Search a string for a particular matching pattern
    """
    p: Regex = compile(pattern, flags)
    return p.search(string)


def match(pattern: str, string: str, flags: int = 0) -> Optional[Match]:
    """ Match a string for a particular pattern
    """
    p: Regex = compile(pattern, flags)
    return p.match(string)


def fullmatch(pattern: str, string: str, flags: int = 0) -> Optional[Match]:
    """
    """
    p: Regex = compile(pattern, flags)
    return p.fullmatch(string)


def split(pattern: str, string: str, maxsplit: int = 0, flags: int = 0) -> List[Any]:
    """
    """
    p: Regex = compile(pattern, flags)
    return p.split(string, maxsplit)


def findall(pattern: str, string: str, flags: int = 0) -> List[Any]:
    """
    """
    p: Regex = compile(pattern, flags)
    return p.findall(string)


def finditer(pattern: str, string: str, flags: int = 0) -> Iterator[Match]:
    """
    """
    p: Regex = compile(pattern, flags)
    return p.finditer(string, 0, None)


def sub(pattern: str, repl: Any, string: str, count: int = 0, flags: int = 0) -> str:
    """
    """
    p: Regex = compile(pattern, flags)
    return p.sub(repl, string, count)


def subn(pattern: str, repl: Any, string: str, count: int = 0, flags: int = 0) -> Tuple[str, int]:
    """
    """
    p: Regex = compile(pattern, flags)
    return p.subn(repl, string, count)


def escape(string: str) -> str:
    """ Escape a passed string so that we can send it to the
    regular expressions engine.
    """
    ret: Optional[str] = None

    def replfunc(m: Any) -> str:
        if m[0] == '\\':
            return '\\\\\\\\'
        else:
            return '\\\\' + m[0]

    __pragma__('js', '{}', '''
        var r = /[^A-Za-z:;\\d]/g;
        ret = string.replace(r, replfunc);
        ''')
    if ret is not None:
        return ret
    else:
        raise Exception('Failed to escape the passed string')


def purge() -> None:
    """ I think this function is unnecessary but included to keep interface
    consistent.
    """
    pass
