from org.transcrypt.stubs.browser import __pragma__
from re.translate import translate

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

    def __init__(self, msg: str, error: Exception, pattern: str = None, flags: int = 0, pos: int = None) -> None:
        """
        """
        Exception.__init__(self, msg, error=error)
        self.pattern: str = pattern
        self.flags: int = flags
        self.pos: int = pos

class ReIndexError(IndexError):
    """ Index Error variant for the re module - primarily used for
    the group method in the Match Object.
    """

    def __init__(self) -> None:
        IndexError.__init__(self, 'no such group')

class Match(object):
    """ Resulting Match from a Regex Operation
    """

    def __init__(self, mObj: list, string: str, pos: int, endpos: int, rObj: Regex, namedGroups: dict = None) -> None:
        """
        """
        for index, match in enumerate(mObj):
            mObj[index] = None if mObj[index] == js_undefined else mObj[index]
        self._obj: list = mObj
        self._pos: int = pos
        self._endpos: int = endpos
        self._re: Regex = rObj
        self._string: str = string
        self._namedGroups: dict = namedGroups
        self._lastindex: int = self._lastMatchGroup()
        if self._namedGroups is not None:
            self._lastgroup: str = self._namedGroups[self._lastindex]
        else:
            self._lastgroup: str = None

    def _getPos(self) -> int:
        return self._pos

    def _setPos(self, val: int) -> None:
        raise AttributeError('readonly attribute')
    pos: int = property(_getPos, _setPos)

    def _getEndPos(self) -> int:
        return self._endpos

    def _setEndPos(self, val: int) -> None:
        raise AttributeError('readonly attribute')
    endpos: int = property(_getEndPos, _setEndPos)

    def _getRe(self) -> Regex:
        return self._re

    def _setRe(self, val: Regex) -> None:
        raise AttributeError('readonly attribute')
    re: Regex = property(_getRe, _setRe)

    def _getString(self) -> str:
        return self._string

    def _setString(self, val: str) -> None:
        raise AttributeError('readonly attribute')
    string: str = property(_getString, _setString)

    def _getLastGroup(self) -> str:
        return self._lastgroup

    def _setLastGroup(self, val: str) -> None:
        raise AttributeError('readonly attribute')
    lastgroup: str = property(_getLastGroup, _setLastGroup)

    def _getLastIndex(self) -> int:
        return self._lastindex

    def _setLastIndex(self, val: int) -> None:
        raise AttributeError('readonly attribute')
    lastindex: int = property(_getLastIndex, _setLastIndex)

    def _lastMatchGroup(self) -> int:
        """ Determine the last matching group in the object
        """
        if len(self._obj) > 1:
            for i in range(len(self._obj) - 1, 0, -1):
                if self._obj[i] is not None:
                    return i
            return None
        else:
            return None

    def expand(self, template: str) -> None:
        """
        """
        raise NotImplementedError()

    def group(self, *args: int) -> str:
        """ Return the string[s] for a group[s]
        if only one group is provided, a string is returned
        if multiple groups are provided, a tuple of strings is returned
        """
        ret: list = []
        if len(args) > 0:
            for index in args:
                if type(index) is str:
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

    def groups(self, default: str = None) -> tuple:
        """ Get all the groups in this match. Replace any
        groups that did not contribute to the match with default
        value.
        """
        if len(self._obj) > 1:
            ret: list = self._obj[1:]
            return tuple([x if x is not None else default for x in ret])
        else:
            return tuple()

    def groupdict(self, default: str = None) -> dict:
        """ The concept of named captures doesn't exist
        in javascript so this will likely never be implemented.
        For the python translated re we will have a group dict where
        possible.
        """
        if self._namedGroups is not None:
            ret: dict = {}
            for gName, gId in self._namedGroups.items():
                value = self._obj[gId]
                ret[gName] = value if value is not None else default
            return ret
        else:
            raise NotImplementedError('No NamedGroups Available')

    def start(self, group: int = 0) -> int:
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
        if type(group) is str:
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

    def end(self, group: int = 0) -> int:
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
        if type(group) is str:
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

    def span(self, group: int = 0) -> tuple:
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
        self._jsFlags, self._obj = self._compileWrapper(pattern, flags)
        self._jspattern: str = pattern
        self._pypattern: str = pattern
        _, groupCounterRegex = self._compileWrapper(pattern + '|', flags)
        self._groups: int = groupCounterRegex.exec('').length - 1
        self._groupindex: dict = None

    def _getPattern(self) -> str:
        ret: str = self._pypattern.replace('\\', '\\\\')
        return ret

    def _setPattern(self, val: str) -> None:
        raise AttributeError('readonly attribute')
    pattern: str = property(_getPattern, _setPattern)

    def _getFlags(self) -> int:
        return self._flags

    def _setFlags(self, val: int) -> None:
        raise AttributeError('readonly attribute')
    flags: int = property(_getFlags, _setFlags)

    def _getGroups(self) -> int:
        return self._groups

    def _setGroups(self, val: int) -> None:
        raise AttributeError('readonly attribute')
    groups: int = property(_getGroups, _setGroups)

    def _getGroupIndex(self) -> dict:
        if self._groupindex is None:
            return {}
        else:
            return self._groupindex

    def _setGroupIndex(self, val: dict) -> None:
        raise AttributeError('readonly attribute')
    groupindex: dict = property(_getGroupIndex, _setGroupIndex)

    def _compileWrapper(self, pattern: str, flags: int = 0) -> tuple:
        """ This function wraps the creation of the the
        regular expresion so that we can catch the
        Syntax Error exception and turn it into a
        Python Exception
        """
        jsFlags: str = self._convertFlags(flags)
        rObj: object = None
        errObj: object = None
        __pragma__('js', '{}', '\n                   try {\n                     rObj = new RegExp(pattern, jsFlags)\n                   } catch( err ) {\n                     errObj = err\n                   }\n                   ')
        if errObj is not None:
            raise error(errObj.message, errObj, pattern, flags)
        return (jsFlags, rObj)

    def _convertFlags(self, flags: int) -> str:
        """ Convert the Integer map based flags to a
        string list of flags for js
        """
        bitmaps: list = [(DEBUG, ''), (IGNORECASE, 'i'), (MULTILINE, 'm'), (STICKY, 'y'), (GLOBAL, 'g'), (UNICODE, 'u')]
        ret: str = ''.join([x[1] for x in bitmaps if x[0] & flags > 0])
        return ret

    def _getTargetStr(self, string: str, pos: int, endpos: int) -> str:
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

    def search(self, string: str, pos: int = 0, endpos: int = None) -> Match:
        """ Search through a string for matches to
        this regex object. @see the python docs
        """
        if endpos is None:
            endpos = len(string)
        rObj: object = self._obj
        m: object = rObj.exec(string)
        if m:
            if m.index < pos or m.index > endpos:
                return None
            else:
                return Match(m, string, pos, endpos, self, self._groupindex)
        else:
            return None

    def match(self, string: str, pos: int = 0, endpos: int = None) -> Match:
        """ Match this regex at the beginning of the passed
        string. @see the python docs
        """
        target: str = string
        if endpos is not None:
            target = target[:endpos]
        else:
            endpos = len(string)
        rObj: object = self._obj
        m: object = rObj.exec(target)
        if m:
            if m.index == pos:
                return Match(m, string, pos, endpos, self, self._groupindex)
            else:
                return None
        else:
            return None

    def fullmatch(self, string: str, pos: int = 0, endpos: int = None) -> Match:
        """ Match the entirety of the passed string to this regex
        object. @see the python docs
        """
        target: str = string
        strEndPos: int = len(string)
        if endpos is not None:
            target = target[:endpos]
            strEndPos = endpos
        rObj: object = self._obj
        m: object = rObj.exec(target)
        if m:
            obsEndPos: int = m.index + len(m[0])
            if m.index == pos and obsEndPos == strEndPos:
                return Match(m, string, pos, strEndPos, self, self._groupindex)
            else:
                return None
        else:
            return None

    def split(self, string: str, maxsplit: int = 0) -> list:
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
        mObj: object = None
        rObj: object = self._obj
        if maxsplit == 0:
            mObj = string.split(rObj)
            return mObj
        else:
            flags: int = self._flags
            flags |= GLOBAL
            _, rObj = self._compileWrapper(self._jspattern, flags)
            ret: list = []
            lastM: object = None
            cnt: int = 0
            for i in range(0, maxsplit):
                m: object = rObj.exec(string)
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

    def _findAllMatches(self, string: str, pos: int = 0, endpos: int = None) -> list:
        target: str = self._getTargetStr(string, pos, endpos)
        flags: int = self._flags
        flags |= GLOBAL
        _, rObj: object = self._compileWrapper(self._jspattern, flags)
        ret: list = []
        while True:
            m: object = rObj.exec(target)
            if m:
                ret.append(m)
            else:
                break
        return ret

    def findall(self, string: str, pos: int = 0, endpos: int = None) -> list:
        """ Find All the matches to this regex in the passed string
        @return either:
          List of strings of the matched regex has 1 or 0 capture
            groups
          List of elements that are each a list of the groups matched
            at each location in the string.
        @see the python docs
        """
        mlist: list = self._findAllMatches(string, pos, endpos)

        def mSelect(m: object) -> object:
            if len(m) > 2:
                return tuple(m[1:])
            elif len(m) == 2:
                return m[1]
            else:
                return m[0]
        ret: list = map(mSelect, mlist)
        return ret

    def finditer(self, string: str, pos: int, endpos: int = None) -> iter:
        """ Like findall but returns an iterator instead of
        a list.
        @see the python docs
        """
        mlist: list = self._findAllMatches(string, pos, endpos)
        ret: iter = map(lambda m: Match(m, string, 0, len(string), self, self._groupindex), mlist)
        return iter(ret)

    def sub(self, repl: str, string: str, count: int = 0) -> str:
        """ Substitude each match of this regex in the passed string
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

    def subn(self, repl: str, string: str, count: int = 0) -> tuple:
        """ Similar to sub except that instead of just returning the
        augmented string, it returns a tuple of the augmented string
        and the number of times that the replacement op occured.
        (augstr, numreplacements)
        @see the python docs
        """
        flags: int = self._flags
        flags |= GLOBAL
        _, rObj: object = self._compileWrapper(self._jspattern, flags)
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
            m: object = rObj.exec(string)
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
        jsTokens: list
        inlineFlags: int
        namedGroups: dict
        nCapGroups: int
        n_splits: int
        jsTokens, inlineFlags, namedGroups, nCapGroups, n_splits = translate(pyPattern)
        flags |= inlineFlags
        jsPattern: str = ''.join(jsTokens)
        Regex.__init__(self, jsPattern, flags)
        self._pypattern: str = pyPattern
        self._nsplits: int = n_splits
        self._jsTokens: list = jsTokens
        self._capgroups: int = nCapGroups
        self._groupindex: dict = namedGroups

def compile(pattern: str, flags: int = 0) -> Regex:
    """ Compile a regex object and return
    an object that can be used for further processing.
    """
    if flags & JSSTRICT:
        p: Regex = Regex(pattern, flags)
    else:
        p: Regex = PyRegExp(pattern, flags)
    return p

def search(pattern: str, string: str, flags: int = 0) -> Match:
    """ Search a string for a particular matching pattern
    """
    p: Regex = compile(pattern, flags)
    return p.search(string)

def match(pattern: str, string: str, flags: int = 0) -> Match:
    """ Match a string for a particular pattern
    """
    p: Regex = compile(pattern, flags)
    return p.match(string)

def fullmatch(pattern: str, string: str, flags: int = 0) -> Match:
    """
    """
    p: Regex = compile(pattern, flags)
    return p.fullmatch(string)

def split(pattern: str, string: str, maxsplit: int = 0, flags: int = 0) -> list:
    """
    """
    p: Regex = compile(pattern, flags)
    return p.split(string, maxsplit)

def findall(pattern: str, string: str, flags: int = 0) -> list:
    """
    """
    p: Regex = compile(pattern, flags)
    return p.findall(string)

def finditer(pattern: str, string: str, flags: int = 0) -> iter:
    """
    """
    p: Regex = compile(pattern, flags)
    return p.finditer(string)

def sub(pattern: str, repl: str, string: str, count: int = 0, flags: int = 0) -> str:
    """
    """
    p: Regex = compile(pattern, flags)
    return p.sub(repl, string, count)

def subn(pattern: str, repl: str, string: str, count: int = 0, flags: int = 0) -> tuple:
    """
    """
    p: Regex = compile(pattern, flags)
    return p.subn(repl, string, count)

def escape(string: str) -> str:
    """ Escape a passed string so that we can send it to the
    regular expressions engine.
    """
    ret: str = None

    def replfunc(m: str) -> str:
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
    """ I think this function is unnecessary but included to keep interface
    consistent.
    """
    pass
