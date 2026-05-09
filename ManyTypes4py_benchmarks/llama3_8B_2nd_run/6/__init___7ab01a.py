from org.transcrypt.stubs.browser import __pragma__

class error(Exception):
    """ Regular Expression Exception Class
    """
    def __init__(self, msg: str, error: Exception, pattern: str = None, flags: int = 0, pos: int = None) -> None:
        """
        """
        Exception.__init__(self, msg, error=error)
        self.pattern = pattern
        self.flags = flags
        self.pos = pos

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

    # ... (other methods)

class Regex(object):
    """ Regular Expression Object
    """
    def __init__(self, pattern: str, flags: int) -> None:
        """
        @param pattern - javascript regular expression pattern as a string
        @param flags - string of javascript flags for the subsequently
           created RegExp object.
        """
        if not flags & ASCII > 0:
            flags |= UNICODE
        self._flags = flags
        self._jsFlags, self._obj = self._compileWrapper(pattern, flags)
        self._jspattern = pattern
        self._pypattern = pattern
        _, groupCounterRegex = self._compileWrapper(pattern + '|', flags)
        self._groups = groupCounterRegex.exec('').length - 1
        self._groupindex = None

    # ... (other methods)

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
        jsTokens, inlineFlags, namedGroups, nCapGroups, n_splits = translate(pyPattern)
        flags |= inlineFlags
        jsPattern = ''.join(jsTokens)
        Regex.__init__(self, jsPattern, flags)
        self._pypattern = pyPattern
        self._nsplits = n_splits
        self._jsTokens = jsTokens
        self._capgroups = nCapGroups
        self._groupindex = namedGroups

def compile(pattern: str, flags: int = 0) -> Regex:
    """ Compile a regex object and return
    an object that can be used for further processing.
    """
    if flags & JSSTRICT:
        p = Regex(pattern, flags)
    else:
        p = PyRegExp(pattern, flags)
    return p

def search(pattern: str, string: str, flags: int = 0) -> Match:
    """ Search a string for a particular matching pattern
    """
    p = compile(pattern, flags)
    return p.search(string)

def match(pattern: str, string: str, flags: int = 0) -> Match:
    """ Match a string for a particular pattern
    """
    p = compile(pattern, flags)
    return p.match(string)

def fullmatch(pattern: str, string: str, flags: int = 0) -> Match:
    """
    """
    p = compile(pattern, flags)
    return p.fullmatch(string)

def split(pattern: str, string: str, maxsplit: int = 0, flags: int = 0) -> list:
    """
    """
    p = compile(pattern, flags)
    return p.split(string, maxsplit)

def findall(pattern: str, string: str, flags: int = 0) -> list:
    """
    """
    p = compile(pattern, flags)
    return p.findall(string)

def finditer(pattern: str, string: str, flags: int = 0) -> iter:
    """
    """
    p = compile(pattern, flags)
    return p.finditer(string)

def sub(pattern: str, repl: str, string: str, count: int = 0, flags: int = 0) -> str:
    """
    """
    p = compile(pattern, flags)
    return p.sub(repl, string, count)

def subn(pattern: str, repl: str, string: str, count: int = 0, flags: int = 0) -> tuple:
    """
    """
    p = compile(pattern, flags)
    return p.subn(repl, string, count)

def escape(string: str) -> str:
    """ Escape a passed string so that we can send it to the
    regular expressions engine.
    """
    ret = None

    def replfunc(m: match) -> str:
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
