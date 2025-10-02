from org.transcrypt.stubs.browser import __pragma__
from re.translate import translate

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
    def __init__(self, msg: str, error, pattern=None, flags: int = 0, pos=None):
        ...

class ReIndexError(IndexError):
    def __init__(self):
        ...

class Match:
    def __init__(self, mObj, string: str, pos: int, endpos: int, rObj, namedGroups=None):
        ...

    def expand(self, template: str):
        ...

    def group(self, *args):
        ...

    def groups(self, default=None):
        ...

    def groupdict(self, default=None):
        ...

    def start(self, group=0):
        ...

    def end(self, group=0):
        ...

    def span(self, group=0):
        ...

class Regex:
    def __init__(self, pattern: str, flags: int):
        ...

    def search(self, string: str, pos: int = 0, endpos=None):
        ...

    def match(self, string: str, pos: int = 0, endpos=None):
        ...

    def fullmatch(self, string: str, pos: int = 0, endpos=None):
        ...

    def split(self, string: str, maxsplit: int = 0):
        ...

    def findall(self, string: str, pos: int = 0, endpos=None):
        ...

    def finditer(self, string: str, pos, endpos=None):
        ...

    def sub(self, repl, string: str, count: int = 0):
        ...

    def subn(self, repl, string: str, count: int = 0):
        ...

class PyRegExp(Regex):
    def __init__(self, pyPattern: str, flags: int):
        ...

def compile(pattern: str, flags: int = 0):
    ...

def search(pattern: str, string: str, flags: int = 0):
    ...

def match(pattern: str, string: str, flags: int = 0):
    ...

def fullmatch(pattern: str, string: str, flags: int = 0):
    ...

def split(pattern: str, string: str, maxsplit: int = 0, flags: int = 0):
    ...

def findall(pattern: str, string: str, flags: int = 0):
    ...

def finditer(pattern: str, string: str, flags: int = 0):
    ...

def sub(pattern: str, repl, string: str, count: int = 0, flags: int = 0):
    ...

def subn(pattern: str, repl, string: str, count: int = 0, flags: int = 0):
    ...

def escape(string: str):
    ...

def purge():
    ...
