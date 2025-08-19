"""
Provide the class Message and its subclasses.
"""

from typing import ClassVar, Optional, Protocol, Tuple, Any, cast


class HasLineNo(Protocol):
    lineno: int


class Message(object):
    message: ClassVar[str] = ''
    message_args: tuple[Any, ...] = ()

    filename: str
    lineno: int
    col: int

    def __init__(self, filename: str, loc: HasLineNo) -> None:
        self.filename = filename
        self.lineno = loc.lineno
        self.col = cast(int, getattr(loc, 'col_offset', 0))

    def __str__(self) -> str:
        return '%s:%s: %s' % (self.filename, self.lineno, self.message % self.message_args)


class UnusedImport(Message):
    message: ClassVar[str] = '%r imported but unused'

    def __init__(self, filename: str, loc: HasLineNo, name: str) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (name,)


class RedefinedWhileUnused(Message):
    message: ClassVar[str] = 'redefinition of unused %r from line %r'

    def __init__(self, filename: str, loc: HasLineNo, name: str, orig_loc: HasLineNo) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (name, orig_loc.lineno)


class RedefinedInListComp(Message):
    message: ClassVar[str] = 'list comprehension redefines %r from line %r'

    def __init__(self, filename: str, loc: HasLineNo, name: str, orig_loc: HasLineNo) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (name, orig_loc.lineno)


class ImportShadowedByLoopVar(Message):
    message: ClassVar[str] = 'import %r from line %r shadowed by loop variable'

    def __init__(self, filename: str, loc: HasLineNo, name: str, orig_loc: HasLineNo) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (name, orig_loc.lineno)


class ImportStarUsed(Message):
    message: ClassVar[str] = "'from %s import *' used; unable to detect undefined names"

    def __init__(self, filename: str, loc: HasLineNo, modname: str) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (modname,)


class UndefinedName(Message):
    message: ClassVar[str] = 'undefined name %r'

    def __init__(self, filename: str, loc: HasLineNo, name: str) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (name,)


class DoctestSyntaxError(Message):
    message: ClassVar[str] = 'syntax error in doctest'

    def __init__(self, filename: str, loc: HasLineNo, position: Optional[Tuple[int, int]] = None) -> None:
        Message.__init__(self, filename, loc)
        if position:
            self.lineno, self.col = position
        self.message_args = ()


class UndefinedExport(Message):
    message: ClassVar[str] = 'undefined name %r in __all__'

    def __init__(self, filename: str, loc: HasLineNo, name: str) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (name,)


class UndefinedLocal(Message):
    message: ClassVar[str] = 'local variable %r (defined in enclosing scope on line %r) referenced before assignment'

    def __init__(self, filename: str, loc: HasLineNo, name: str, orig_loc: HasLineNo) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (name, orig_loc.lineno)


class DuplicateArgument(Message):
    message: ClassVar[str] = 'duplicate argument %r in function definition'

    def __init__(self, filename: str, loc: HasLineNo, name: str) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (name,)


class LateFutureImport(Message):
    message: ClassVar[str] = 'future import(s) %r after other statements'

    def __init__(self, filename: str, loc: HasLineNo, names: object) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (names,)


class UnusedVariable(Message):
    """
    Indicates that a variable has been explicity assigned to but not actually
    used.
    """
    message: ClassVar[str] = 'local variable %r is assigned to but never used'

    def __init__(self, filename: str, loc: HasLineNo, names: str) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (names,)


class ReturnWithArgsInsideGenerator(Message):
    """
    Indicates a return statement with arguments inside a generator.
    """
    message: ClassVar[str] = "'return' with argument inside generator"


class ReturnOutsideFunction(Message):
    """
    Indicates a return statement outside of a function/method.
    """
    message: ClassVar[str] = "'return' outside function"