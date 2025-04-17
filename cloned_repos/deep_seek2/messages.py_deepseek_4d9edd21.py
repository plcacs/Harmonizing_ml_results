from typing import Any, Tuple, Optional

class Message(object):
    message: str = ''
    message_args: Tuple[Any, ...] = ()

    def __init__(self, filename: str, loc: Any) -> None:
        self.filename: str = filename
        self.lineno: int = loc.lineno
        self.col: int = getattr(loc, 'col_offset', 0)

    def __str__(self) -> str:
        return '%s:%s: %s' % (self.filename, self.lineno,
                              self.message % self.message_args)


class UnusedImport(Message):
    message: str = '%r imported but unused'

    def __init__(self, filename: str, loc: Any, name: str) -> None:
        super().__init__(filename, loc)
        self.message_args: Tuple[str] = (name,)


class RedefinedWhileUnused(Message):
    message: str = 'redefinition of unused %r from line %r'

    def __init__(self, filename: str, loc: Any, name: str, orig_loc: Any) -> None:
        super().__init__(filename, loc)
        self.message_args: Tuple[str, int] = (name, orig_loc.lineno)


class RedefinedInListComp(Message):
    message: str = 'list comprehension redefines %r from line %r'

    def __init__(self, filename: str, loc: Any, name: str, orig_loc: Any) -> None:
        super().__init__(filename, loc)
        self.message_args: Tuple[str, int] = (name, orig_loc.lineno)


class ImportShadowedByLoopVar(Message):
    message: str = 'import %r from line %r shadowed by loop variable'

    def __init__(self, filename: str, loc: Any, name: str, orig_loc: Any) -> None:
        super().__init__(filename, loc)
        self.message_args: Tuple[str, int] = (name, orig_loc.lineno)


class ImportStarUsed(Message):
    message: str = "'from %s import *' used; unable to detect undefined names"

    def __init__(self, filename: str, loc: Any, modname: str) -> None:
        super().__init__(filename, loc)
        self.message_args: Tuple[str] = (modname,)


class UndefinedName(Message):
    message: str = 'undefined name %r'

    def __init__(self, filename: str, loc: Any, name: str) -> None:
        super().__init__(filename, loc)
        self.message_args: Tuple[str] = (name,)


class DoctestSyntaxError(Message):
    message: str = 'syntax error in doctest'

    def __init__(self, filename: str, loc: Any, position: Optional[Tuple[int, int]] = None) -> None:
        super().__init__(filename, loc)
        if position:
            (self.lineno, self.col) = position
        self.message_args: Tuple[()] = ()


class UndefinedExport(Message):
    message: str = 'undefined name %r in __all__'

    def __init__(self, filename: str, loc: Any, name: str) -> None:
        super().__init__(filename, loc)
        self.message_args: Tuple[str] = (name,)


class UndefinedLocal(Message):
    message: str = ('local variable %r (defined in enclosing scope on line %r) '
                   'referenced before assignment')

    def __init__(self, filename: str, loc: Any, name: str, orig_loc: Any) -> None:
        super().__init__(filename, loc)
        self.message_args: Tuple[str, int] = (name, orig_loc.lineno)


class DuplicateArgument(Message):
    message: str = 'duplicate argument %r in function definition'

    def __init__(self, filename: str, loc: Any, name: str) -> None:
        super().__init__(filename, loc)
        self.message_args: Tuple[str] = (name,)


class LateFutureImport(Message):
    message: str = 'future import(s) %r after other statements'

    def __init__(self, filename: str, loc: Any, names: Tuple[str, ...]) -> None:
        super().__init__(filename, loc)
        self.message_args: Tuple[Tuple[str, ...]] = (names,)


class UnusedVariable(Message):
    """
    Indicates that a variable has been explicity assigned to but not actually
    used.
    """
    message: str = 'local variable %r is assigned to but never used'

    def __init__(self, filename: str, loc: Any, names: Tuple[str, ...]) -> None:
        super().__init__(filename, loc)
        self.message_args: Tuple[Tuple[str, ...]] = (names,)


class ReturnWithArgsInsideGenerator(Message):
    """
    Indicates a return statement with arguments inside a generator.
    """
    message: str = '\'return\' with argument inside generator'


class ReturnOutsideFunction(Message):
    """
    Indicates a return statement outside of a function/method.
    """
    message: str = '\'return\' outside function'
