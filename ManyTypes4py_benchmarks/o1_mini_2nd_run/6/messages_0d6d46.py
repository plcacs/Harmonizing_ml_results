class Message:
    message: str = ''
    message_args: tuple = ()
    filename: str
    lineno: int
    col: int

    def __init__(self, filename: str, loc: 'Location') -> None:
        self.filename = filename
        self.lineno = loc.lineno
        self.col = getattr(loc, 'col_offset', 0)

    def __str__(self) -> str:
        return f"{self.filename}:{self.lineno}: {self.message % self.message_args}"


class UnusedImport(Message):
    message: str = '%r imported but unused'

    def __init__(self, filename: str, loc: 'Location', name: str) -> None:
        super().__init__(filename, loc)
        self.message_args = (name,)


class RedefinedWhileUnused(Message):
    message: str = 'redefinition of unused %r from line %r'

    def __init__(self, filename: str, loc: 'Location', name: str, orig_loc: 'Location') -> None:
        super().__init__(filename, loc)
        self.message_args = (name, orig_loc.lineno)


class RedefinedInListComp(Message):
    message: str = 'list comprehension redefines %r from line %r'

    def __init__(self, filename: str, loc: 'Location', name: str, orig_loc: 'Location') -> None:
        super().__init__(filename, loc)
        self.message_args = (name, orig_loc.lineno)


class ImportShadowedByLoopVar(Message):
    message: str = 'import %r from line %r shadowed by loop variable'

    def __init__(self, filename: str, loc: 'Location', name: str, orig_loc: 'Location') -> None:
        super().__init__(filename, loc)
        self.message_args = (name, orig_loc.lineno)


class ImportStarUsed(Message):
    message: str = "'from %s import *' used; unable to detect undefined names"

    def __init__(self, filename: str, loc: 'Location', modname: str) -> None:
        super().__init__(filename, loc)
        self.message_args = (modname,)


class UndefinedName(Message):
    message: str = 'undefined name %r'

    def __init__(self, filename: str, loc: 'Location', name: str) -> None:
        super().__init__(filename, loc)
        self.message_args = (name,)


class DoctestSyntaxError(Message):
    message: str = 'syntax error in doctest'

    def __init__(self, filename: str, loc: 'Location', position: tuple[int, int] = None) -> None:
        super().__init__(filename, loc)
        if position:
            self.lineno, self.col = position
        self.message_args = ()


class UndefinedExport(Message):
    message: str = 'undefined name %r in __all__'

    def __init__(self, filename: str, loc: 'Location', name: str) -> None:
        super().__init__(filename, loc)
        self.message_args = (name,)


class UndefinedLocal(Message):
    message: str = 'local variable %r (defined in enclosing scope on line %r) referenced before assignment'

    def __init__(self, filename: str, loc: 'Location', name: str, orig_loc: 'Location') -> None:
        super().__init__(filename, loc)
        self.message_args = (name, orig_loc.lineno)


class DuplicateArgument(Message):
    message: str = 'duplicate argument %r in function definition'

    def __init__(self, filename: str, loc: 'Location', name: str) -> None:
        super().__init__(filename, loc)
        self.message_args = (name,)


class LateFutureImport(Message):
    message: str = 'future import(s) %r after other statements'

    def __init__(self, filename: str, loc: 'Location', names: tuple[str, ...]) -> None:
        super().__init__(filename, loc)
        self.message_args = (names,)


class UnusedVariable(Message):
    """
    Indicates that a variable has been explicitly assigned to but not actually
    used.
    """
    message: str = 'local variable %r is assigned to but never used'

    def __init__(self, filename: str, loc: 'Location', names: str) -> None:
        super().__init__(filename, loc)
        self.message_args = (names,)


class ReturnWithArgsInsideGenerator(Message):
    """
    Indicates a return statement with arguments inside a generator.
    """
    message: str = "'return' with argument inside generator"


class ReturnOutsideFunction(Message):
    """
    Indicates a return statement outside of a function/method.
    """
    message: str = "'return' outside function"


class Location:
    lineno: int
    col_offset: int
