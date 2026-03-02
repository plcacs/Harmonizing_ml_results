class Message:
    """Base class for messages."""
    message: str
    message_args: tuple

    def __init__(self, filename: str, loc) -> None:
        self.filename: str = filename
        self.lineno: int = loc.lineno
        self.col: int = getattr(loc, 'col_offset', 0)

    def __str__(self) -> str:
        return '%s:%s: %s' % (self.filename, self.lineno, self.message % self.message_args)

class UnusedImport(Message):
    """Message for unused import."""
    message: str = '%r imported but unused'

    def __init__(self, filename: str, loc, name: str) -> None:
        Message.__init__(self, filename, loc)
        self.message_args: tuple = (name,)

class RedefinedWhileUnused(Message):
    """Message for redefinition of unused variable."""
    message: str = 'redefinition of unused %r from line %r'

    def __init__(self, filename: str, loc, name: str, orig_loc) -> None:
        Message.__init__(self, filename, loc)
        self.message_args: tuple = (name, orig_loc.lineno)

class RedefinedInListComp(Message):
    """Message for redefinition in list comprehension."""
    message: str = 'list comprehension redefines %r from line %r'

    def __init__(self, filename: str, loc, name: str, orig_loc) -> None:
        Message.__init__(self, filename, loc)
        self.message_args: tuple = (name, orig_loc.lineno)

class ImportShadowedByLoopVar(Message):
    """Message for import shadowed by loop variable."""
    message: str = 'import %r from line %r shadowed by loop variable'

    def __init__(self, filename: str, loc, name: str, orig_loc) -> None:
        Message.__init__(self, filename, loc)
        self.message_args: tuple = (name, orig_loc.lineno)

class ImportStarUsed(Message):
    """Message for import star used."""
    message: str = "'from %s import *' used; unable to detect undefined names"

    def __init__(self, filename: str, loc, modname: str) -> None:
        Message.__init__(self, filename, loc)
        self.message_args: tuple = (modname,)

class UndefinedName(Message):
    """Message for undefined name."""
    message: str = 'undefined name %r'

    def __init__(self, filename: str, loc, name: str) -> None:
        Message.__init__(self, filename, loc)
        self.message_args: tuple = (name,)

class DoctestSyntaxError(Message):
    """Message for doctest syntax error."""
    message: str = 'syntax error in doctest'

    def __init__(self, filename: str, loc, position: tuple = None) -> None:
        Message.__init__(self, filename, loc)
        if position:
            self.lineno, self.col = position
        self.message_args: tuple = ()

class UndefinedExport(Message):
    """Message for undefined export."""
    message: str = 'undefined name %r in __all__'

    def __init__(self, filename: str, loc, name: str) -> None:
        Message.__init__(self, filename, loc)
        self.message_args: tuple = (name,)

class UndefinedLocal(Message):
    """Message for undefined local variable."""
    message: str = 'local variable %r (defined in enclosing scope on line %r) referenced before assignment'

    def __init__(self, filename: str, loc, name: str, orig_loc) -> None:
        Message.__init__(self, filename, loc)
        self.message_args: tuple = (name, orig_loc.lineno)

class DuplicateArgument(Message):
    """Message for duplicate argument."""
    message: str = 'duplicate argument %r in function definition'

    def __init__(self, filename: str, loc, name: str) -> None:
        Message.__init__(self, filename, loc)
        self.message_args: tuple = (name,)

class LateFutureImport(Message):
    """Message for late future import."""
    message: str = 'future import(s) %r after other statements'

    def __init__(self, filename: str, loc, names: str) -> None:
        Message.__init__(self, filename, loc)
        self.message_args: tuple = (names,)

class UnusedVariable(Message):
    """Message for unused variable."""
    message: str = 'local variable %r is assigned to but never used'

    def __init__(self, filename: str, loc, names: str) -> None:
        Message.__init__(self, filename, loc)
        self.message_args: tuple = (names,)

class ReturnWithArgsInsideGenerator(Message):
    """Message for return with arguments inside generator."""
    message: str = "'return' with argument inside generator"

class ReturnOutsideFunction(Message):
    """Message for return outside function."""
    message: str = "'return' outside function"
