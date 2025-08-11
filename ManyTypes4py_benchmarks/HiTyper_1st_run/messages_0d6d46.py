"""
Provide the class Message and its subclasses.
"""

class Message(object):
    message = ''
    message_args = ()

    def __init__(self, filename: str, loc: str) -> None:
        self.filename = filename
        self.lineno = loc.lineno
        self.col = getattr(loc, 'col_offset', 0)

    def __str__(self) -> typing.Text:
        return '%s:%s: %s' % (self.filename, self.lineno, self.message % self.message_args)

class UnusedImport(Message):
    message = '%r imported but unused'

    def __init__(self, filename: str, loc: str, name) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (name,)

class RedefinedWhileUnused(Message):
    message = 'redefinition of unused %r from line %r'

    def __init__(self, filename: str, loc: str, name, orig_loc) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (name, orig_loc.lineno)

class RedefinedInListComp(Message):
    message = 'list comprehension redefines %r from line %r'

    def __init__(self, filename: str, loc: str, name, orig_loc) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (name, orig_loc.lineno)

class ImportShadowedByLoopVar(Message):
    message = 'import %r from line %r shadowed by loop variable'

    def __init__(self, filename: str, loc: str, name, orig_loc) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (name, orig_loc.lineno)

class ImportStarUsed(Message):
    message = "'from %s import *' used; unable to detect undefined names"

    def __init__(self, filename: str, loc: str, modname) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (modname,)

class UndefinedName(Message):
    message = 'undefined name %r'

    def __init__(self, filename: str, loc: str, name) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (name,)

class DoctestSyntaxError(Message):
    message = 'syntax error in doctest'

    def __init__(self, filename: str, loc: str, position=None) -> None:
        Message.__init__(self, filename, loc)
        if position:
            self.lineno, self.col = position
        self.message_args = ()

class UndefinedExport(Message):
    message = 'undefined name %r in __all__'

    def __init__(self, filename: str, loc: str, name) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (name,)

class UndefinedLocal(Message):
    message = 'local variable %r (defined in enclosing scope on line %r) referenced before assignment'

    def __init__(self, filename: str, loc: str, name, orig_loc) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (name, orig_loc.lineno)

class DuplicateArgument(Message):
    message = 'duplicate argument %r in function definition'

    def __init__(self, filename: str, loc: str, name) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (name,)

class LateFutureImport(Message):
    message = 'future import(s) %r after other statements'

    def __init__(self, filename: str, loc: str, names: str) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (names,)

class UnusedVariable(Message):
    """
    Indicates that a variable has been explicity assigned to but not actually
    used.
    """
    message = 'local variable %r is assigned to but never used'

    def __init__(self, filename: str, loc: str, names: str) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (names,)

class ReturnWithArgsInsideGenerator(Message):
    """
    Indicates a return statement with arguments inside a generator.
    """
    message = "'return' with argument inside generator"

class ReturnOutsideFunction(Message):
    """
    Indicates a return statement outside of a function/method.
    """
    message = "'return' outside function"