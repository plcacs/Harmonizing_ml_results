"""
Provide the class Message and its subclasses.
"""

from typing import Any, list, str, tuple

class Message(object):
    message = ''
    message_args = ()

    def __init__(self, filename: str, loc: Any) -> None:
        self.filename = filename
        self.lineno = loc.lineno
        self.col = getattr(loc, 'col_offset', 0)

    def __str__(self) -> str:
        return '%s:%s:%s %s' % (self.filename, self.lineno, self.col + 1, self.message % self.message_args)

class UnusedImport(Message):
    message = '%r imported but unused'

    def __init__(self, filename: str, loc: Any, name: str) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (name,)

class RedefinedWhileUnused(Message):
    message = 'redefinition of unused %r from line %r'

    def __init__(self, filename: str, loc: Any, name: str, orig_loc: Any) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (name, orig_loc.lineno)

class RedefinedInListComp(Message):
    message = 'list comprehension redefines %r from line %r'

    def __init__(self, filename: str, loc: Any, name: str, orig_loc: Any) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (name, orig_loc.lineno)

class ImportShadowedByLoopVar(Message):
    message = 'import %r from line %r shadowed by loop variable'

    def __init__(self, filename: str, loc: Any, name: str, orig_loc: Any) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (name, orig_loc.lineno)

class ImportStarNotPermitted(Message):
    message = "'from %s import *' only allowed at module level"

    def __init__(self, filename: str, loc: Any, modname: str) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (modname,)

class ImportStarUsed(Message):
    message = "'from %s import *' used; unable to detect undefined names"

    def __init__(self, filename: str, loc: Any, modname: str) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (modname,)

class ImportStarUsage(Message):
    message = '%r may be undefined, or defined from star imports: %s'

    def __init__(self, filename: str, loc: Any, name: str, from_list: list[str]) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (name, from_list)

class UndefinedName(Message):
    message = 'undefined name %r'

    def __init__(self, filename: str, loc: Any, name: str) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (name,)

class DoctestSyntaxError(Message):
    message = 'syntax error in doctest'

    def __init__(self, filename: str, loc: Any, position: tuple[int, int] | None = None) -> None:
        Message.__init__(self, filename, loc)
        if position:
            self.lineno, self.col = position
        self.message_args = ()

class UndefinedExport(Message):
    message = 'undefined name %r in __all__'

    def __init__(self, filename: str, loc: Any, name: str) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (name,)

class UndefinedLocal(Message):
    message = 'local variable %r {0} referenced before assignment'
    default = 'defined in enclosing scope on line %r'
    builtin = 'defined as a builtin'

    def __init__(self, filename: str, loc: Any, name: str, orig_loc: Any | None) -> None:
        Message.__init__(self, filename, loc)
        if orig_loc is None:
            self.message = self.message.format(self.builtin)
            self.message_args = name
        else:
            self.message = self.message.format(self.default)
            self.message_args = (name, orig_loc.lineno)

class DuplicateArgument(Message):
    message = 'duplicate argument %r in function definition'

    def __init__(self, filename: str, loc: Any, name: str) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (name,)

class MultiValueRepeatedKeyLiteral(Message):
    message = 'dictionary key %r repeated with different values'

    def __init__(self, filename: str, loc: Any, key: str) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (key,)

class MultiValueRepeatedKeyVariable(Message):
    message = 'dictionary key variable %s repeated with different values'

    def __init__(self, filename: str, loc: Any, key: str) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (key,)

class LateFutureImport(Message):
    message = 'from __future__ imports must occur at the beginning of the file'

    def __init__(self, filename: str, loc: Any, names: list[str]) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = ()

class FutureFeatureNotDefined(Message):
    """An undefined __future__ feature name was imported."""
    message = 'future feature %s is not defined'

    def __init__(self, filename: str, loc: Any, name: str) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (name,)

class UnusedVariable(Message):
    """
    Indicates that a variable has been explicitly assigned to but not actually
    used.
    """
    message = 'local variable %r is assigned to but never used'

    def __init__(self, filename: str, loc: Any, names: str) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (names,)