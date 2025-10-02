from typing import Any, Tuple, Optional, Protocol


class Location(Protocol):
    lineno: int
    col_offset: int


class Message(object):
    message: str = ''
    message_args: Tuple[Any, ...] = ()

    def __init__(self, filename: str, loc: Location) -> None:
        self.filename: str = filename
        self.lineno: int = loc.lineno
        self.col: int = getattr(loc, 'col_offset', 0)

    def __str__(self) -> str:
        return '%s:%s:%s %s' % (
            self.filename,
            self.lineno,
            self.col + 1,
            self.message % self.message_args,
        )


class UnusedImport(Message):
    message: str = '%r imported but unused'

    def __init__(self, filename: str, loc: Location, name: str) -> None:
        super().__init__(filename, loc)
        self.message_args = (name,)


class RedefinedWhileUnused(Message):
    message: str = 'redefinition of unused %r from line %r'

    def __init__(self, filename: str, loc: Location, name: str, orig_loc: Location) -> None:
        super().__init__(filename, loc)
        self.message_args = (name, orig_loc.lineno)


class RedefinedInListComp(Message):
    message: str = 'list comprehension redefines %r from line %r'

    def __init__(self, filename: str, loc: Location, name: str, orig_loc: Location) -> None:
        super().__init__(filename, loc)
        self.message_args = (name, orig_loc.lineno)


class ImportShadowedByLoopVar(Message):
    message: str = 'import %r from line %r shadowed by loop variable'

    def __init__(self, filename: str, loc: Location, name: str, orig_loc: Location) -> None:
        super().__init__(filename, loc)
        self.message_args = (name, orig_loc.lineno)


class ImportStarNotPermitted(Message):
    message: str = "'from %s import *' only allowed at module level"

    def __init__(self, filename: str, loc: Location, modname: str) -> None:
        super().__init__(filename, loc)
        self.message_args = (modname,)


class ImportStarUsed(Message):
    message: str = "'from %s import *' used; unable to detect undefined names"

    def __init__(self, filename: str, loc: Location, modname: str) -> None:
        super().__init__(filename, loc)
        self.message_args = (modname,)


class ImportStarUsage(Message):
    message: str = '%r may be undefined, or defined from star imports: %s'

    def __init__(self, filename: str, loc: Location, name: str, from_list: str) -> None:
        super().__init__(filename, loc)
        self.message_args = (name, from_list)


class UndefinedName(Message):
    message: str = 'undefined name %r'

    def __init__(self, filename: str, loc: Location, name: str) -> None:
        super().__init__(filename, loc)
        self.message_args = (name,)


class DoctestSyntaxError(Message):
    message: str = 'syntax error in doctest'

    def __init__(self, filename: str, loc: Location, position: Optional[Tuple[int, int]] = None) -> None:
        super().__init__(filename, loc)
        if position:
            self.lineno, self.col = position
        self.message_args = ()


class UndefinedExport(Message):
    message: str = 'undefined name %r in __all__'

    def __init__(self, filename: str, loc: Location, name: str) -> None:
        super().__init__(filename, loc)
        self.message_args = (name,)


class UndefinedLocal(Message):
    message: str = 'local variable %r {0} referenced before assignment'
    default: str = 'defined in enclosing scope on line %r'
    builtin: str = 'defined as a builtin'

    def __init__(self, filename: str, loc: Location, name: str, orig_loc: Optional[Location]) -> None:
        super().__init__(filename, loc)
        if orig_loc is None:
            self.message = self.message.format(self.builtin)
            self.message_args = name  # type: ignore
        else:
            self.message = self.message.format(self.default)
            self.message_args = (name, orig_loc.lineno)


class DuplicateArgument(Message):
    message: str = 'duplicate argument %r in function definition'

    def __init__(self, filename: str, loc: Location, name: str) -> None:
        super().__init__(filename, loc)
        self.message_args = (name,)


class MultiValueRepeatedKeyLiteral(Message):
    message: str = 'dictionary key %r repeated with different values'

    def __init__(self, filename: str, loc: Location, key: Any) -> None:
        super().__init__(filename, loc)
        self.message_args = (key,)


class MultiValueRepeatedKeyVariable(Message):
    message: str = 'dictionary key variable %s repeated with different values'

    def __init__(self, filename: str, loc: Location, key: Any) -> None:
        super().__init__(filename, loc)
        self.message_args = (key,)


class LateFutureImport(Message):
    message: str = 'from __future__ imports must occur at the beginning of the file'

    def __init__(self, filename: str, loc: Location, names: Tuple[str, ...]) -> None:
        super().__init__(filename, loc)
        self.message_args = ()


class FutureFeatureNotDefined(Message):
    """An undefined __future__ feature name was imported."""
    message: str = 'future feature %s is not defined'

    def __init__(self, filename: str, loc: Location, name: str) -> None:
        super().__init__(filename, loc)
        self.message_args = (name,)


class UnusedVariable(Message):
    """
    Indicates that a variable has been explicitly assigned to but not actually
    used.
    """
    message: str = 'local variable %r is assigned to but never used'

    def __init__(self, filename: str, loc: Location, names: str) -> None:
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


class YieldOutsideFunction(Message):
    """
    Indicates a yield or yield from statement outside of a function/method.
    """
    message: str = "'yield' outside function"


class ContinueOutsideLoop(Message):
    """
    Indicates a continue statement outside of a while or for loop.
    """
    message: str = "'continue' not properly in loop"


class BreakOutsideLoop(Message):
    """
    Indicates a break statement outside of a while or for loop.
    """
    message: str = "'break' outside loop"


class ContinueInFinally(Message):
    """
    Indicates a continue statement in a finally block in a while or for loop.
    """
    message: str = "'continue' not supported inside 'finally' clause"


class DefaultExceptNotLast(Message):
    """
    Indicates an except: block as not the last exception handler.
    """
    message: str = "default 'except:' must be last"


class TwoStarredExpressions(Message):
    """
    Two or more starred expressions in an assignment (a, *b, *c = d).
    """
    message: str = 'two starred expressions in assignment'


class TooManyExpressionsInStarredAssignment(Message):
    """
    Too many expressions in an assignment with star-unpacking
    """
    message: str = 'too many expressions in star-unpacking assignment'


class IfTuple(Message):
    """
    Conditional test is a non-empty tuple literal, which are always True.
    """
    message: str = "'if tuple literal' is always true, perhaps remove accidental comma?"


class AssertTuple(Message):
    """
    Assertion test is a non-empty tuple literal, which are always True.
    """
    message: str = 'assertion is always true, perhaps remove parentheses?'


class ForwardAnnotationSyntaxError(Message):
    message: str = 'syntax error in forward annotation %r'

    def __init__(self, filename: str, loc: Location, annotation: str) -> None:
        super().__init__(filename, loc)
        self.message_args = (annotation,)


class CommentAnnotationSyntaxError(Message):
    message: str = 'syntax error in type comment %r'

    def __init__(self, filename: str, loc: Location, annotation: str) -> None:
        super().__init__(filename, loc)
        self.message_args = (annotation,)


class RaiseNotImplemented(Message):
    message: str = "'raise NotImplemented' should be 'raise NotImplementedError'"


class InvalidPrintSyntax(Message):
    message: str = 'use of >> is invalid with print function'


class IsLiteral(Message):
    message: str = 'use ==/!= to compare constant literals (str, bytes, int, float, tuple)'


class FStringMissingPlaceholders(Message):
    message: str = 'f-string is missing placeholders'


class StringDotFormatExtraPositionalArguments(Message):
    message: str = "'...'.format(...) has unused arguments at position(s): %s"

    def __init__(self, filename: str, loc: Location, extra_positions: str) -> None:
        super().__init__(filename, loc)
        self.message_args = (extra_positions,)


class StringDotFormatExtraNamedArguments(Message):
    message: str = "'...'.format(...) has unused named argument(s): %s"

    def __init__(self, filename: str, loc: Location, extra_keywords: str) -> None:
        super().__init__(filename, loc)
        self.message_args = (extra_keywords,)


class StringDotFormatMissingArgument(Message):
    message: str = "'...'.format(...) is missing argument(s) for placeholder(s): %s"

    def __init__(self, filename: str, loc: Location, missing_arguments: str) -> None:
        super().__init__(filename, loc)
        self.message_args = (missing_arguments,)


class StringDotFormatMixingAutomatic(Message):
    message: str = "'...'.format(...) mixes automatic and manual numbering"


class StringDotFormatInvalidFormat(Message):
    message: str = "'...'.format(...) has invalid format string: %s"

    def __init__(self, filename: str, loc: Location, error: str) -> None:
        super().__init__(filename, loc)
        self.message_args = (error,)


class PercentFormatInvalidFormat(Message):
    message: str = "'...' %% ... has invalid format string: %s"

    def __init__(self, filename: str, loc: Location, error: str) -> None:
        super().__init__(filename, loc)
        self.message_args = (error,)


class PercentFormatMixedPositionalAndNamed(Message):
    message: str = "'...' %% ... has mixed positional and named placeholders"


class PercentFormatUnsupportedFormatCharacter(Message):
    message: str = "'...' %% ... has unsupported format character %r"

    def __init__(self, filename: str, loc: Location, c: str) -> None:
        super().__init__(filename, loc)
        self.message_args = (c,)


class PercentFormatPositionalCountMismatch(Message):
    message: str = "'...' %% ... has %d placeholder(s) but %d substitution(s)"

    def __init__(self, filename: str, loc: Location, n_placeholders: int, n_substitutions: int) -> None:
        super().__init__(filename, loc)
        self.message_args = (n_placeholders, n_substitutions)


class PercentFormatExtraNamedArguments(Message):
    message: str = "'...' %% ... has unused named argument(s): %s"

    def __init__(self, filename: str, loc: Location, extra_keywords: str) -> None:
        super().__init__(filename, loc)
        self.message_args = (extra_keywords,)


class PercentFormatMissingArgument(Message):
    message: str = "'...' %% ... is missing argument(s) for placeholder(s): %s"

    def __init__(self, filename: str, loc: Location, missing_arguments: str) -> None:
        super().__init__(filename, loc)
        self.message_args = (missing_arguments,)


class PercentFormatExpectedMapping(Message):
    message: str = "'...' %% ... expected mapping but got sequence"


class PercentFormatExpectedSequence(Message):
    message: str = "'...' %% ... expected sequence but got mapping"


class PercentFormatStarRequiresSequence(Message):
    message: str = "'...' %% ... `*` specifier requires sequence"