"""
Provide the class Message and its subclasses.
"""

class Message:
    """Base class for messages."""
    message: str
    message_args: tuple

    def __init__(self, filename: str, loc) -> None:
        self.filename = filename
        self.lineno = loc.lineno
        self.col = getattr(loc, 'col_offset', 0)

    def __str__(self) -> str:
        return '%s:%s:%s %s' % (self.filename, self.lineno, self.col + 1, self.message % self.message_args)

class UnusedImport(Message):
    """Indicates that an import is not used."""
    message = '%r imported but unused'

    def __init__(self, filename: str, loc, name: str) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (name,)

class RedefinedWhileUnused(Message):
    """Indicates that a name is redefined while it's unused."""
    message = 'redefinition of unused %r from line %r'

    def __init__(self, filename: str, loc, name: str, orig_loc) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (name, orig_loc.lineno)

class RedefinedInListComp(Message):
    """Indicates that a name is redefined in a list comprehension."""
    message = 'list comprehension redefines %r from line %r'

    def __init__(self, filename: str, loc, name: str, orig_loc) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (name, orig_loc.lineno)

class ImportShadowedByLoopVar(Message):
    """Indicates that an import is shadowed by a loop variable."""
    message = 'import %r from line %r shadowed by loop variable'

    def __init__(self, filename: str, loc, name: str, orig_loc) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (name, orig_loc.lineno)

class ImportStarNotPermitted(Message):
    """Indicates that a from-import-star is not permitted."""
    message = "'from %s import *' only allowed at module level"

    def __init__(self, filename: str, loc, modname: str) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (modname,)

class ImportStarUsed(Message):
    """Indicates that a from-import-star is used."""
    message = "'from %s import *' used; unable to detect undefined names"

    def __init__(self, filename: str, loc, modname: str) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (modname,)

class ImportStarUsage(Message):
    """Indicates that a name may be undefined."""
    message = '%r may be undefined, or defined from star imports: %s'

    def __init__(self, filename: str, loc, name: str, from_list) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (name, from_list)

class UndefinedName(Message):
    """Indicates that a name is undefined."""
    message = 'undefined name %r'

    def __init__(self, filename: str, loc, name: str) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (name,)

class DoctestSyntaxError(Message):
    """Indicates a syntax error in a doctest."""
    message = 'syntax error in doctest'

    def __init__(self, filename: str, loc, position=None) -> None:
        Message.__init__(self, filename, loc)
        if position:
            self.lineno, self.col = position
        self.message_args = ()

class UndefinedExport(Message):
    """Indicates that a name is undefined in __all__."""
    message = 'undefined name %r in __all__'

    def __init__(self, filename: str, loc, name: str) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (name,)

class UndefinedLocal(Message):
    """Indicates that a local variable is referenced before assignment."""
    message = 'local variable %r {0} referenced before assignment'
    default = 'defined in enclosing scope on line %r'
    builtin = 'defined as a builtin'

    def __init__(self, filename: str, loc, name: str, orig_loc=None) -> None:
        Message.__init__(self, filename, loc)
        if orig_loc is None:
            self.message = self.message.format(self.builtin)
            self.message_args = name
        else:
            self.message = self.message.format(self.default)
            self.message_args = (name, orig_loc.lineno)

class DuplicateArgument(Message):
    """Indicates that a duplicate argument is in a function definition."""
    message = 'duplicate argument %r in function definition'

    def __init__(self, filename: str, loc, name: str) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (name,)

class MultiValueRepeatedKeyLiteral(Message):
    """Indicates that a dictionary key is repeated with different values."""
    message = 'dictionary key %r repeated with different values'

    def __init__(self, filename: str, loc, key) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (key,)

class MultiValueRepeatedKeyVariable(Message):
    """Indicates that a dictionary key variable is repeated with different values."""
    message = 'dictionary key variable %s repeated with different values'

    def __init__(self, filename: str, loc, key) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (key,)

class LateFutureImport(Message):
    """Indicates that a from __future__ import is not at the beginning of the file."""
    message = 'from __future__ imports must occur at the beginning of the file'

    def __init__(self, filename: str, loc, names) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = ()

class FutureFeatureNotDefined(Message):
    """An undefined __future__ feature name was imported."""
    message = 'future feature %s is not defined'

    def __init__(self, filename: str, loc, name: str) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (name,)

class UnusedVariable(Message):
    """Indicates that a variable has been explicitly assigned to but not actually used."""
    message = 'local variable %r is assigned to but never used'

    def __init__(self, filename: str, loc, names) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (names,)

class ReturnWithArgsInsideGenerator(Message):
    """Indicates a return statement with arguments inside a generator."""
    message = "'return' with argument inside generator"

class ReturnOutsideFunction(Message):
    """Indicates a return statement outside of a function/method."""
    message = "'return' outside function"

class YieldOutsideFunction(Message):
    """Indicates a yield or yield from statement outside of a function/method."""
    message = "'yield' outside function"

class ContinueOutsideLoop(Message):
    """Indicates a continue statement outside of a while or for loop."""
    message = "'continue' not properly in loop"

class BreakOutsideLoop(Message):
    """Indicates a break statement outside of a while or for loop."""
    message = "'break' outside loop"

class ContinueInFinally(Message):
    """Indicates a continue statement in a finally block in a while or for loop."""
    message = "'continue' not supported inside 'finally' clause"

class DefaultExceptNotLast(Message):
    """Indicates an except: block as not the last exception handler."""
    message = "default 'except:' must be last"

class TwoStarredExpressions(Message):
    """Indicates two or more starred expressions in an assignment."""
    message = 'two starred expressions in assignment'

class TooManyExpressionsInStarredAssignment(Message):
    """Indicates too many expressions in an assignment with star-unpacking."""
    message = 'too many expressions in star-unpacking assignment'

class IfTuple(Message):
    """Indicates a conditional test is a non-empty tuple literal, which are always True."""
    message = "'if tuple literal' is always true, perhaps remove accidental comma?"

class AssertTuple(Message):
    """Indicates an assertion test is a non-empty tuple literal, which are always True."""
    message = 'assertion is always true, perhaps remove parentheses?'

class ForwardAnnotationSyntaxError(Message):
    """Indicates a syntax error in a forward annotation."""
    message = 'syntax error in forward annotation %r'

    def __init__(self, filename: str, loc, annotation) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (annotation,)

class CommentAnnotationSyntaxError(Message):
    """Indicates a syntax error in a type comment."""
    message = 'syntax error in type comment %r'

    def __init__(self, filename: str, loc, annotation) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (annotation,)

class RaiseNotImplemented(Message):
    """Indicates that 'raise NotImplemented' should be 'raise NotImplementedError'."""
    message = "'raise NotImplemented' should be 'raise NotImplementedError'"

class InvalidPrintSyntax(Message):
    """Indicates the use of >> is invalid with print function."""
    message = 'use of >> is invalid with print function'

class IsLiteral(Message):
    """Indicates the use of ==/!= to compare constant literals."""
    message = 'use ==/!= to compare constant literals (str, bytes, int, float, tuple)'

class FStringMissingPlaceholders(Message):
    """Indicates an f-string is missing placeholders."""
    message = 'f-string is missing placeholders'

class StringDotFormatExtraPositionalArguments(Message):
    """Indicates that a string-dot-format has unused positional arguments."""
    message = "'...'.format(...) has unused arguments at position(s): %s"

    def __init__(self, filename: str, loc, extra_positions) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (extra_positions,)

class StringDotFormatExtraNamedArguments(Message):
    """Indicates that a string-dot-format has unused named arguments."""
    message = "'...'.format(...) has unused named argument(s): %s"

    def __init__(self, filename: str, loc, extra_keywords) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (extra_keywords,)

class StringDotFormatMissingArgument(Message):
    """Indicates that a string-dot-format is missing arguments for placeholders."""
    message = "'...'.format(...) is missing argument(s) for placeholder(s): %s"

    def __init__(self, filename: str, loc, missing_arguments) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (missing_arguments,)

class StringDotFormatMixingAutomatic(Message):
    """Indicates that a string-dot-format mixes automatic and manual numbering."""
    message = "'...'.format(...) mixes automatic and manual numbering"

class StringDotFormatInvalidFormat(Message):
    """Indicates that a string-dot-format has an invalid format string."""
    message = "'...'.format(...) has invalid format string: %s"

    def __init__(self, filename: str, loc, error) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (error,)

class PercentFormatInvalidFormat(Message):
    """Indicates that a percent-format has an invalid format string."""
    message = "'...' %% ... has invalid format string: %s"

    def __init__(self, filename: str, loc, error) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (error,)

class PercentFormatMixedPositionalAndNamed(Message):
    """Indicates that a percent-format has mixed positional and named placeholders."""
    message = "'...' %% ... has mixed positional and named placeholders"

class PercentFormatUnsupportedFormatCharacter(Message):
    """Indicates that a percent-format has an unsupported format character."""
    message = "'...' %% ... has unsupported format character %r"

    def __init__(self, filename: str, loc, c) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (c,)

class PercentFormatPositionalCountMismatch(Message):
    """Indicates that a percent-format has a positional count mismatch."""
    message = "'...' %% ... has %d placeholder(s) but %d substitution(s)"

    def __init__(self, filename: str, loc, n_placeholders: int, n_substitutions: int) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (n_placeholders, n_substitutions)

class PercentFormatExtraNamedArguments(Message):
    """Indicates that a percent-format has unused named arguments."""
    message = "'...' %% ... has unused named argument(s): %s"

    def __init__(self, filename: str, loc, extra_keywords) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (extra_keywords,)

class PercentFormatMissingArgument(Message):
    """Indicates that a percent-format is missing arguments for placeholders."""
    message = "'...' %% ... is missing argument(s) for placeholder(s): %s"

    def __init__(self, filename: str, loc, missing_arguments) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (missing_arguments,)

class PercentFormatExpectedMapping(Message):
    """Indicates that a percent-format expected a mapping but got a sequence."""
    message = "'...' %% ... expected mapping but got sequence"

class PercentFormatExpectedSequence(Message):
    """Indicates that a percent-format expected a sequence but got a mapping."""
    message = "'...' %% ... expected sequence but got mapping"

class PercentFormatStarRequiresSequence(Message):
    """Indicates that a percent-format's star specifier requires a sequence."""
    message = "'...' %% ... `*` specifier requires sequence"
