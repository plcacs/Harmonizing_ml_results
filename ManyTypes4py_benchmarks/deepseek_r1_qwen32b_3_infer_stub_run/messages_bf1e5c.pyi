"""
Provide the class Message and its subclasses.
"""

from typing import Any, Optional, Tuple, Union

class Message(object):
    message: str
    message_args: Tuple[Any, ...]
    filename: str
    lineno: int
    col: int

    def __init__(self, filename: str, loc: object) -> None: ...
    def __str__(self) -> str: ...

class UnusedImport(Message):
    message: str
    message_args: Tuple[str, ...]

    def __init__(self, filename: str, loc: object, name: str) -> None: ...

class RedefinedWhileUnused(Message):
    message: str
    message_args: Tuple[str, int]

    def __init__(self, filename: str, loc: object, name: str, orig_loc: object) -> None: ...

class RedefinedInListComp(Message):
    message: str
    message_args: Tuple[str, int]

    def __init__(self, filename: str, loc: object, name: str, orig_loc: object) -> None: ...

class ImportShadowedByLoopVar(Message):
    message: str
    message_args: Tuple[str, int]

    def __init__(self, filename: str, loc: object, name: str, orig_loc: object) -> None: ...

class ImportStarNotPermitted(Message):
    message: str
    message_args: Tuple[str, ...]

    def __init__(self, filename: str, loc: object, modname: str) -> None: ...

class ImportStarUsed(Message):
    message: str
    message_args: Tuple[str, ...]

    def __init__(self, filename: str, loc: object, modname: str) -> None: ...

class ImportStarUsage(Message):
    message: str
    message_args: Tuple[str, str]

    def __init__(self, filename: str, loc: object, name: str, from_list: str) -> None: ...

class UndefinedName(Message):
    message: str
    message_args: Tuple[str, ...]

    def __init__(self, filename: str, loc: object, name: str) -> None: ...

class DoctestSyntaxError(Message):
    message: str
    message_args: Tuple[Any, ...]

    def __init__(self, filename: str, loc: object, position: Optional[Tuple[int, int]] = None) -> None: ...

class UndefinedExport(Message):
    message: str
    message_args: Tuple[str, ...]

    def __init__(self, filename: str, loc: object, name: str) -> None: ...

class UndefinedLocal(Message):
    message: str
    message_args: Union[Tuple[str], Tuple[str, int]]
    default: str
    builtin: str

    def __init__(self, filename: str, loc: object, name: str, orig_loc: Optional[object]) -> None: ...

class DuplicateArgument(Message):
    message: str
    message_args: Tuple[str, ...]

    def __init__(self, filename: str, loc: object, name: str) -> None: ...

class MultiValueRepeatedKeyLiteral(Message):
    message: str
    message_args: Tuple[Any, ...]

    def __init__(self, filename: str, loc: object, key: Any) -> None: ...

class MultiValueRepeatedKeyVariable(Message):
    message: str
    message_args: Tuple[Any, ...]

    def __init__(self, filename: str, loc: object, key: Any) -> None: ...

class LateFutureImport(Message):
    message: str
    message_args: Tuple[Any, ...]

    def __init__(self, filename: str, loc: object, names: Any) -> None: ...

class FutureFeatureNotDefined(Message):
    message: str
    message_args: Tuple[str, ...]

    def __init__(self, filename: str, loc: object, name: str) -> None: ...

class UnusedVariable(Message):
    message: str
    message_args: Tuple[str, ...]

    def __init__(self, filename: str, loc: object, names: str) -> None: ...

class ReturnWithArgsInsideGenerator(Message):
    message: str
    message_args: Tuple[Any, ...]

    def __init__(self, filename: str, loc: object) -> None: ...

class ReturnOutsideFunction(Message):
    message: str
    message_args: Tuple[Any, ...]

    def __init__(self, filename: str, loc: object) -> None: ...

class YieldOutsideFunction(Message):
    message: str
    message_args: Tuple[Any, ...]

    def __init__(self, filename: str, loc: object) -> None: ...

class ContinueOutsideLoop(Message):
    message: str
    message_args: Tuple[Any, ...]

    def __init__(self, filename: str, loc: object) -> None: ...

class BreakOutsideLoop(Message):
    message: str
    message_args: Tuple[Any, ...]

    def __init__(self, filename: str, loc: object) -> None: ...

class ContinueInFinally(Message):
    message: str
    message_args: Tuple[Any, ...]

    def __init__(self, filename: str, loc: object) -> None: ...

class DefaultExceptNotLast(Message):
    message: str
    message_args: Tuple[Any, ...]

    def __init__(self, filename: str, loc: object) -> None: ...

class TwoStarredExpressions(Message):
    message: str
    message_args: Tuple[Any, ...]

    def __init__(self, filename: str, loc: object) -> None: ...

class TooManyExpressionsInStarredAssignment(Message):
    message: str
    message_args: Tuple[Any, ...]

    def __init__(self, filename: str, loc: object) -> None: ...

class IfTuple(Message):
    message: str
    message_args: Tuple[Any, ...]

    def __init__(self, filename: str, loc: object) -> None: ...

class AssertTuple(Message):
    message: str
    message_args: Tuple[Any, ...]

    def __init__(self, filename: str, loc: object) -> None: ...

class ForwardAnnotationSyntaxError(Message):
    message: str
    message_args: Tuple[str, ...]

    def __init__(self, filename: str, loc: object, annotation: str) -> None: ...

class CommentAnnotationSyntaxError(Message):
    message: str
    message_args: Tuple[str, ...]

    def __init__(self, filename: str, loc: object, annotation: str) -> None: ...

class RaiseNotImplemented(Message):
    message: str
    message_args: Tuple[Any, ...]

    def __init__(self, filename: str, loc: object) -> None: ...

class InvalidPrintSyntax(Message):
    message: str
    message_args: Tuple[Any, ...]

    def __init__(self, filename: str, loc: object) -> None: ...

class IsLiteral(Message):
    message: str
    message_args: Tuple[Any, ...]

    def __init__(self, filename: str, loc: object) -> None: ...

class FStringMissingPlaceholders(Message):
    message: str
    message_args: Tuple[Any, ...]

    def __init__(self, filename: str, loc: object) -> None: ...

class StringDotFormatExtraPositionalArguments(Message):
    message: str
    message_args: Tuple[str, ...]

    def __init__(self, filename: str, loc: object, extra_positions: str) -> None: ...

class StringDotFormatExtraNamedArguments(Message):
    message: str
    message_args: Tuple[str, ...]

    def __init__(self, filename: str, loc: object, extra_keywords: str) -> None: ...

class StringDotFormatMissingArgument(Message):
    message: str
    message_args: Tuple[str, ...]

    def __init__(self, filename: str, loc: object, missing_arguments: str) -> None: ...

class StringDotFormatMixingAutomatic(Message):
    message: str
    message_args: Tuple[Any, ...]

    def __init__(self, filename: str, loc: object) -> None: ...

class StringDotFormatInvalidFormat(Message):
    message: str
    message_args: Tuple[str, ...]

    def __init__(self, filename: str, loc: object, error: str) -> None: ...

class PercentFormatInvalidFormat(Message):
    message: str
    message_args: Tuple[str, ...]

    def __init__(self, filename: str, loc: object, error: str) -> None: ...

class PercentFormatMixedPositionalAndNamed(Message):
    message: str
    message_args: Tuple[Any, ...]

    def __init__(self, filename: str, loc: object) -> None: ...

class PercentFormatUnsupportedFormatCharacter(Message):
    message: str
    message_args: Tuple[str, ...]

    def __init__(self, filename: str, loc: object, c: str) -> None: ...

class PercentFormatPositionalCountMismatch(Message):
    message: str
    message_args: Tuple[int, int]

    def __init__(self, filename: str, loc: object, n_placeholders: int, n_substitutions: int) -> None: ...

class PercentFormatExtraNamedArguments(Message):
    message: str
    message_args: Tuple[str, ...]

    def __init__(self, filename: str, loc: object, extra_keywords: str) -> None: ...

class PercentFormatMissingArgument(Message):
    message: str
    message_args: Tuple[str, ...]

    def __init__(self, filename: str, loc: object, missing_arguments: str) -> None: ...

class PercentFormatExpectedMapping(Message):
    message: str
    message_args: Tuple[Any, ...]

    def __init__(self, filename: str, loc: object) -> None: ...

class PercentFormatExpectedSequence(Message):
    message: str
    message_args: Tuple[Any, ...]

    def __init__(self, filename: str, loc: object) -> None: ...

class PercentFormatStarRequiresSequence(Message):
    message: str
    message_args: Tuple[Any, ...]

    def __init__(self, filename: str, loc: object) -> None: ...