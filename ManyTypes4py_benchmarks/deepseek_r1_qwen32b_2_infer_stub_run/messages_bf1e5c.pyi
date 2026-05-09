"""
Stub file for messages_bf1e5c module
"""

from typing import Any, Optional, Tuple, List, Union

class Message:
    message: str
    message_args: Tuple[Any, ...]

    def __init__(self, filename: str, loc: Any) -> None:
        ...

    def __str__(self) -> str:
        ...


class UnusedImport(Message):
    message: str

    def __init__(self, filename: str, loc: Any, name: str) -> None:
        ...


class RedefinedWhileUnused(Message):
    message: str

    def __init__(self, filename: str, loc: Any, name: str, orig_loc: Any) -> None:
        ...


class RedefinedInListComp(Message):
    message: str

    def __init__(self, filename: str, loc: Any, name: str, orig_loc: Any) -> None:
        ...


class ImportShadowedByLoopVar(Message):
    message: str

    def __init__(self, filename: str, loc: Any, name: str, orig_loc: Any) -> None:
        ...


class ImportStarNotPermitted(Message):
    message: str

    def __init__(self, filename: str, loc: Any, modname: str) -> None:
        ...


class ImportStarUsed(Message):
    message: str

    def __init__(self, filename: str, loc: Any, modname: str) -> None:
        ...


class ImportStarUsage(Message):
    message: str

    def __init__(self, filename: str, loc: Any, name: str, from_list: List[str]) -> None:
        ...


class UndefinedName(Message):
    message: str

    def __init__(self, filename: str, loc: Any, name: str) -> None:
        ...


class DoctestSyntaxError(Message):
    message: str

    def __init__(self, filename: str, loc: Any, position: Optional[Tuple[int, int]] = None) -> None:
        ...


class UndefinedExport(Message):
    message: str

    def __init__(self, filename: str, loc: Any, name: str) -> None:
        ...


class UndefinedLocal(Message):
    message: str
    default: str
    builtin: str

    def __init__(self, filename: str, loc: Any, name: str, orig_loc: Optional[Any]) -> None:
        ...


class DuplicateArgument(Message):
    message: str

    def __init__(self, filename: str, loc: Any, name: str) -> None:
        ...


class MultiValueRepeatedKeyLiteral(Message):
    message: str

    def __init__(self, filename: str, loc: Any, key: str) -> None:
        ...


class MultiValueRepeatedKeyVariable(Message):
    message: str

    def __init__(self, filename: str, loc: Any, key: str) -> None:
        ...


class LateFutureImport(Message):
    message: str

    def __init__(self, filename: str, loc: Any, names: Any) -> None:
        ...


class FutureFeatureNotDefined(Message):
    message: str

    def __init__(self, filename: str, loc: Any, name: str) -> None:
        ...


class UnusedVariable(Message):
    message: str

    def __init__(self, filename: str, loc: Any, names: str) -> None:
        ...


class ReturnWithArgsInsideGenerator(Message):
    message: str

    def __init__(self, filename: str, loc: Any) -> None:
        ...


class ReturnOutsideFunction(Message):
    message: str

    def __init__(self, filename: str, loc: Any) -> None:
        ...


class YieldOutsideFunction(Message):
    message: str

    def __init__(self, filename: str, loc: Any) -> None:
        ...


class ContinueOutsideLoop(Message):
    message: str

    def __init__(self, filename: str, loc: Any) -> None:
        ...


class BreakOutsideLoop(Message):
    message: str

    def __init__(self, filename: str, loc: Any) -> None:
        ...


class ContinueInFinally(Message):
    message: str

    def __init__(self, filename: str, loc: Any) -> None:
        ...


class DefaultExceptNotLast(Message):
    message: str

    def __init__(self, filename: str, loc: Any) -> None:
        ...


class TwoStarredExpressions(Message):
    message: str

    def __init__(self, filename: str, loc: Any) -> None:
        ...


class TooManyExpressionsInStarredAssignment(Message):
    message: str

    def __init__(self, filename: str, loc: Any) -> None:
        ...


class IfTuple(Message):
    message: str

    def __init__(self, filename: str, loc: Any) -> None:
        ...


class AssertTuple(Message):
    message: str

    def __init__(self, filename: str, loc: Any) -> None:
        ...


class ForwardAnnotationSyntaxError(Message):
    message: str

    def __init__(self, filename: str, loc: Any, annotation: str) -> None:
        ...


class CommentAnnotationSyntaxError(Message):
    message: str

    def __init__(self, filename: str, loc: Any, annotation: str) -> None:
        ...


class RaiseNotImplemented(Message):
    message: str

    def __init__(self, filename: str, loc: Any) -> None:
        ...


class InvalidPrintSyntax(Message):
    message: str

    def __init__(self, filename: str, loc: Any) -> None:
        ...


class IsLiteral(Message):
    message: str

    def __init__(self, filename: str, loc: Any) -> None:
        ...


class FStringMissingPlaceholders(Message):
    message: str

    def __init__(self, filename: str, loc: Any) -> None:
        ...


class StringDotFormatExtraPositionalArguments(Message):
    message: str

    def __init__(self, filename: str, loc: Any, extra_positions: str) -> None:
        ...


class StringDotFormatExtraNamedArguments(Message):
    message: str

    def __init__(self, filename: str, loc: Any, extra_keywords: str) -> None:
        ...


class StringDotFormatMissingArgument(Message):
    message: str

    def __init__(self, filename: str, loc: Any, missing_arguments: str) -> None:
        ...


class StringDotFormatMixingAutomatic(Message):
    message: str

    def __init__(self, filename: str, loc: Any) -> None:
        ...


class StringDotFormatInvalidFormat(Message):
    message: str

    def __init__(self, filename: str, loc: Any, error: str) -> None:
        ...


class PercentFormatInvalidFormat(Message):
    message: str

    def __init__(self, filename: str, loc: Any, error: str) -> None:
        ...


class PercentFormatMixedPositionalAndNamed(Message):
    message: str

    def __init__(self, filename: str, loc: Any) -> None:
        ...


class PercentFormatUnsupportedFormatCharacter(Message):
    message: str

    def __init__(self, filename: str, loc: Any, c: str) -> None:
        ...


class PercentFormatPositionalCountMismatch(Message):
    message: str

    def __init__(self, filename: str, loc: Any, n_placeholders: int, n_substitutions: int) -> None:
        ...


class PercentFormatExtraNamedArguments(Message):
    message: str

    def __init__(self, filename: str, loc: Any, extra_keywords: str) -> None:
        ...


class PercentFormatMissingArgument(Message):
    message: str

    def __init__(self, filename: str, loc: Any, missing_arguments: str) -> None:
        ...


class PercentFormatExpectedMapping(Message):
    message: str

    def __init__(self, filename: str, loc: Any) -> None:
        ...


class PercentFormatExpectedSequence(Message):
    message: str

    def __init__(self, filename: str, loc: Any) -> None:
        ...


class PercentFormatStarRequiresSequence(Message):
    message: str

    def __init__(self, filename: str, loc: Any) -> None:
        ...