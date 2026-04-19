from typing import Any, Optional, Protocol, Tuple


class _HasLineno(Protocol):
    lineno: int


class Message(object):
    message: str
    message_args: object
    filename: str
    lineno: int
    col: int

    def __init__(self, filename: str, loc: _HasLineno) -> None: ...
    def __str__(self) -> str: ...


class UnusedImport(Message):
    def __init__(self, filename: str, loc: _HasLineno, name: str) -> None: ...


class RedefinedWhileUnused(Message):
    def __init__(self, filename: str, loc: _HasLineno, name: str, orig_loc: _HasLineno) -> None: ...


class RedefinedInListComp(Message):
    def __init__(self, filename: str, loc: _HasLineno, name: str, orig_loc: _HasLineno) -> None: ...


class ImportShadowedByLoopVar(Message):
    def __init__(self, filename: str, loc: _HasLineno, name: str, orig_loc: _HasLineno) -> None: ...


class ImportStarNotPermitted(Message):
    def __init__(self, filename: str, loc: _HasLineno, modname: str) -> None: ...


class ImportStarUsed(Message):
    def __init__(self, filename: str, loc: _HasLineno, modname: str) -> None: ...


class ImportStarUsage(Message):
    def __init__(self, filename: str, loc: _HasLineno, name: str, from_list: str) -> None: ...


class UndefinedName(Message):
    def __init__(self, filename: str, loc: _HasLineno, name: str) -> None: ...


class DoctestSyntaxError(Message):
    def __init__(self, filename: str, loc: _HasLineno, position: Optional[Tuple[int, int]] = ...) -> None: ...


class UndefinedExport(Message):
    def __init__(self, filename: str, loc: _HasLineno, name: str) -> None: ...


class UndefinedLocal(Message):
    default: str
    builtin: str

    def __init__(self, filename: str, loc: _HasLineno, name: str, orig_loc: Optional[_HasLineno]) -> None: ...


class DuplicateArgument(Message):
    def __init__(self, filename: str, loc: _HasLineno, name: str) -> None: ...


class MultiValueRepeatedKeyLiteral(Message):
    def __init__(self, filename: str, loc: _HasLineno, key: Any) -> None: ...


class MultiValueRepeatedKeyVariable(Message):
    def __init__(self, filename: str, loc: _HasLineno, key: str) -> None: ...


class LateFutureImport(Message):
    def __init__(self, filename: str, loc: _HasLineno, names: Any) -> None: ...


class FutureFeatureNotDefined(Message):
    def __init__(self, filename: str, loc: _HasLineno, name: str) -> None: ...


class UnusedVariable(Message):
    def __init__(self, filename: str, loc: _HasLineno, names: Any) -> None: ...


class ReturnWithArgsInsideGenerator(Message): ...


class ReturnOutsideFunction(Message): ...


class YieldOutsideFunction(Message): ...


class ContinueOutsideLoop(Message): ...


class BreakOutsideLoop(Message): ...


class ContinueInFinally(Message): ...


class DefaultExceptNotLast(Message): ...


class TwoStarredExpressions(Message): ...


class TooManyExpressionsInStarredAssignment(Message): ...


class IfTuple(Message): ...


class AssertTuple(Message): ...


class ForwardAnnotationSyntaxError(Message):
    def __init__(self, filename: str, loc: _HasLineno, annotation: str) -> None: ...


class CommentAnnotationSyntaxError(Message):
    def __init__(self, filename: str, loc: _HasLineno, annotation: str) -> None: ...


class RaiseNotImplemented(Message): ...


class InvalidPrintSyntax(Message): ...


class IsLiteral(Message): ...


class FStringMissingPlaceholders(Message): ...


class StringDotFormatExtraPositionalArguments(Message):
    def __init__(self, filename: str, loc: _HasLineno, extra_positions: str) -> None: ...


class StringDotFormatExtraNamedArguments(Message):
    def __init__(self, filename: str, loc: _HasLineno, extra_keywords: str) -> None: ...


class StringDotFormatMissingArgument(Message):
    def __init__(self, filename: str, loc: _HasLineno, missing_arguments: str) -> None: ...


class StringDotFormatMixingAutomatic(Message): ...


class StringDotFormatInvalidFormat(Message):
    def __init__(self, filename: str, loc: _HasLineno, error: Any) -> None: ...


class PercentFormatInvalidFormat(Message):
    def __init__(self, filename: str, loc: _HasLineno, error: Any) -> None: ...


class PercentFormatMixedPositionalAndNamed(Message): ...


class PercentFormatUnsupportedFormatCharacter(Message):
    def __init__(self, filename: str, loc: _HasLineno, c: str) -> None: ...


class PercentFormatPositionalCountMismatch(Message):
    def __init__(self, filename: str, loc: _HasLineno, n_placeholders: int, n_substitutions: int) -> None: ...


class PercentFormatExtraNamedArguments(Message):
    def __init__(self, filename: str, loc: _HasLineno, extra_keywords: str) -> None: ...


class PercentFormatMissingArgument(Message):
    def __init__(self, filename: str, loc: _HasLineno, missing_arguments: str) -> None: ...


class PercentFormatExpectedMapping(Message): ...


class PercentFormatExpectedSequence(Message): ...


class PercentFormatStarRequiresSequence(Message): ...