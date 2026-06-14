from typing import Optional, Protocol, Tuple


class _Loc(Protocol):
    lineno: int


class Message(object):
    message: str
    message_args: object
    filename: str
    lineno: int
    col: int
    def __init__(self, filename: str, loc: _Loc) -> None: ...
    def __str__(self) -> str: ...


class UnusedImport(Message):
    def __init__(self, filename: str, loc: _Loc, name: str) -> None: ...


class RedefinedWhileUnused(Message):
    def __init__(self, filename: str, loc: _Loc, name: str, orig_loc: _Loc) -> None: ...


class RedefinedInListComp(Message):
    def __init__(self, filename: str, loc: _Loc, name: str, orig_loc: _Loc) -> None: ...


class ImportShadowedByLoopVar(Message):
    def __init__(self, filename: str, loc: _Loc, name: str, orig_loc: _Loc) -> None: ...


class ImportStarNotPermitted(Message):
    def __init__(self, filename: str, loc: _Loc, modname: str) -> None: ...


class ImportStarUsed(Message):
    def __init__(self, filename: str, loc: _Loc, modname: str) -> None: ...


class ImportStarUsage(Message):
    def __init__(self, filename: str, loc: _Loc, name: str, from_list: str) -> None: ...


class UndefinedName(Message):
    def __init__(self, filename: str, loc: _Loc, name: str) -> None: ...


class DoctestSyntaxError(Message):
    def __init__(self, filename: str, loc: _Loc, position: Optional[Tuple[int, int]] = ...) -> None: ...


class UndefinedExport(Message):
    def __init__(self, filename: str, loc: _Loc, name: str) -> None: ...


class UndefinedLocal(Message):
    message: str
    default: str
    builtin: str
    def __init__(self, filename: str, loc: _Loc, name: str, orig_loc: Optional[_Loc]) -> None: ...


class DuplicateArgument(Message):
    def __init__(self, filename: str, loc: _Loc, name: str) -> None: ...


class MultiValueRepeatedKeyLiteral(Message):
    def __init__(self, filename: str, loc: _Loc, key: object) -> None: ...


class MultiValueRepeatedKeyVariable(Message):
    def __init__(self, filename: str, loc: _Loc, key: str) -> None: ...


class LateFutureImport(Message):
    def __init__(self, filename: str, loc: _Loc, names: object) -> None: ...


class FutureFeatureNotDefined(Message):
    def __init__(self, filename: str, loc: _Loc, name: str) -> None: ...


class UnusedVariable(Message):
    def __init__(self, filename: str, loc: _Loc, names: str) -> None: ...


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
    def __init__(self, filename: str, loc: _Loc, annotation: str) -> None: ...


class CommentAnnotationSyntaxError(Message):
    def __init__(self, filename: str, loc: _Loc, annotation: str) -> None: ...


class RaiseNotImplemented(Message): ...


class InvalidPrintSyntax(Message): ...


class IsLiteral(Message): ...


class FStringMissingPlaceholders(Message): ...


class StringDotFormatExtraPositionalArguments(Message):
    def __init__(self, filename: str, loc: _Loc, extra_positions: str) -> None: ...


class StringDotFormatExtraNamedArguments(Message):
    def __init__(self, filename: str, loc: _Loc, extra_keywords: str) -> None: ...


class StringDotFormatMissingArgument(Message):
    def __init__(self, filename: str, loc: _Loc, missing_arguments: str) -> None: ...


class StringDotFormatMixingAutomatic(Message): ...


class StringDotFormatInvalidFormat(Message):
    def __init__(self, filename: str, loc: _Loc, error: object) -> None: ...


class PercentFormatInvalidFormat(Message):
    def __init__(self, filename: str, loc: _Loc, error: object) -> None: ...


class PercentFormatMixedPositionalAndNamed(Message): ...


class PercentFormatUnsupportedFormatCharacter(Message):
    def __init__(self, filename: str, loc: _Loc, c: str) -> None: ...


class PercentFormatPositionalCountMismatch(Message):
    def __init__(self, filename: str, loc: _Loc, n_placeholders: int, n_substitutions: int) -> None: ...


class PercentFormatExtraNamedArguments(Message):
    def __init__(self, filename: str, loc: _Loc, extra_keywords: str) -> None: ...


class PercentFormatMissingArgument(Message):
    def __init__(self, filename: str, loc: _Loc, missing_arguments: str) -> None: ...


class PercentFormatExpectedMapping(Message): ...


class PercentFormatExpectedSequence(Message): ...


class PercentFormatStarRequiresSequence(Message): ...