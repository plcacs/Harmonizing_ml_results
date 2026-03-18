from typing import Any

class Message(object):
    message: str
    message_args: Any
    filename: Any
    lineno: Any
    col: Any
    def __init__(self, filename: Any, loc: Any) -> None: ...
    def __str__(self) -> str: ...

class UnusedImport(Message):
    message: str
    def __init__(self, filename: Any, loc: Any, name: Any) -> None: ...

class RedefinedWhileUnused(Message):
    message: str
    def __init__(self, filename: Any, loc: Any, name: Any, orig_loc: Any) -> None: ...

class RedefinedInListComp(Message):
    message: str
    def __init__(self, filename: Any, loc: Any, name: Any, orig_loc: Any) -> None: ...

class ImportShadowedByLoopVar(Message):
    message: str
    def __init__(self, filename: Any, loc: Any, name: Any, orig_loc: Any) -> None: ...

class ImportStarNotPermitted(Message):
    message: str
    def __init__(self, filename: Any, loc: Any, modname: Any) -> None: ...

class ImportStarUsed(Message):
    message: str
    def __init__(self, filename: Any, loc: Any, modname: Any) -> None: ...

class ImportStarUsage(Message):
    message: str
    def __init__(self, filename: Any, loc: Any, name: Any, from_list: Any) -> None: ...

class UndefinedName(Message):
    message: str
    def __init__(self, filename: Any, loc: Any, name: Any) -> None: ...

class DoctestSyntaxError(Message):
    message: str
    def __init__(self, filename: Any, loc: Any, position: Any = ...) -> None: ...

class UndefinedExport(Message):
    message: str
    def __init__(self, filename: Any, loc: Any, name: Any) -> None: ...

class UndefinedLocal(Message):
    message: str
    default: str
    builtin: str
    def __init__(self, filename: Any, loc: Any, name: Any, orig_loc: Any) -> None: ...

class DuplicateArgument(Message):
    message: str
    def __init__(self, filename: Any, loc: Any, name: Any) -> None: ...

class MultiValueRepeatedKeyLiteral(Message):
    message: str
    def __init__(self, filename: Any, loc: Any, key: Any) -> None: ...

class MultiValueRepeatedKeyVariable(Message):
    message: str
    def __init__(self, filename: Any, loc: Any, key: Any) -> None: ...

class LateFutureImport(Message):
    message: str
    def __init__(self, filename: Any, loc: Any, names: Any) -> None: ...

class FutureFeatureNotDefined(Message):
    message: str
    def __init__(self, filename: Any, loc: Any, name: Any) -> None: ...

class UnusedVariable(Message):
    message: str
    def __init__(self, filename: Any, loc: Any, names: Any) -> None: ...

class ReturnWithArgsInsideGenerator(Message):
    message: str

class ReturnOutsideFunction(Message):
    message: str

class YieldOutsideFunction(Message):
    message: str

class ContinueOutsideLoop(Message):
    message: str

class BreakOutsideLoop(Message):
    message: str

class ContinueInFinally(Message):
    message: str

class DefaultExceptNotLast(Message):
    message: str

class TwoStarredExpressions(Message):
    message: str

class TooManyExpressionsInStarredAssignment(Message):
    message: str

class IfTuple(Message):
    message: str

class AssertTuple(Message):
    message: str

class ForwardAnnotationSyntaxError(Message):
    message: str
    def __init__(self, filename: Any, loc: Any, annotation: Any) -> None: ...

class CommentAnnotationSyntaxError(Message):
    message: str
    def __init__(self, filename: Any, loc: Any, annotation: Any) -> None: ...

class RaiseNotImplemented(Message):
    message: str

class InvalidPrintSyntax(Message):
    message: str

class IsLiteral(Message):
    message: str

class FStringMissingPlaceholders(Message):
    message: str

class StringDotFormatExtraPositionalArguments(Message):
    message: str
    def __init__(self, filename: Any, loc: Any, extra_positions: Any) -> None: ...

class StringDotFormatExtraNamedArguments(Message):
    message: str
    def __init__(self, filename: Any, loc: Any, extra_keywords: Any) -> None: ...

class StringDotFormatMissingArgument(Message):
    message: str
    def __init__(self, filename: Any, loc: Any, missing_arguments: Any) -> None: ...

class StringDotFormatMixingAutomatic(Message):
    message: str

class StringDotFormatInvalidFormat(Message):
    message: str
    def __init__(self, filename: Any, loc: Any, error: Any) -> None: ...

class PercentFormatInvalidFormat(Message):
    message: str
    def __init__(self, filename: Any, loc: Any, error: Any) -> None: ...

class PercentFormatMixedPositionalAndNamed(Message):
    message: str

class PercentFormatUnsupportedFormatCharacter(Message):
    message: str
    def __init__(self, filename: Any, loc: Any, c: Any) -> None: ...

class PercentFormatPositionalCountMismatch(Message):
    message: str
    def __init__(self, filename: Any, loc: Any, n_placeholders: Any, n_substitutions: Any) -> None: ...

class PercentFormatExtraNamedArguments(Message):
    message: str
    def __init__(self, filename: Any, loc: Any, extra_keywords: Any) -> None: ...

class PercentFormatMissingArgument(Message):
    message: str
    def __init__(self, filename: Any, loc: Any, missing_arguments: Any) -> None: ...

class PercentFormatExpectedMapping(Message):
    message: str

class PercentFormatExpectedSequence(Message):
    message: str

class PercentFormatStarRequiresSequence(Message):
    message: str