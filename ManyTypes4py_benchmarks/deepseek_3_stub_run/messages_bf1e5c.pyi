"""
Provide the class Message and its subclasses.
"""

from typing import Any, Optional, Tuple

class Message:
    message: str = ...
    message_args: Tuple[Any, ...] = ...
    
    def __init__(self, filename: str, loc: Any) -> None: ...
    def __str__(self) -> str: ...

class UnusedImport(Message):
    def __init__(self, filename: str, loc: Any, name: str) -> None: ...

class RedefinedWhileUnused(Message):
    def __init__(self, filename: str, loc: Any, name: str, orig_loc: Any) -> None: ...

class RedefinedInListComp(Message):
    def __init__(self, filename: str, loc: Any, name: str, orig_loc: Any) -> None: ...

class ImportShadowedByLoopVar(Message):
    def __init__(self, filename: str, loc: Any, name: str, orig_loc: Any) -> None: ...

class ImportStarNotPermitted(Message):
    def __init__(self, filename: str, loc: Any, modname: str) -> None: ...

class ImportStarUsed(Message):
    def __init__(self, filename: str, loc: Any, modname: str) -> None: ...

class ImportStarUsage(Message):
    def __init__(self, filename: str, loc: Any, name: str, from_list: str) -> None: ...

class UndefinedName(Message):
    def __init__(self, filename: str, loc: Any, name: str) -> None: ...

class DoctestSyntaxError(Message):
    def __init__(self, filename: str, loc: Any, position: Optional[Tuple[int, int]] = None) -> None: ...

class UndefinedExport(Message):
    def __init__(self, filename: str, loc: Any, name: str) -> None: ...

class UndefinedLocal(Message):
    default: str = ...
    builtin: str = ...
    
    def __init__(self, filename: str, loc: Any, name: str, orig_loc: Optional[Any]) -> None: ...

class DuplicateArgument(Message):
    def __init__(self, filename: str, loc: Any, name: str) -> None: ...

class MultiValueRepeatedKeyLiteral(Message):
    def __init__(self, filename: str, loc: Any, key: str) -> None: ...

class MultiValueRepeatedKeyVariable(Message):
    def __init__(self, filename: str, loc: Any, key: str) -> None: ...

class LateFutureImport(Message):
    def __init__(self, filename: str, loc: Any, names: Any) -> None: ...

class FutureFeatureNotDefined(Message):
    def __init__(self, filename: str, loc: Any, name: str) -> None: ...

class UnusedVariable(Message):
    def __init__(self, filename: str, loc: Any, names: str) -> None: ...

class ReturnWithArgsInsideGenerator(Message):
    pass

class ReturnOutsideFunction(Message):
    pass

class YieldOutsideFunction(Message):
    pass

class ContinueOutsideLoop(Message):
    pass

class BreakOutsideLoop(Message):
    pass

class ContinueInFinally(Message):
    pass

class DefaultExceptNotLast(Message):
    pass

class TwoStarredExpressions(Message):
    pass

class TooManyExpressionsInStarredAssignment(Message):
    pass

class IfTuple(Message):
    pass

class AssertTuple(Message):
    pass

class ForwardAnnotationSyntaxError(Message):
    def __init__(self, filename: str, loc: Any, annotation: str) -> None: ...

class CommentAnnotationSyntaxError(Message):
    def __init__(self, filename: str, loc: Any, annotation: str) -> None: ...

class RaiseNotImplemented(Message):
    pass

class InvalidPrintSyntax(Message):
    pass

class IsLiteral(Message):
    pass

class FStringMissingPlaceholders(Message):
    pass

class StringDotFormatExtraPositionalArguments(Message):
    def __init__(self, filename: str, loc: Any, extra_positions: str) -> None: ...

class StringDotFormatExtraNamedArguments(Message):
    def __init__(self, filename: str, loc: Any, extra_keywords: str) -> None: ...

class StringDotFormatMissingArgument(Message):
    def __init__(self, filename: str, loc: Any, missing_arguments: str) -> None: ...

class StringDotFormatMixingAutomatic(Message):
    pass

class StringDotFormatInvalidFormat(Message):
    def __init__(self, filename: str, loc: Any, error: str) -> None: ...

class PercentFormatInvalidFormat(Message):
    def __init__(self, filename: str, loc: Any, error: str) -> None: ...

class PercentFormatMixedPositionalAndNamed(Message):
    pass

class PercentFormatUnsupportedFormatCharacter(Message):
    def __init__(self, filename: str, loc: Any, c: str) -> None: ...

class PercentFormatPositionalCountMismatch(Message):
    def __init__(self, filename: str, loc: Any, n_placeholders: int, n_substitutions: int) -> None: ...

class PercentFormatExtraNamedArguments(Message):
    def __init__(self, filename: str, loc: Any, extra_keywords: str) -> None: ...

class PercentFormatMissingArgument(Message):
    def __init__(self, filename: str, loc: Any, missing_arguments: str) -> None: ...

class PercentFormatExpectedMapping(Message):
    pass

class PercentFormatExpectedSequence(Message):
    pass

class PercentFormatStarRequiresSequence(Message):
    pass