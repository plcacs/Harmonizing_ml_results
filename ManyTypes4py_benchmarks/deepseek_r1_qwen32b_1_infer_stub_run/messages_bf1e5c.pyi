"""
Stub file for 'messages_bf1e5c' module.
"""

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
    overload,
)

class Message:
    message: str
    message_args: Tuple[Any, ...]
    
    def __init__(self, filename: str, loc: object) -> None: ...
    
    def __str__(self) -> str: ...

class UnusedImport(Message):
    message: str
    
    def __init__(self, filename: str, loc: object, name: str) -> None: ...

class RedefinedWhileUnused(Message):
    message: str
    
    def __init__(self, filename: str, loc: object, name: str, orig_loc: object) -> None: ...

class RedefinedInListComp(Message):
    message: str
    
    def __init__(self, filename: str, loc: object, name: str, orig_loc: object) -> None: ...

class ImportShadowedByLoopVar(Message):
    message: str
    
    def __init__(self, filename: str, loc: object, name: str, orig_loc: object) -> None: ...

class ImportStarNotPermitted(Message):
    message: str
    
    def __init__(self, filename: str, loc: object, modname: str) -> None: ...

class ImportStarUsed(Message):
    message: str
    
    def __init__(self, filename: str, loc: object, modname: str) -> None: ...

class ImportStarUsage(Message):
    message: str
    
    def __init__(self, filename: str, loc: object, name: str, from_list: Iterable[str]) -> None: ...

class UndefinedName(Message):
    message: str
    
    def __init__(self, filename: str, loc: object, name: str) -> None: ...

class DoctestSyntaxError(Message):
    message: str
    
    def __init__(self, filename: str, loc: object, position: Optional[Tuple[int, int]] = None) -> None: ...

class UndefinedExport(Message):
    message: str
    
    def __init__(self, filename: str, loc: object, name: str) -> None: ...

class UndefinedLocal(Message):
    message: str
    default: str
    builtin: str
    
    def __init__(self, filename: str, loc: object, name: str, orig_loc: Optional[object]) -> None: ...

class DuplicateArgument(Message):
    message: str
    
    def __init__(self, filename: str, loc: object, name: str) -> None: ...

class MultiValueRepeatedKeyLiteral(Message):
    message: str
    
    def __init__(self, filename: str, loc: object, key: str) -> None: ...

class MultiValueRepeatedKeyVariable(Message):
    message: str
    
    def __init__(self, filename: str, loc: object, key: str) -> None: ...

class LateFutureImport(Message):
    message: str
    
    def __init__(self, filename: str, loc: object, names: Iterable[str]) -> None: ...

class FutureFeatureNotDefined(Message):
    message: str
    
    def __init__(self, filename: str, loc: object, name: str) -> None: ...

class UnusedVariable(Message):
    message: str
    
    def __init__(self, filename: str, loc: object, names: Iterable[str]) -> None: ...

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
    
    def __init__(self, filename: str, loc: object, annotation: str) -> None: ...

class CommentAnnotationSyntaxError(Message):
    message: str
    
    def __init__(self, filename: str, loc: object, annotation: str) -> None: ...

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
    
    def __init__(self, filename: str, loc: object, extra_positions: Iterable[int]) -> None: ...

class StringDotFormatExtraNamedArguments(Message):
    message: str
    
    def __init__(self, filename: str, loc: object, extra_keywords: Iterable[str]) -> None: ...

class StringDotFormatMissingArgument(Message):
    message: str
    
    def __init__(self, filename: str, loc: object, missing_arguments: Iterable[Union[str, int]]) -> None: ...

class StringDotFormatMixingAutomatic(Message):
    message: str

class StringDotFormatInvalidFormat(Message):
    message: str
    
    def __init__(self, filename: str, loc: object, error: str) -> None: ...

class PercentFormatInvalidFormat(Message):
    message: str
    
    def __init__(self, filename: str, loc: object, error: str) -> None: ...

class PercentFormatMixedPositionalAndNamed(Message):
    message: str

class PercentFormatUnsupportedFormatCharacter(Message):
    message: str
    
    def __init__(self, filename: str, loc: object, c: str) -> None: ...

class PercentFormatPositionalCountMismatch(Message):
    message: str
    
    def __init__(self, filename: str, loc: object, n_placeholders: int, n_substitutions: int) -> None: ...

class PercentFormatExtraNamedArguments(Message):
    message: str
    
    def __init__(self, filename: str, loc: object, extra_keywords: Iterable[str]) -> None: ...

class PercentFormatMissingArgument(Message):
    message: str
    
    def __init__(self, filename: str, loc: object, missing_arguments: Iterable[Union[str, int]]) -> None: ...

class PercentFormatExpectedMapping(Message):
    message: str

class PercentFormatExpectedSequence(Message):
    message: str

class PercentFormatStarRequiresSequence(Message):
    message: str