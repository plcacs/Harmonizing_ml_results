from typing import Tuple

class Message:
    message: str
    message_args: Tuple

    def __init__(self, filename: str, loc: Any) -> None:
        self.filename: str = filename
        self.lineno: int = loc.lineno
        self.col: int = getattr(loc, 'col_offset', 0)

    def __str__(self) -> str:
        return '%s:%s: %s' % (self.filename, self.lineno, self.message % self.message_args)

class UnusedImport(Message):
    def __init__(self, filename: str, loc: Any, name: str) -> None:
        ...

class RedefinedWhileUnused(Message):
    def __init__(self, filename: str, loc: Any, name: str, orig_loc: Any) -> None:
        ...

class RedefinedInListComp(Message):
    def __init__(self, filename: str, loc: Any, name: str, orig_loc: Any) -> None:
        ...

class ImportShadowedByLoopVar(Message):
    def __init__(self, filename: str, loc: Any, name: str, orig_loc: Any) -> None:
        ...

class ImportStarUsed(Message):
    def __init__(self, filename: str, loc: Any, modname: str) -> None:
        ...

class UndefinedName(Message):
    def __init__(self, filename: str, loc: Any, name: str) -> None:
        ...

class DoctestSyntaxError(Message):
    def __init__(self, filename: str, loc: Any, position: Tuple[int, int]) -> None:
        ...

class UndefinedExport(Message):
    def __init__(self, filename: str, loc: Any, name: str) -> None:
        ...

class UndefinedLocal(Message):
    def __init__(self, filename: str, loc: Any, name: str, orig_loc: Any) -> None:
        ...

class DuplicateArgument(Message):
    def __init__(self, filename: str, loc: Any, name: str) -> None:
        ...

class LateFutureImport(Message):
    def __init__(self, filename: str, loc: Any, names: Tuple[str]) -> None:
        ...

class UnusedVariable(Message):
    def __init__(self, filename: str, loc: Any, names: Tuple[str]) -> None:
        ...

class ReturnWithArgsInsideGenerator(Message):
    ...

class ReturnOutsideFunction(Message):
    ...
