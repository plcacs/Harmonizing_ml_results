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
        super().__init__(filename, loc)
        self.message_args = (name,)

class RedefinedWhileUnused(Message):
    def __init__(self, filename: str, loc: Any, name: str, orig_loc: Any) -> None:
        super().__init__(filename, loc)
        self.message_args = (name, orig_loc.lineno)

class RedefinedInListComp(Message):
    def __init__(self, filename: str, loc: Any, name: str, orig_loc: Any) -> None:
        super().__init__(filename, loc)
        self.message_args = (name, orig_loc.lineno)

class ImportShadowedByLoopVar(Message):
    def __init__(self, filename: str, loc: Any, name: str, orig_loc: Any) -> None:
        super().__init__(filename, loc)
        self.message_args = (name, orig_loc.lineno)

class ImportStarUsed(Message):
    def __init__(self, filename: str, loc: Any, modname: str) -> None:
        super().__init__(filename, loc)
        self.message_args = (modname,)

class UndefinedName(Message):
    def __init__(self, filename: str, loc: Any, name: str) -> None:
        super().__init__(filename, loc)
        self.message_args = (name,)

class DoctestSyntaxError(Message):
    def __init__(self, filename: str, loc: Any, position: Tuple[int, int] = None) -> None:
        super().__init__(filename, loc)
        if position:
            self.lineno, self.col = position
        self.message_args = ()

class UndefinedExport(Message):
    def __init__(self, filename: str, loc: Any, name: str) -> None:
        super().__init__(filename, loc)
        self.message_args = (name,)

class UndefinedLocal(Message):
    def __init__(self, filename: str, loc: Any, name: str, orig_loc: Any) -> None:
        super().__init__(filename, loc)
        self.message_args = (name, orig_loc.lineno)

class DuplicateArgument(Message):
    def __init__(self, filename: str, loc: Any, name: str) -> None:
        super().__init__(filename, loc)
        self.message_args = (name,)

class LateFutureImport(Message):
    def __init__(self, filename: str, loc: Any, names: Tuple[str]) -> None:
        super().__init__(filename, loc)
        self.message_args = (names,)

class UnusedVariable(Message):
    def __init__(self, filename: str, loc: Any, names: Tuple[str]) -> None:
        super().__init__(filename, loc)
        self.message_args = (names,)

class ReturnWithArgsInsideGenerator(Message):
    pass

class ReturnOutsideFunction(Message):
    pass
