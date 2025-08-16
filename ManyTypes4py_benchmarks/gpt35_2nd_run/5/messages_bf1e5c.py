from typing import Tuple

class Message:
    message: str = ''
    message_args: Tuple = ()

    def __init__(self, filename: str, loc: Any) -> None:
        self.filename: str = filename
        self.lineno: int = loc.lineno
        self.col: int = getattr(loc, 'col_offset', 0)

    def __str__(self) -> str:
        return '%s:%s:%s %s' % (self.filename, self.lineno, self.col + 1, self.message % self.message_args)

class UnusedImport(Message):
    message: str = '%r imported but unused'

    def __init__(self, filename: str, loc: Any, name: str) -> None:
        Message.__init__(self, filename, loc)
        self.message_args = (name,)

# Define other subclasses with appropriate type annotations
