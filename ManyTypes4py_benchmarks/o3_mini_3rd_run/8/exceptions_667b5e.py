from collections import namedtuple
from typing import Optional, Any, List, Dict

Position = namedtuple('Position', ['line_no', 'column_no', 'index'])
# Optionally, you can add type hints for Position fields if needed.

class ErrorMessage:
    text: str
    code: Any
    index: Optional[List[Any]]
    position: Optional[Position]

    def __init__(self, text: str, code: Any, index: Optional[List[Any]] = None, position: Optional[Position] = None) -> None:
        self.text = text
        self.code = code
        self.index = index
        self.position = position

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ErrorMessage):
            return False
        return (
            self.text == other.text and
            self.code == other.code and
            self.index == other.index and
            self.position == other.position
        )

    def __repr__(self) -> str:
        return '%s(%s, code=%s, index=%s, position=%s)' % (
            self.__class__.__name__,
            repr(self.text),
            repr(self.code),
            repr(self.index),
            repr(self.position)
        )

class DecodeError(Exception):
    messages: List[ErrorMessage]
    summary: Optional[Any]

    def __init__(self, messages: List[ErrorMessage], summary: Optional[Any] = None) -> None:
        self.messages = messages
        self.summary = summary
        super().__init__(messages)

class ParseError(DecodeError):
    pass

class ValidationError(DecodeError):
    def as_dict(self) -> Dict[Any, Any]:
        ret: Dict[Any, Any] = {}
        for message in self.messages:
            lookup: Dict[Any, Any] = ret
            if message.index:
                for key in message.index[:-1]:
                    lookup.setdefault(key, {})
                    lookup = lookup[key]
            key = message.index[-1] if message.index else None
            lookup[key] = message.text
        return ret

class ErrorResponse(Exception):
    title: str
    status_code: int
    content: Any

    def __init__(self, title: str, status_code: int, content: Any) -> None:
        self.title = title
        self.status_code = status_code
        self.content = content

class ClientError(Exception):
    messages: List[ErrorMessage]

    def __init__(self, messages: List[ErrorMessage]) -> None:
        self.messages = messages
        super().__init__(messages)