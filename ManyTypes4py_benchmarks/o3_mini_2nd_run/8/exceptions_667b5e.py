from collections import namedtuple
from typing import Optional, List, Any, Dict

Position = namedtuple('Position', ['line_no', 'column_no', 'index'])

class ErrorMessage:
    def __init__(self, text: str, code: Any, index: Optional[List[Any]] = None, position: Optional[Position] = None) -> None:
        self.text: str = text
        self.code: Any = code
        self.index: Optional[List[Any]] = index
        self.position: Optional[Position] = position

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ErrorMessage):
            return NotImplemented
        return (self.text == other.text and 
                self.code == other.code and 
                self.index == other.index and 
                self.position == other.position)

    def __repr__(self) -> str:
        return '%s(%s, code=%s, index=%s, position=%s)' % (
            self.__class__.__name__,
            repr(self.text),
            repr(self.code),
            repr(self.index),
            repr(self.position)
        )

class DecodeError(Exception):
    def __init__(self, messages: List[ErrorMessage], summary: Optional[str] = None) -> None:
        self.messages: List[ErrorMessage] = messages
        self.summary: Optional[str] = summary
        super().__init__(messages)

class ParseError(DecodeError):
    pass

class ValidationError(DecodeError):
    def as_dict(self) -> Dict[Optional[Any], Any]:
        ret: Dict[Optional[Any], Any] = {}
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
    """
    Raised when a client request results in an error response being returned.
    """
    def __init__(self, title: str, status_code: int, content: Any) -> None:
        self.title: str = title
        self.status_code: int = status_code
        self.content: Any = content

class ClientError(Exception):
    """
    Raised when a client is unable to fulfil an API request.
    """
    def __init__(self, messages: List[ErrorMessage]) -> None:
        self.messages: List[ErrorMessage] = messages
        super().__init__(messages)