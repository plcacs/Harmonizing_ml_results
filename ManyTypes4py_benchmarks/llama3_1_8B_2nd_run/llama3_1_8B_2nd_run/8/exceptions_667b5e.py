from collections import namedtuple
from typing import Optional, List, Dict

Position = namedtuple('Position', ['line_no', 'column_no', 'index'])

class ErrorMessage:
    def __init__(self, text: str, code: str, index: Optional[int] = None, position: Optional[Position] = None):
        self.text: str = text
        self.code: str = code
        self.index: Optional[int] = index
        self.position: Optional[Position] = position

    def __eq__(self, other: 'ErrorMessage') -> bool:
        return self.text == other.text and self.code == other.code and (self.index == other.index) and (self.position == other.position)

    def __repr__(self) -> str:
        return '%s(%s, code=%s, index=%s, position=%s)' % (self.__class__.__name__, repr(self.text), repr(self.code), repr(self.index), repr(self.position))

class DecodeError(Exception):
    def __init__(self, messages: List[ErrorMessage], summary: Optional[str] = None):
        self.messages: List[ErrorMessage] = messages
        self.summary: Optional[str] = summary
        super().__init__(str(messages))

class ParseError(DecodeError):
    pass

class ValidationError(DecodeError):
    def as_dict(self) -> Dict[str, Dict[str, str]]:
        ret: Dict[str, Dict[str, str]] = {}
        for message in self.messages:
            lookup: Dict[str, Dict[str, str]] = ret
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

    def __init__(self, title: str, status_code: int, content: str):
        self.title: str = title
        self.status_code: int = status_code
        self.content: str = content

class ClientError(Exception):
    """
    Raised when a client is unable to fulfil an API request.
    """

    def __init__(self, messages: List[ErrorMessage]):
        self.messages: List[ErrorMessage] = messages
        super().__init__(str(messages))
