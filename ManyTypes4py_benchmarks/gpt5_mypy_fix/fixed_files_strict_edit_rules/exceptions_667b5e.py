from typing import Any, Dict, NamedTuple, Optional, Sequence
from collections.abc import Hashable


class Position(NamedTuple):
    line_no: int
    column_no: int
    index: int


IndexKey = Hashable
IndexPath = Sequence[IndexKey]


class ErrorMessage:
    def __init__(
        self,
        text: str,
        code: str,
        index: Optional[IndexPath] = None,
        position: Optional[Position] = None,
    ) -> None:
        self.text: str = text
        self.code: str = code
        self.index: Optional[IndexPath] = index
        self.position: Optional[Position] = position

    def __eq__(self, other: Any) -> bool:
        return (
            self.text == other.text
            and self.code == other.code
            and (self.index == other.index)
            and (self.position == other.position)
        )

    def __repr__(self) -> str:
        return "%s(%s, code=%s, index=%s, position=%s)" % (
            self.__class__.__name__,
            repr(self.text),
            repr(self.code),
            repr(self.index),
            repr(self.position),
        )


class DecodeError(Exception):
    def __init__(
        self, messages: Sequence[ErrorMessage], summary: Optional[str] = None
    ) -> None:
        self.messages: Sequence[ErrorMessage] = messages
        self.summary: Optional[str] = summary
        super().__init__(messages)


class ParseError(DecodeError):
    pass


class ValidationError(DecodeError):
    def as_dict(self) -> Dict[Optional[IndexKey], Any]:
        ret: Dict[Optional[IndexKey], Any] = {}
        for message in self.messages:
            lookup: Dict[Any, Any] = ret
            if message.index:
                for key in message.index[:-1]:
                    lookup.setdefault(key, {})
                    lookup = lookup[key]
            key: Optional[IndexKey] = message.index[-1] if message.index else None
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

    def __init__(self, messages: Sequence[str]) -> None:
        self.messages: Sequence[str] = messages
        super().__init__(messages)