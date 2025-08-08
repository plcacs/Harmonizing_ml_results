from __future__ import annotations
import io
import itertools
import sys
import typing
from .._models import Request, Response
from .._types import SyncByteStream
from .base import BaseTransport
if typing.TYPE_CHECKING:
    from _typeshed import OptExcInfo
    from _typeshed.wsgi import WSGIApplication
_T = typing.TypeVar('_T')
__all__: typing.List[str] = ['WSGITransport']

def _skip_leading_empty_chunks(body: typing.Iterable[bytes]) -> typing.Iterable[bytes]:
    body = iter(body)
    for chunk in body:
        if chunk:
            return itertools.chain([chunk], body)
    return []

class WSGIByteStream(SyncByteStream):

    def __init__(self, result: typing.Any) -> None:
        self._close = getattr(result, 'close', None)
        self._result = _skip_leading_empty_chunks(result)

    def __iter__(self) -> typing.Iterator[bytes]:
        for part in self._result:
            yield part

    def close(self) -> None:
        if self._close is not None:
            self._close()

class WSGITransport(BaseTransport):
    """
    A custom transport that handles sending requests directly to an WSGI app.
    The simplest way to use this functionality is to use the `app` argument.

    