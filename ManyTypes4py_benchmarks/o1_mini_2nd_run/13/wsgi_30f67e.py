from __future__ import annotations
import io
import itertools
import sys
import typing
from typing import Any, Callable, Iterator, List, Optional, Tuple, TypeVar, Iterable
from .._models import Request, Response
from .._types import SyncByteStream
from .base import BaseTransport

if typing.TYPE_CHECKING:
    from _typeshed import OptExcInfo
    from _typeshed.wsgi import WSGIApplication

_T = TypeVar('_T')
__all__ = ['WSGITransport']


def _skip_leading_empty_chunks(body: Iterable[bytes]) -> Iterator[bytes]:
    body_iter = iter(body)
    for chunk in body_iter:
        if chunk:
            return itertools.chain([chunk], body_iter)
    return iter([])


class WSGIByteStream(SyncByteStream):

    def __init__(self, result: Iterable[bytes]) -> None:
        self._close: Optional[Callable[[], None]] = getattr(result, 'close', None)
        self._result: Iterator[bytes] = _skip_leading_empty_chunks(result)

    def __iter__(self) -> Iterator[bytes]:
        for part in self._result:
            yield part

    def close(self) -> None:
        if self._close is not None:
            self._close()


class WSGITransport(BaseTransport):
    """
    A custom transport that handles sending requests directly to an WSGI app.
    The simplest way to use this functionality is to use the `app` argument.

    