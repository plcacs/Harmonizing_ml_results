import base64
import binascii
import json
import re
import sys
import uuid
import warnings
import zlib
from collections import deque
from types import TracebackType
from typing import TYPE_CHECKING, Any, Deque, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Type, Union, cast
from urllib.parse import parse_qsl, unquote, urlencode
from multidict import CIMultiDict, CIMultiDictProxy
from .compression_utils import ZLibCompressor, ZLibDecompressor
from .hdrs import CONTENT_DISPOSITION, CONTENT_ENCODING, CONTENT_LENGTH, CONTENT_TRANSFER_ENCODING, CONTENT_TYPE
from .helpers import CHAR, TOKEN, parse_mimetype, reify
from .http import HeadersParser
from .payload import JsonPayload, LookupError, Order, Payload, StringPayload, get_payload, payload_type
from .streams import StreamReader

class BodyPartReader:
    """Multipart reader for single body part."""
    chunk_size: int = 8192

    def __init__(self, boundary: bytes, headers: Dict[str, str], content: StreamReader, *, subtype: str = 'mixed', default_charset: str = None) -> None:
        # ...

    def at_eof(self) -> bool:
        """Returns True if the boundary was reached or False otherwise."""
        return self._at_eof

    async def read(self) -> bytes:
        """Reads body part data."""
        # ...

    async def text(self, *, encoding: str = None) -> str:
        """Like read(), but assumes that body part contains text data."""
        # ...

    async def json(self, *, encoding: str = None) -> Dict[str, Any]:
        """Like read(), but assumes that body parts contains JSON data."""
        # ...

    async def form(self, *, encoding: str = None) -> List[Tuple[str, str]]:
        """Like read(), but assumes that body parts contain form urlencoded data."""
        # ...

class MultipartReader:
    """Multipart body reader."""

    def __init__(self, headers: Dict[str, str], content: StreamReader) -> None:
        # ...

    async def next(self) -> Optional[BodyPartReader]:
        """Emits the next multipart body part."""
        # ...

    async def release(self) -> None:
        """Reads all the body parts to the void till the final boundary."""
        # ...

class MultipartWriter(Payload):
    """Multipart body writer."""

    def __init__(self, subtype: str = 'mixed', boundary: str = None) -> None:
        # ...

    def append(self, obj: Any, headers: Dict[str, str] = None) -> Payload:
        """Adds a new body part to multipart writer."""
        # ...

    def append_json(self, obj: Any, headers: Dict[str, str] = None) -> Payload:
        """Helper to append JSON part."""
        # ...

    def append_form(self, obj: Any, headers: Dict[str, str] = None) -> Payload:
        """Helper to append form urlencoded part."""
        # ...

    async def write(self, writer: StreamWriter, close_boundary: bool = True) -> None:
        """Write body."""
        # ...

class MultipartPayloadWriter:
    """Multipart payload writer."""

    def __init__(self, writer: StreamWriter) -> None:
        # ...

    def enable_encoding(self, encoding: str) -> None:
        # ...

    def enable_compression(self, encoding: str = 'deflate', strategy: int = zlib.Z_DEFAULT_STRATEGY) -> None:
        # ...

    async def write_eof(self) -> None:
        """Write end of payload."""
        # ...

    async def write(self, chunk: bytes) -> None:
        """Write payload chunk."""
        # ...
