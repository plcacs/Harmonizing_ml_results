from __future__ import absolute_import
import email.utils
import mimetypes
from .packages import six

def guess_content_type(filename: str, default: str='application/octet-stream') -> str:
    ...

def format_header_param(name: str, value: str) -> str:
    ...

class RequestField(object):
    def __init__(self, name: str, data: str, filename: str=None, headers: dict=None) -> None:
        ...

    @classmethod
    def from_tuples(cls, fieldname: str, value: Union[str, Tuple[str, str, Optional[str]]]) -> 'RequestField':
        ...

    def _render_part(self, name: str, value: str) -> str:
        ...

    def _render_parts(self, header_parts: Union[Dict[str, str], List[Tuple[str, str]]]) -> str:
        ...

    def render_headers(self) -> str:
        ...

    def make_multipart(self, content_disposition: str=None, content_type: str=None, content_location: str=None) -> None:
        ...
