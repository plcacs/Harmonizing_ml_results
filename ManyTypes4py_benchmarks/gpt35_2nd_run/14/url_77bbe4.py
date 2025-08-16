from __future__ import absolute_import
from collections import namedtuple
from typing import Optional

url_attrs = ['scheme', 'auth', 'host', 'port', 'path', 'query', 'fragment']
NORMALIZABLE_SCHEMES = ('http', 'https', None)

class Url(namedtuple('Url', url_attrs)):
    __slots__ = ()

    def __new__(cls, scheme: Optional[str] = None, auth: Optional[str] = None, host: Optional[str] = None, port: Optional[int] = None, path: Optional[str] = None, query: Optional[str] = None, fragment: Optional[str] = None):
        ...

    @property
    def hostname(self) -> Optional[str]:
        ...

    @property
    def request_uri(self) -> str:
        ...

    @property
    def netloc(self) -> str:
        ...

    @property
    def url(self) -> str:
        ...

    def __str__(self) -> str:
        ...

def split_first(s: str, delims: str) -> tuple:
    ...

def parse_url(url: str) -> Url:
    ...

def get_host(url: str) -> tuple:
    ...
