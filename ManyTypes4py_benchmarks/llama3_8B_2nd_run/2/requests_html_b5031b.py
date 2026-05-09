import asyncio
from urllib.parse import urlparse, urlunparse, urljoin
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures._base import TimeoutError
from functools import partial
from typing import Set, Union, List, MutableMapping, Optional

class MaxRetries(Exception):
    def __init__(self, message):
        self.message = message

class BaseParser:
    def __init__(self, *, element: Element, default_encoding: str = None, html: str = None, url: str):
        # ...

    @property
    def raw_html(self) -> bytes:
        # ...

    @property
    def html(self) -> str:
        # ...

    def find(self, selector: str, *, containing: Optional[str] = None, clean: bool = False, first: bool = False, _encoding: Optional[str] = None) -> Union[List[Element], Element]:
        # ...

    def xpath(self, selector: str, *, clean: bool = False, first: bool = False, _encoding: Optional[str] = None) -> Union[List[Element], Element]:
        # ...

class Element(BaseParser):
    # ...

class HTML(BaseParser):
    # ...

class HTMLResponse(requests.Response):
    # ...

def user_agent(style: Optional[str] = None) -> str:
    # ...

def _get_first_or_list(l: List[Element], first: bool = False) -> Union[Element, List[Element]]:
    # ...

class BaseSession(requests.Session):
    # ...

class HTMLSession(BaseSession):
    # ...

class AsyncHTMLSession(BaseSession):
    # ...
