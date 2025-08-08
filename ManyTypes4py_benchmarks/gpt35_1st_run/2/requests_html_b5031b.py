from typing import Set, Union, List, MutableMapping, Optional
import pyppeteer
import requests
import http.cookiejar
from pyquery import PyQuery
from lxml.html.clean import Cleaner
import lxml
from lxml import etree
from lxml.html import HtmlElement
from lxml.html import tostring as lxml_html_tostring
from lxml.html.soupparser import fromstring as soup_parse
from parse import search as parse_search
from parse import findall, Result
from w3lib.encoding import html_to_unicode

_DEFAULT_ENCODING: str = 'utf-8'
_DEFAULT_URL: str = 'https://example.org/'
_DEFAULT_USER_AGENT: str = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/603.3.8 (KHTML, like Gecko) Version/10.1.2 Safari/603.3.8'
_DEFAULT_NEXT_SYMBOL: List[str] = ['next', 'more', 'older']

class MaxRetries(Exception):
    def __init__(self, message: str):
        self.message = message

class BaseParser:
    def __init__(self, *, element: HtmlElement, default_encoding: Optional[str] = None, html: Optional[str] = None, url: str):
        self.element = element
        self.url = url
        self.skip_anchors = True
        self.default_encoding = default_encoding
        self._encoding = None
        self._html = html.encode(_DEFAULT_ENCODING) if isinstance(html, str) else html
        self._lxml = None
        self._pq = None

    @property
    def raw_html(self) -> bytes:
        ...

    @property
    def html(self) -> str:
        ...

    @html.setter
    def html(self, html: str):
        ...

    @property
    def encoding(self) -> str:
        ...

    @encoding.setter
    def encoding(self, enc: str):
        ...

    @property
    def pq(self) -> PyQuery:
        ...

    @property
    def lxml(self) -> HtmlElement:
        ...

    @property
    def text(self) -> str:
        ...

    @property
    def full_text(self) -> str:
        ...

    def find(self, selector: str = '*', *, containing: Union[str, List[str]] = None, clean: bool = False, first: bool = False, _encoding: Optional[str] = None) -> _Find:
        ...

    def xpath(self, selector: str, *, clean: bool = False, first: bool = False, _encoding: Optional[str] = None) -> _XPath:
        ...

    def search(self, template: str) -> _Search:
        ...

    def search_all(self, template: str) -> _Result:
        ...

    @property
    def links(self) -> Set[str]:
        ...

    @property
    def absolute_links(self) -> Set[str]:
        ...

    @property
    def base_url(self) -> str:
        ...

class Element(BaseParser):
    def __init__(self, *, element: HtmlElement, url: str, default_encoding: Optional[str] = None):
        ...

    def __repr__(self) -> str:
        ...

    @property
    def attrs(self) -> MutableMapping:
        ...

class HTML(BaseParser):
    def __init__(self, *, session: Optional[Union[AsyncHTMLSession, HTMLSession]] = None, url: str = _DEFAULT_URL, html: Union[str, bytes], default_encoding: str = _DEFAULT_ENCODING, async_: bool = False):
        ...

    def __repr__(self) -> str:
        ...

    def next(self, fetch: bool = False, next_symbol: Optional[List[str]] = None) -> Optional[Union[HTML, str]]:
        ...

    def __iter__(self):
        ...

    def __next__(self):
        ...

    def __aiter__(self):
        ...

    async def __anext__(self):
        ...

    def add_next_symbol(self, next_symbol: str):
        ...

    async def _async_render(self, *, url: str, script: Optional[str], scrolldown: int, sleep: int, wait: int, reload: bool, content: str, timeout: float, keep_page: bool, cookies: List[MutableMapping] = [{}]) -> Optional[tuple]:
        ...

    def _convert_cookiejar_to_render(self, session_cookiejar: http.cookiejar.CookieJar) -> MutableMapping:
        ...

    def _convert_cookiesjar_to_render(self) -> List[MutableMapping]:
        ...

    def render(self, retries: int = 8, script: Optional[str] = None, wait: float = 0.2, scrolldown: int = 0, sleep: int = 0, reload: bool = True, timeout: float = 8.0, keep_page: bool = False, cookies: List[MutableMapping] = [{}], send_cookies_session: bool = False) -> Optional[Union[dict, None]]:
        ...

    async def arender(self, retries: int = 8, script: Optional[str] = None, wait: float = 0.2, scrolldown: int = 0, sleep: int = 0, reload: bool = True, timeout: float = 8.0, keep_page: bool = False, cookies: List[MutableMapping] = [{}], send_cookies_session: bool = False) -> Optional[Union[dict, None]]:
        ...

class HTMLResponse(requests.Response):
    def __init__(self, session: Union[AsyncHTMLSession, HTMLSession]):
        ...

    @property
    def html(self) -> HTML:
        ...

    @classmethod
    def _from_response(cls, response: requests.Response, session: Union[AsyncHTMLSession, HTMLSession]) -> HTMLResponse:
        ...

class BaseSession(requests.Session):
    def __init__(self, mock_browser: bool = True, verify: bool = True, browser_args: List[str] = ['--no-sandbox']):
        ...

    def response_hook(self, response: requests.Response, **kwargs) -> HTMLResponse:
        ...

    @property
    async def browser(self) -> pyppeteer.browser.Browser:
        ...

class HTMLSession(BaseSession):
    def __init__(self, **kwargs):
        ...

    @property
    def browser(self) -> pyppeteer.browser.Browser:
        ...

    def close(self):
        ...

class AsyncHTMLSession(BaseSession):
    def __init__(self, loop: Optional[asyncio.AbstractEventLoop] = None, workers: Optional[int] = None, mock_browser: bool = True, *args, **kwargs):
        ...

    def request(self, *args, **kwargs):
        ...

    async def close(self):
        ...

    def run(self, *coros) -> List:
        ...
