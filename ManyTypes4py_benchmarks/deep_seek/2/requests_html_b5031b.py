import sys
import asyncio
from urllib.parse import urlparse, urlunparse, urljoin
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures._base import TimeoutError
from functools import partial
from typing import Set, Union, List, MutableMapping, Optional, Dict, Any, Tuple, Generator, Iterator, AsyncIterator, Callable, TypeVar, cast
import pyppeteer
import requests
import http.cookiejar
from pyquery import PyQuery
from fake_useragent import UserAgent
from lxml.html.clean import Cleaner
import lxml
from lxml import etree
from lxml.html import HtmlElement
from lxml.html import tostring as lxml_html_tostring
from lxml.html.soupparser import fromstring as soup_parse
from parse import search as parse_search
from parse import findall, Result
from w3lib.encoding import html_to_unicode

DEFAULT_ENCODING: str = 'utf-8'
DEFAULT_URL: str = 'https://example.org/'
DEFAULT_USER_AGENT: str = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/603.3.8 (KHTML, like Gecko) Version/10.1.2 Safari/603.3.8'
DEFAULT_NEXT_SYMBOL: List[str] = ['next', 'more', 'older']

cleaner: Cleaner = Cleaner()
cleaner.javascript = True
cleaner.style = True

useragent: Optional[UserAgent] = None

T = TypeVar('T')
_Find = Union[List[T], T]
_XPath = Union[List[str], List[HtmlElement], str, HtmlElement]
_Result = Union[List[Result], Result]
_HTML = Union[str, bytes]
_BaseHTML = str
_UserAgent = str
_DefaultEncoding = str
_URL = str
_RawHTML = bytes
_Encoding = str
_LXML = HtmlElement
_Text = str
_Search = Result
_Containing = Union[str, List[str]]
_Links = Set[str]
_Attrs = MutableMapping[str, Any]
_Next = Union['HTML', List[str]]
_NextSymbol = List[str]

try:
    assert sys.version_info.major == 3
    assert sys.version_info.minor > 5
except AssertionError:
    raise RuntimeError('Requests-HTML requires Python 3.6+!')

class MaxRetries(Exception):
    def __init__(self, message: str) -> None:
        self.message: str = message

class BaseParser:
    def __init__(
        self,
        *,
        element: Any,
        default_encoding: Optional[str] = None,
        html: Optional[Union[str, bytes]] = None,
        url: str
    ) -> None:
        self.element: Any = element
        self.url: str = url
        self.skip_anchors: bool = True
        self.default_encoding: Optional[str] = default_encoding
        self._encoding: Optional[str] = None
        self._html: Optional[bytes] = html.encode(DEFAULT_ENCODING) if isinstance(html, str) else html
        self._lxml: Optional[HtmlElement] = None
        self._pq: Optional[PyQuery] = None

    @property
    def raw_html(self) -> bytes:
        if self._html:
            return self._html
        else:
            return etree.tostring(self.element, encoding='unicode').strip().encode(self.encoding)

    @property
    def html(self) -> str:
        if self._html:
            return self.raw_html.decode(self.encoding, errors='replace')
        else:
            return etree.tostring(self.element, encoding='unicode').strip()

    @html.setter
    def html(self, html: str) -> None:
        self._html = html.encode(self.encoding)

    @raw_html.setter
    def raw_html(self, html: bytes) -> None:
        self._html = html

    @property
    def encoding(self) -> str:
        if self._encoding:
            return self._encoding
        if self._html:
            self._encoding = html_to_unicode(self.default_encoding, self._html)[0]
            try:
                self.raw_html.decode(self.encoding, errors='replace')
            except UnicodeDecodeError:
                self._encoding = self.default_encoding
        return self._encoding if self._encoding else cast(str, self.default_encoding)

    @encoding.setter
    def encoding(self, enc: str) -> None:
        self._encoding = enc

    @property
    def pq(self) -> PyQuery:
        if self._pq is None:
            self._pq = PyQuery(self.lxml)
        return self._pq

    @property
    def lxml(self) -> HtmlElement:
        if self._lxml is None:
            try:
                self._lxml = soup_parse(self.html, features='html.parser')
            except ValueError:
                self._lxml = lxml.html.fromstring(self.raw_html)
        return self._lxml

    @property
    def text(self) -> str:
        return self.pq.text()

    @property
    def full_text(self) -> str:
        return self.lxml.text_content()

    def find(
        self,
        selector: str = '*',
        *,
        containing: Optional[_Containing] = None,
        clean: bool = False,
        first: bool = False,
        _encoding: Optional[str] = None
    ) -> Optional[Union[List['Element'], 'Element']]:
        if isinstance(containing, str):
            containing = [containing]
        encoding: str = _encoding or self.encoding
        elements: List['Element'] = [
            Element(element=found, url=self.url, default_encoding=encoding)
            for found in self.pq(selector)
        ]
        if containing:
            elements_copy = elements.copy()
            elements = []
            for element in elements_copy:
                if any([c.lower() in element.full_text.lower() for c in containing]):
                    elements.append(element)
            elements.reverse()
        if clean:
            elements_copy = elements.copy()
            elements = []
            for element in elements_copy:
                element.raw_html = lxml_html_tostring(cleaner.clean_html(element.lxml))
                elements.append(element)
        return _get_first_or_list(elements, first)

    def xpath(
        self,
        selector: str,
        *,
        clean: bool = False,
        first: bool = False,
        _encoding: Optional[str] = None
    ) -> Optional[Union[List[Union[str, 'Element']], str, 'Element']]:
        selected: List[Any] = self.lxml.xpath(selector)
        elements: List[Union['Element', str]] = [
            Element(element=selection, url=self.url, default_encoding=_encoding or self.encoding)
            if not isinstance(selection, etree._ElementUnicodeResult)
            else str(selection)
            for selection in selected
        ]
        if clean:
            elements_copy = elements.copy()
            elements = []
            for element in elements_copy:
                if isinstance(element, str):
                    continue
                element.raw_html = lxml_html_tostring(cleaner.clean_html(element.lxml))
                elements.append(element)
        return _get_first_or_list(elements, first)

    def search(self, template: str) -> Optional[Result]:
        return parse_search(template, self.html)

    def search_all(self, template: str) -> List[Result]:
        return [r for r in findall(template, self.html)]

    @property
    def links(self) -> Set[str]:
        def gen() -> Generator[str, None, None]:
            for link in self.find('a'):
                try:
                    href = link.attrs['href'].strip()
                    if href and (not (href.startswith('#') and self.skip_anchors)) and (not href.startswith(('javascript:', 'mailto:'))):
                        yield href
                except KeyError:
                    pass
        return set(gen())

    def _make_absolute(self, link: str) -> str:
        parsed = urlparse(link)._asdict()
        if not parsed['netloc']:
            return urljoin(self.base_url, link)
        if not parsed['scheme']:
            parsed['scheme'] = urlparse(self.base_url).scheme
            parsed = (v for v in parsed.values())
            return urlunparse(parsed)
        return link

    @property
    def absolute_links(self) -> Set[str]:
        def gen() -> Generator[str, None, None]:
            for link in self.links:
                yield self._make_absolute(link)
        return set(gen())

    @property
    def base_url(self) -> str:
        base = self.find('base', first=True)
        if base:
            result = base.attrs.get('href', '').strip()
            if result:
                return result
        parsed = urlparse(self.url)._asdict()
        parsed['path'] = '/'.join(parsed['path'].split('/')[:-1]) + '/'
        parsed = (v for v in parsed.values())
        url = urlunparse(parsed)
        return url

class Element(BaseParser):
    __slots__ = ['element', 'url', 'skip_anchors', 'default_encoding', '_encoding', '_html', '_lxml', '_pq', '_attrs', 'session']

    def __init__(self, *, element: Any, url: str, default_encoding: Optional[str] = None) -> None:
        super(Element, self).__init__(element=element, url=url, default_encoding=default_encoding)
        self.element: Any = element
        self.tag: str = element.tag
        self.lineno: int = element.sourceline
        self._attrs: Optional[Dict[str, Any]] = None

    def __repr__(self) -> str:
        attrs = ['{}={}'.format(attr, repr(self.attrs[attr])) for attr in self.attrs]
        return '<Element {} {}>'.format(repr(self.element.tag), ' '.join(attrs))

    @property
    def attrs(self) -> Dict[str, Any]:
        if self._attrs is None:
            self._attrs = {k: v for k, v in self.element.items()}
            for attr in ['class', 'rel']:
                if attr in self._attrs:
                    self._attrs[attr] = tuple(self._attrs[attr].split())
        return self._attrs

class HTML(BaseParser):
    def __init__(
        self,
        *,
        session: Optional[Union['HTMLSession', 'AsyncHTMLSession']] = None,
        url: str = DEFAULT_URL,
        html: Union[str, bytes],
        default_encoding: str = DEFAULT_ENCODING,
        async_: bool = False
    ) -> None:
        if isinstance(html, str):
            html = html.encode(DEFAULT_ENCODING)
        pq: PyQuery = PyQuery(html)
        super(HTML, self).__init__(
            element=pq('html') or pq.wrapAll('<html></html>')('html'),
            html=html,
            url=url,
            default_encoding=default_encoding
        )
        self.session: Union['HTMLSession', 'AsyncHTMLSession'] = session or (async_ and AsyncHTMLSession()) or HTMLSession()
        self.page: Optional[Any] = None
        self.next_symbol: List[str] = DEFAULT_NEXT_SYMBOL

    def __repr__(self) -> str:
        return f'<HTML url={self.url!r}>'

    def next(
        self,
        fetch: bool = False,
        next_symbol: Optional[List[str]] = None
    ) -> Optional[Union['HTML', str]]:
        if next_symbol is None:
            next_symbol = DEFAULT_NEXT_SYMBOL

        def get_next() -> Optional[str]:
            candidates = self.find('a', containing=next_symbol)
            for candidate in candidates:
                if candidate.attrs.get('href'):
                    if 'next' in candidate.attrs.get('rel', []):
                        return candidate.attrs['href']
                    for _class in candidate.attrs.get('class', []):
                        if 'next' in _class:
                            return candidate.attrs['href']
                    if 'page' in candidate.attrs['href']:
                        return candidate.attrs['href']
            try:
                return candidates[-1].attrs['href']
            except IndexError:
                return None
        __next = get_next()
        if __next:
            url = self._make_absolute(__next)
        else:
            return None
        if fetch:
            return self.session.get(url)
        else:
            return url

    def __iter__(self) -> Iterator['HTML']:
        next_page = self
        while True:
            yield next_page
            try:
                next_page = next_page.next(fetch=True, next_symbol=self.next_symbol).html
            except AttributeError:
                break

    def __next__(self) -> 'HTML':
        return cast('HTML', self.next(fetch=True, next_symbol=self.next_symbol).html)

    def __aiter__(self) -> AsyncIterator['HTML']:
        return self

    async def __anext__(self) -> 'HTML':
        while True:
            url = self.next(fetch=False, next_symbol=self.next_symbol)
            if not url:
                break
            response = await self.session.get(url)
            return response.html

    def add_next_symbol(self, next_symbol: str) -> None:
        self.next_symbol.append(next_symbol)

    async def _async_render(
        self,
        *,
        url: str,
        script: Optional[str] = None,
        scrolldown: Union[int, bool],
        sleep: Union[int, float],
        wait: Union[int, float],
        reload: bool,
        content: Optional[str],
        timeout: float,
        keep_page: bool,
        cookies: List[Dict[str, Any]] = [{}]
    ) -> Optional[Tuple[str, Any, Any]]:
        try:
            page = await self.browser.newPage()
            await asyncio.sleep(wait)
            if cookies:
                for cookie in cookies:
                    if cookie:
                        await page.setCookie(cookie)
            if reload:
                await page.goto(url, options={'timeout': int(timeout * 1000)})
            else:
                await page.goto(f'data:text/html,{self.html}', options={'timeout': int(timeout * 1000)})
            result = None
            if script:
                result = await page.evaluate(script)
            if scrolldown:
                for _ in range(scrolldown):
                    await page._keyboard.down('PageDown')
                    await asyncio.sleep(sleep)
            else:
                await asyncio.sleep(sleep)
            if scrolldown:
                await page._keyboard.up('PageDown')
            content = await page.content()
            if not keep_page:
                await page.close()
                page = None
            return (content, result, page)
        except TimeoutError:
            await page.close()
            page = None
            return None

    def _convert_cookiejar_to_render(self, session_cookiejar: http.cookiejar.Cookie) -> Dict[str, Any]:
        cookie_render: Dict[str, Any] = {}

        def __convert(cookiejar: http.cookiejar.Cookie, key: str) -> Dict[str, Any]:
            try:
                v = eval('cookiejar.' + key)
                if not v:
                    kv: Dict[str, Any] = {}
                else:
                    kv = {key: v}
            except:
                kv = {}
            return kv
        keys = ['name', 'value', 'url', 'domain', 'path', 'sameSite', 'expires', 'httpOnly', 'secure']
        for key in keys:
            cookie_render.update(__convert(session_cookiejar, key))
        return cookie_render

    def _convert_cookiesjar_to_render(self) -> List[Dict[str, Any]]:
        cookies_render: List[Dict[str, Any]] = []
        if isinstance(self.session.cookies, http.cookiejar.CookieJar):
            for cookie in self.session.cookies:
                cookies_render.append(self._convert_cookiejar_to_render(cookie))
        return cookies_render

    def render(
        self,
        retries: int = 8,
        script: Optional[str] = None,
        wait: float = 0.2,
        scrolldown: Union[int, bool] = False,
        sleep: Union[int, float] = 0,
        reload: bool = True,
        timeout: float = 8.0,
        keep_page: bool = False,
        cookies: List[Dict[str, Any]] = [{}],
        send_cookies_session: bool = False
    ) -> Any:
        self.browser = self.session.browser
        content = None
        if self.url == DEFAULT_URL:
            reload = False
        if send_cookies_session:
            cookies = self._convert_cookiesjar_to_render()
        for i in range(retries):
            if not content:
                try:
                    content, result, page = self.session.loop.run_until_complete(
                        self._async_render(
                            url=self.url,
                            script=script,
                            sleep=sleep,
                            wait=wait,
                            content=self.html,
                            reload=reload,
                            scrolldown=scrolldown,
                            timeout=timeout,
                            keep_page=keep_page,
                            cookies=cookies
                        )
                    )
                except TypeError:
                    pass
            else:
                break
        if not content:
            raise MaxRetries('Unable to render the page. Try increasing timeout')
        html = HTML(url=self.url, html=content.encode(DEFAULT_ENCODING), default_encoding=DEFAULT_ENCODING)
        self.__dict__.update(html.__dict__)
        self.page = page
        return result

    async def arender(
        self,
        retries: int = 8,
        script: Optional[str] = None,
        wait: float = 0.2,
        scrolldown: Union[int, bool] = False,
        sleep: Union[int, float] = 0,
        reload: bool = True,
        timeout: float = 8.0,
        keep_page: bool = False,
        cookies: List[Dict[str, Any]] = [{}],
        send_cookies_session: bool = False
    ) -> Any:
        self.browser = await self.session.browser
        content = None
        if self.url