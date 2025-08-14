#!/usr/bin/env python3
import sys
import asyncio
from urllib.parse import urlparse, urlunparse, urljoin
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from functools import partial
from typing import Set, Union, List, MutableMapping, Optional, Any, Tuple, Dict, Callable, Awaitable, Iterator, AsyncIterator

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

# Typing aliases.
_Find = Union[List["Element"], "Element"]
_XPath = Union[List[str], List["Element"], str, "Element"]
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
_Next = Union["HTML", List[str]]
_NextSymbol = List[str]

# Sanity checking.
try:
    assert sys.version_info.major == 3
    assert sys.version_info.minor > 5
except AssertionError:
    raise RuntimeError('Requests-HTML requires Python 3.6+!')


class MaxRetries(Exception):
    def __init__(self, message: str) -> None:
        self.message = message


class BaseParser:
    def __init__(self, *, element: Any, default_encoding: Optional[_DefaultEncoding] = None, html: Optional[_HTML] = None, url: _URL) -> None:
        self.element: Any = element
        self.url: _URL = url
        self.skip_anchors: bool = True
        self.default_encoding: _DefaultEncoding = default_encoding if default_encoding is not None else DEFAULT_ENCODING
        self._encoding: Optional[_Encoding] = None
        self._html: Optional[bytes] = html.encode(DEFAULT_ENCODING) if isinstance(html, str) else html
        self._lxml: Optional[HtmlElement] = None
        self._pq: Optional[PyQuery] = None

    @property
    def raw_html(self) -> _RawHTML:
        if self._html:
            return self._html
        else:
            return etree.tostring(self.element, encoding='unicode').strip().encode(self.encoding)

    @property
    def html(self) -> _BaseHTML:
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
    def encoding(self) -> _Encoding:
        if self._encoding:
            return self._encoding

        if self._html:
            self._encoding = html_to_unicode(self.default_encoding, self._html)[0]
            try:
                self.raw_html.decode(self.encoding, errors='replace')
            except UnicodeDecodeError:
                self._encoding = self.default_encoding

        return self._encoding if self._encoding else self.default_encoding

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
    def text(self) -> _Text:
        return self.pq.text()

    @property
    def full_text(self) -> _Text:
        return self.lxml.text_content()

    def find(self, selector: str = "*", *, containing: Optional[_Containing] = None, clean: bool = False, first: bool = False, _encoding: Optional[str] = None) -> _Find:
        if isinstance(containing, str):
            containing = [containing]

        encoding: str = _encoding or self.encoding
        elements: List[Element] = [
            Element(element=found, url=self.url, default_encoding=encoding)
            for found in self.pq(selector)
        ]

        if containing:
            elements_copy: List[Element] = elements.copy()
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

    def xpath(self, selector: str, *, clean: bool = False, first: bool = False, _encoding: Optional[str] = None) -> Union[List[Union[Element, str]], Union[Element, str]]:
        selected = self.lxml.xpath(selector)
        elements: List[Union[Element, str]] = [
            Element(element=selection, url=self.url, default_encoding=_encoding or self.encoding)
            if not isinstance(selection, etree._ElementUnicodeResult) else str(selection)
            for selection in selected
        ]

        if clean:
            elements_copy = elements.copy()
            elements = []
            for element in elements_copy:
                if isinstance(element, Element):
                    element.raw_html = lxml_html_tostring(cleaner.clean_html(element.lxml))
                    elements.append(element)
                else:
                    elements.append(element)
        return _get_first_or_list(elements, first)

    def search(self, template: str) -> Result:
        return parse_search(template, self.html)

    def search_all(self, template: str) -> List[Result]:
        return [r for r in findall(template, self.html)]

    @property
    def links(self) -> _Links:
        def gen() -> Iterator[str]:
            for link in self.find('a'):
                try:
                    href: str = link.attrs['href'].strip()
                    if href and not (href.startswith('#') and self.skip_anchors) and not href.startswith(('javascript:', 'mailto:')):
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
            return urlunparse(tuple(parsed))
        return link

    @property
    def absolute_links(self) -> _Links:
        def gen() -> Iterator[str]:
            for link in self.links:
                yield self._make_absolute(link)
        return set(gen())

    @property
    def base_url(self) -> _URL:
        base: Optional[Element] = self.find('base', first=True)
        if base:
            result: str = base.attrs.get('href', '').strip()
            if result:
                return result
        parsed = urlparse(self.url)._asdict()
        parsed['path'] = '/'.join(parsed['path'].split('/')[:-1]) + '/'
        parsed = (v for v in parsed.values())
        url: str = urlunparse(tuple(parsed))
        return url


class Element(BaseParser):
    __slots__ = [
        'element', 'url', 'skip_anchors', 'default_encoding', '_encoding',
        '_html', '_lxml', '_pq', '_attrs', 'session', 'tag', 'lineno'
    ]

    def __init__(self, *, element: Any, url: _URL, default_encoding: Optional[_DefaultEncoding] = None) -> None:
        super(Element, self).__init__(element=element, url=url, default_encoding=default_encoding)
        self.element = element
        self.tag: str = element.tag
        self.lineno: Optional[int] = element.sourceline
        self._attrs: Optional[MutableMapping[str, Any]] = None

    def __repr__(self) -> str:
        attrs = ['{}={}'.format(attr, repr(self.attrs[attr])) for attr in self.attrs]
        return "<Element {} {}>".format(repr(self.element.tag), ' '.join(attrs))

    @property
    def attrs(self) -> _Attrs:
        if self._attrs is None:
            self._attrs = {k: v for k, v in self.element.items()}
            for attr in ['class', 'rel']:
                if attr in self._attrs:
                    self._attrs[attr] = tuple(self._attrs[attr].split())
        return self._attrs


class HTML(BaseParser):
    def __init__(self, *, session: Optional[Union['HTMLSession', 'AsyncHTMLSession']] = None, url: str = DEFAULT_URL, html: _HTML, default_encoding: str = DEFAULT_ENCODING, async_: bool = False) -> None:
        if isinstance(html, str):
            html = html.encode(DEFAULT_ENCODING)
        pq = PyQuery(html)
        super(HTML, self).__init__(
            element=pq('html') or pq.wrapAll('<html></html>')('html'),
            html=html,
            url=url,
            default_encoding=default_encoding
        )
        self.session: Union['HTMLSession', 'AsyncHTMLSession'] = session if session is not None else (AsyncHTMLSession() if async_ else HTMLSession())
        self.page: Optional[Any] = None
        self.next_symbol: _NextSymbol = DEFAULT_NEXT_SYMBOL.copy()

    def __repr__(self) -> str:
        return f"<HTML url={self.url!r}>"

    def next(self, fetch: bool = False, next_symbol: Optional[_NextSymbol] = None) -> Union["HTML", str, None]:
        if next_symbol is None:
            next_symbol = DEFAULT_NEXT_SYMBOL

        def get_next() -> Optional[str]:
            candidates: Union[List[Element], Element] = self.find('a', containing=next_symbol)
            if isinstance(candidates, Element):
                candidates = [candidates]
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

        __next: Optional[str] = get_next()
        if __next:
            url: str = self._make_absolute(__next)
        else:
            return None

        if fetch:
            response: HTMLResponse = self.session.get(url)
            return response.html
        else:
            return url

    def __iter__(self) -> Iterator["HTML"]:
        next_page: Union["HTML", str, None] = self
        while True:
            yield next_page  # type: ignore
            try:
                response: HTMLResponse = next_page.next(fetch=True, next_symbol=self.next_symbol)  # type: ignore
                next_page = response.html
            except AttributeError:
                break

    def __next__(self) -> "HTML":
        response: HTMLResponse = self.next(fetch=True, next_symbol=self.next_symbol)  # type: ignore
        return response.html

    def __aiter__(self) -> AsyncIterator["HTML"]:
        return self  # type: ignore

    async def __anext__(self) -> "HTML":
        while True:
            url: Optional[Union[str, None]] = self.next(fetch=False, next_symbol=self.next_symbol)
            if not url:
                break
            response: HTMLResponse = await self.session.get(url)  # type: ignore
            return response.html
        raise StopAsyncIteration

    def add_next_symbol(self, next_symbol: str) -> None:
        self.next_symbol.append(next_symbol)

    async def _async_render(self, *, url: str, script: Optional[str] = None, scrolldown: Union[int, bool], sleep: int, wait: float, reload: bool, content: Optional[str], timeout: Union[float, int], keep_page: bool, cookies: List[Dict[str, Any]]) -> Optional[Tuple[str, Any, Any]]:
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
            result: Any = None
            if script:
                result = await page.evaluate(script)
            if scrolldown:
                for _ in range(int(scrolldown)):
                    await page._keyboard.down('PageDown')
                    await asyncio.sleep(sleep)
            else:
                await asyncio.sleep(sleep)
            if scrolldown:
                await page._keyboard.up('PageDown')
            content_render: str = await page.content()
            if not keep_page:
                await page.close()
                page = None
            return content_render, result, page
        except TimeoutError:
            await page.close()
            page = None
            return None

    def _convert_cookiejar_to_render(self, session_cookiejar: http.cookiejar.Cookie) -> Dict[str, Any]:
        cookie_render: Dict[str, Any] = {}
        def __convert(cookiejar: http.cookiejar.Cookie, key: str) -> Dict[str, Any]:
            try:
                v = getattr(cookiejar, key)
                kv: Dict[str, Any] = {key: v} if v else {}
            except Exception:
                kv = {}
            return kv
        keys: List[str] = [
            'name',
            'value',
            'url',
            'domain',
            'path',
            'sameSite',
            'expires',
            'httpOnly',
            'secure',
        ]
        for key in keys:
            cookie_render.update(__convert(session_cookiejar, key))
        return cookie_render

    def _convert_cookiesjar_to_render(self) -> List[Dict[str, Any]]:
        cookies_render: List[Dict[str, Any]] = []
        if isinstance(self.session.cookies, http.cookiejar.CookieJar):
            for cookie in self.session.cookies:
                cookies_render.append(self._convert_cookiejar_to_render(cookie))
        return cookies_render

    def render(self, retries: int = 8, script: Optional[str] = None, wait: float = 0.2, scrolldown: Union[int, bool] = False, sleep: int = 0, reload: bool = True, timeout: Union[float, int] = 8.0, keep_page: bool = False, cookies: List[Dict[str, Any]] = [{}], send_cookies_session: bool = False) -> Any:
        self.browser = self.session.browser
        content: Optional[str] = None
        if self.url == DEFAULT_URL:
            reload = False
        if send_cookies_session:
            cookies = self._convert_cookiesjar_to_render()
        for i in range(retries):
            if not content:
                try:
                    ret: Optional[Tuple[str, Any, Any]] = self.session.loop.run_until_complete(
                        self._async_render(url=self.url, script=script, sleep=sleep, wait=wait, content=self.html, reload=reload, scrolldown=scrolldown, timeout=timeout, keep_page=keep_page, cookies=cookies)
                    )
                    if ret:
                        content, result, page = ret
                except TypeError:
                    pass
            else:
                break
        if not content:
            raise MaxRetries("Unable to render the page. Try increasing timeout")
        html_instance = HTML(url=self.url, html=content.encode(DEFAULT_ENCODING), default_encoding=DEFAULT_ENCODING)
        self.__dict__.update(html_instance.__dict__)
        self.page = page
        return result

    async def arender(self, retries: int = 8, script: Optional[str] = None, wait: float = 0.2, scrolldown: Union[int, bool] = False, sleep: int = 0, reload: bool = True, timeout: Union[float, int] = 8.0, keep_page: bool = False, cookies: List[Dict[str, Any]] = [{}], send_cookies_session: bool = False) -> Any:
        self.browser = await self.session.browser  # type: ignore
        content: Optional[str] = None
        if self.url == DEFAULT_URL:
            reload = False
        if send_cookies_session:
            cookies = self._convert_cookiesjar_to_render()
        for _ in range(retries):
            if not content:
                try:
                    ret = await self._async_render(url=self.url, script=script, sleep=sleep, wait=wait, content=self.html, reload=reload, scrolldown=scrolldown, timeout=timeout, keep_page=keep_page, cookies=cookies)
                    if ret:
                        content, result, page = ret
                except TypeError:
                    pass
            else:
                break
        if not content:
            raise MaxRetries("Unable to render the page. Try increasing timeout")
        html_instance = HTML(url=self.url, html=content.encode(DEFAULT_ENCODING), default_encoding=DEFAULT_ENCODING)
        self.__dict__.update(html_instance.__dict__)
        self.page = page
        return result


class HTMLResponse(requests.Response):
    def __init__(self, session: Union['HTMLSession', 'AsyncHTMLSession']) -> None:
        super(HTMLResponse, self).__init__()
        self._html: Optional[HTML] = None
        self.session: Union['HTMLSession', 'AsyncHTMLSession'] = session

    @property
    def html(self) -> HTML:
        if not self._html:
            self._html = HTML(session=self.session, url=self.url, html=self.content, default_encoding=self.encoding)
        return self._html

    @classmethod
    def _from_response(cls, response: requests.Response, session: Union['HTMLSession', 'AsyncHTMLSession']) -> "HTMLResponse":
        html_r = cls(session=session)
        html_r.__dict__.update(response.__dict__)
        return html_r


def user_agent(style: Optional[str] = None) -> _UserAgent:
    global useragent
    if (not useragent) and style:
        useragent = UserAgent()
    return useragent[style] if style else DEFAULT_USER_AGENT


def _get_first_or_list(l: List[Any], first: bool = False) -> Any:
    if first:
        try:
            return l[0]
        except IndexError:
            return None
    else:
        return l


class BaseSession(requests.Session):
    def __init__(self, mock_browser: bool = True, verify: bool = True, browser_args: List[str] = ['--no-sandbox']) -> None:
        super().__init__()
        if mock_browser:
            self.headers['User-Agent'] = user_agent()
        self.hooks['response'].append(self.response_hook)
        self.verify: bool = verify
        self.__browser_args: List[str] = browser_args

    def response_hook(self, response: requests.Response, **kwargs: Any) -> HTMLResponse:
        if not response.encoding:
            response.encoding = DEFAULT_ENCODING
        return HTMLResponse._from_response(response, self)

    @property
    async def browser(self) -> Any:
        if not hasattr(self, "_browser"):
            self._browser = await pyppeteer.launch(ignoreHTTPSErrors=not self.verify, headless=True, args=self.__browser_args)
        return self._browser


class HTMLSession(BaseSession):
    def __init__(self, **kwargs: Any) -> None:
        super(HTMLSession, self).__init__(**kwargs)

    @property
    def browser(self) -> Any:
        if not hasattr(self, "_browser"):
            self.loop = asyncio.get_event_loop()
            if self.loop.is_running():
                raise RuntimeError("Cannot use HTMLSession within an existing event loop. Use AsyncHTMLSession instead.")
            self._browser = self.loop.run_until_complete(super().browser)
        return self._browser

    def close(self) -> None:
        if hasattr(self, "_browser"):
            self.loop.run_until_complete(self._browser.close())
        super().close()


class AsyncHTMLSession(BaseSession):
    def __init__(self, loop: Optional[asyncio.AbstractEventLoop] = None, workers: Optional[int] = None, mock_browser: bool = True, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.loop: asyncio.AbstractEventLoop = loop if loop is not None else asyncio.get_event_loop()
        self.thread_pool: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=workers)

    def request(self, *args: Any, **kwargs: Any) -> Awaitable[Any]:
        func: Callable[[], Any] = partial(super().request, *args, **kwargs)
        return self.loop.run_in_executor(self.thread_pool, func)

    async def close(self) -> None:
        if hasattr(self, "_browser"):
            await self._browser.close()
        super().close()

    def run(self, *coros: Callable[[], Awaitable[Any]]) -> List[Any]:
        tasks = [asyncio.ensure_future(coro()) for coro in coros]
        done, _ = self.loop.run_until_complete(asyncio.wait(tasks))
        return [t.result() for t in done]
