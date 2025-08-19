import sys
import asyncio
from urllib.parse import urlparse, urlunparse, urljoin
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures._base import TimeoutError
from functools import partial
from typing import Set, Union, List, MutableMapping, Optional, Any, Dict, Iterator, AsyncIterator, Tuple, Callable, Awaitable
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

DEFAULT_ENCODING = 'utf-8'
DEFAULT_URL = 'https://example.org/'
DEFAULT_USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/603.3.8 (KHTML, like Gecko) Version/10.1.2 Safari/603.3.8'
DEFAULT_NEXT_SYMBOL: List[str] = ['next', 'more', 'older']
cleaner = Cleaner()
cleaner.javascript = True
cleaner.style = True
useragent: Optional[UserAgent] = None
_Find = Union[List['Element'], 'Element']
_XPath = Union[List[str], List['Element'], str, 'Element']
_Result = Union[List['Result'], 'Result']
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
_Attrs = MutableMapping
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
    """A basic HTML/Element Parser, for Humans.

    :param element: The element from which to base the parsing upon.
    :param default_encoding: Which encoding to default to.
    :param html: HTML from which to base the parsing upon (optional).
    :param url: The URL from which the HTML originated, used for ``absolute_links``.

    """

    def __init__(self, *, element: Any, default_encoding: Optional[_DefaultEncoding] = None, html: Optional[_HTML] = None, url: _URL) -> None:
        self.element: Any = element
        self.url: _URL = url
        self.skip_anchors: bool = True
        self.default_encoding: Optional[_DefaultEncoding] = default_encoding
        self._encoding: Optional[_Encoding] = None
        self._html: Optional[_RawHTML] = html.encode(DEFAULT_ENCODING) if isinstance(html, str) else html
        self._lxml: Optional[_LXML] = None
        self._pq: Optional[PyQuery] = None

    @property
    def raw_html(self) -> _RawHTML:
        """Bytes representation of the HTML content.
        (`learn more <http://www.diveintopython3.net/strings.html>`_).
        """
        if self._html:
            return self._html
        else:
            return etree.tostring(self.element, encoding='unicode').strip().encode(self.encoding)

    @property
    def html(self) -> _BaseHTML:
        """Unicode representation of the HTML content
        (`learn more <http://www.diveintopython3.net/strings.html>`_).
        """
        if self._html:
            return self.raw_html.decode(self.encoding, errors='replace')
        else:
            return etree.tostring(self.element, encoding='unicode').strip()

    @html.setter
    def html(self, html: _BaseHTML) -> None:
        self._html = html.encode(self.encoding)

    @raw_html.setter
    def raw_html(self, html: _RawHTML) -> None:
        """Property setter for self.html."""
        self._html = html

    @property
    def encoding(self) -> _Encoding:
        """The encoding string to be used, extracted from the HTML and
        :class:`HTMLResponse <HTMLResponse>` headers.
        """
        if self._encoding:
            return self._encoding
        if self._html:
            self._encoding = html_to_unicode(self.default_encoding, self._html)[0]
            try:
                self.raw_html.decode(self.encoding, errors='replace')
            except UnicodeDecodeError:
                self._encoding = self.default_encoding
        return self._encoding if self._encoding else self.default_encoding  # type: ignore[return-value]

    @encoding.setter
    def encoding(self, enc: _Encoding) -> None:
        """Property setter for self.encoding."""
        self._encoding = enc

    @property
    def pq(self) -> PyQuery:
        """`PyQuery <https://pythonhosted.org/pyquery/>`_ representation
        of the :class:`Element <Element>` or :class:`HTML <HTML>`.
        """
        if self._pq is None:
            self._pq = PyQuery(self.lxml)
        return self._pq

    @property
    def lxml(self) -> _LXML:
        """`lxml <http://lxml.de>`_ representation of the
        :class:`Element <Element>` or :class:`HTML <HTML>`.
        """
        if self._lxml is None:
            try:
                self._lxml = soup_parse(self.html, features='html.parser')
            except ValueError:
                self._lxml = lxml.html.fromstring(self.raw_html)
        return self._lxml

    @property
    def text(self) -> _Text:
        """The text content of the
        :class:`Element <Element>` or :class:`HTML <HTML>`.
        """
        return self.pq.text()

    @property
    def full_text(self) -> _Text:
        """The full text content (including links) of the
        :class:`Element <Element>` or :class:`HTML <HTML>`.
        """
        return self.lxml.text_content()

    def find(self, selector: str = '*', *, containing: Optional[_Containing] = None, clean: bool = False, first: bool = False, _encoding: Optional[_DefaultEncoding] = None) -> Optional[_Find]:
        """Given a CSS Selector, returns a list of
        :class:`Element <Element>` objects or a single one.

        :param selector: CSS Selector to use.
        :param clean: Whether or not to sanitize the found HTML of ``<script>`` and ``<style>`` tags.
        :param containing: If specified, only return elements that contain the provided text.
        :param first: Whether or not to return just the first result.
        :param _encoding: The encoding format.
        """
        if isinstance(containing, str):
            containing = [containing]
        encoding = _encoding or self.encoding
        elements: List[Element] = [Element(element=found, url=self.url, default_encoding=encoding) for found in self.pq(selector)]
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

    def xpath(self, selector: str, *, clean: bool = False, first: bool = False, _encoding: Optional[_DefaultEncoding] = None) -> Optional[_XPath]:
        """Given an XPath selector, returns a list of
        :class:`Element <Element>` objects or a single one.

        :param selector: XPath Selector to use.
        :param clean: Whether or not to sanitize the found HTML of ``<script>`` and ``<style>`` tags.
        :param first: Whether or not to return just the first result.
        :param _encoding: The encoding format.
        """
        selected = self.lxml.xpath(selector)
        elements: List[Union[Element, str]] = [
            Element(element=selection, url=self.url, default_encoding=_encoding or self.encoding) if not isinstance(selection, etree._ElementUnicodeResult) else str(selection)
            for selection in selected
        ]
        if clean:
            elements_copy = elements.copy()
            elements = []
            for element in elements_copy:
                if isinstance(element, Element):
                    element.raw_html = lxml_html_tostring(cleaner.clean_html(element.lxml))
                elements.append(element)
        return _get_first_or_list(elements, first)

    def search(self, template: str) -> Optional[_Search]:
        """Search the :class:`Element <Element>` for the given Parse template.

        :param template: The Parse template to use.
        """
        return parse_search(template, self.html)

    def search_all(self, template: str) -> List[Result]:
        """Search the :class:`Element <Element>` (multiple times) for the given parse
        template.

        :param template: The Parse template to use.
        """
        return [r for r in findall(template, self.html)]

    @property
    def links(self) -> _Links:
        """All found links on page, in as–is form."""

        def gen() -> Iterator[str]:
            for link in self.find('a') or []:
                try:
                    href = link.attrs['href'].strip()
                    if href and (not (href.startswith('#') and self.skip_anchors)) and (not href.startswith(('javascript:', 'mailto:'))):
                        yield href
                except KeyError:
                    pass
        return set(gen())

    def _make_absolute(self, link: str) -> str:
        """Makes a given link absolute."""
        parsed = urlparse(link)._asdict()
        if not parsed['netloc']:
            return urljoin(self.base_url, link)
        if not parsed['scheme']:
            parsed['scheme'] = urlparse(self.base_url).scheme
            parsed_iter = (v for v in parsed.values())
            return urlunparse(parsed_iter)
        return link

    @property
    def absolute_links(self) -> _Links:
        """All found links on page, in absolute form
        (`learn more <https://www.navegabem.com/absolute-or-relative-links.html>`_).
        """

        def gen() -> Iterator[str]:
            for link in self.links:
                yield self._make_absolute(link)
        return set(gen())

    @property
    def base_url(self) -> _URL:
        """The base URL for the page. Supports the ``<base>`` tag
        (`learn more <https://www.w3schools.com/tags/tag_base.asp>`_)."""
        base = self.find('base', first=True)
        if base:
            result = base.attrs.get('href', '').strip()
            if result:
                return result
        parsed = urlparse(self.url)._asdict()
        parsed['path'] = '/'.join(parsed['path'].split('/')[:-1]) + '/'
        parsed_iter = (v for v in parsed.values())
        url = urlunparse(parsed_iter)
        return url


class Element(BaseParser):
    """An element of HTML.

    :param element: The element from which to base the parsing upon.
    :param url: The URL from which the HTML originated, used for ``absolute_links``.
    :param default_encoding: Which encoding to default to.
    """
    __slots__ = ['element', 'url', 'skip_anchors', 'default_encoding', '_encoding', '_html', '_lxml', '_pq', '_attrs', 'session']

    def __init__(self, *, element: _LXML, url: _URL, default_encoding: Optional[_DefaultEncoding] = None) -> None:
        super(Element, self).__init__(element=element, url=url, default_encoding=default_encoding)
        self.element: _LXML = element
        self.tag: str = element.tag
        self.lineno: Optional[int] = element.sourceline
        self._attrs: Optional[MutableMapping[str, Any]] = None

    def __repr__(self) -> str:
        attrs = ['{}={}'.format(attr, repr(self.attrs[attr])) for attr in self.attrs]
        return '<Element {} {}>'.format(repr(self.element.tag), ' '.join(attrs))

    @property
    def attrs(self) -> MutableMapping[str, Any]:
        """Returns a dictionary of the attributes of the :class:`Element <Element>`
        (`learn more <https://www.w3schools.com/tags/ref_attributes.asp>`_).
        """
        if self._attrs is None:
            self._attrs = {k: v for k, v in self.element.items()}
            for attr in ['class', 'rel']:
                if attr in self._attrs:
                    self._attrs[attr] = tuple(self._attrs[attr].split())
        return self._attrs


class HTML(BaseParser):
    """An HTML document, ready for parsing.

    :param url: The URL from which the HTML originated, used for ``absolute_links``.
    :param html: HTML from which to base the parsing upon (optional).
    :param default_encoding: Which encoding to default to.
    """

    def __init__(self, *, session: Optional[Union['AsyncHTMLSession', 'HTMLSession']] = None, url: _URL = DEFAULT_URL, html: _HTML, default_encoding: _DefaultEncoding = DEFAULT_ENCODING, async_: bool = False) -> None:
        if isinstance(html, str):
            html = html.encode(DEFAULT_ENCODING)
        pq = PyQuery(html)
        super(HTML, self).__init__(element=pq('html') or pq.wrapAll('<html></html>')('html'), html=html, url=url, default_encoding=default_encoding)
        self.session: Union['AsyncHTMLSession', 'HTMLSession'] = session or (async_ and AsyncHTMLSession()) or HTMLSession()
        self.page: Optional[Any] = None
        self.next_symbol: _NextSymbol = DEFAULT_NEXT_SYMBOL

    def __repr__(self) -> str:
        return f'<HTML url={self.url!r}>'

    def next(self, fetch: bool = False, next_symbol: Optional[_NextSymbol] = None) -> Optional[Union[_URL, 'HTMLResponse']]:
        """Attempts to find the next page, if there is one. If ``fetch``
        is ``True`` (default), returns :class:`HTML <HTML>` object of
        next page. If ``fetch`` is ``False``, simply returns the next URL.

        """
        if next_symbol is None:
            next_symbol = DEFAULT_NEXT_SYMBOL

        def get_next() -> Optional[str]:
            candidates = self.find('a', containing=next_symbol)
            for candidate in candidates or []:
                if candidate.attrs.get('href'):
                    if 'next' in candidate.attrs.get('rel', []):
                        return candidate.attrs['href']
                    for _class in candidate.attrs.get('class', []):
                        if 'next' in _class:
                            return candidate.attrs['href']
                    if 'page' in candidate.attrs['href']:
                        return candidate.attrs['href']
            try:
                return (candidates or [])[-1].attrs['href']
            except IndexError:
                return None
        __next = get_next()
        if __next:
            url = self._make_absolute(__next)
        else:
            return None
        if fetch:
            return self.session.get(url)  # type: ignore[return-value]
        else:
            return url

    def __iter__(self) -> Iterator['HTML']:
        next: HTML = self
        while True:
            yield next
            try:
                next = next.next(fetch=True, next_symbol=self.next_symbol).html  # type: ignore[assignment]
            except AttributeError:
                break

    def __next__(self) -> 'HTML':
        return self.next(fetch=True, next_symbol=self.next_symbol).html  # type: ignore[return-value]

    def __aiter__(self) -> AsyncIterator['HTML']:
        return self

    async def __anext__(self) -> 'HTML':
        while True:
            url = self.next(fetch=False, next_symbol=self.next_symbol)
            if not url:
                break
            response = await self.session.get(url)  # type: ignore[union-attr]
            return response.html  # type: ignore[return-value]

    def add_next_symbol(self, next_symbol: str) -> None:
        self.next_symbol.append(next_symbol)

    async def _async_render(self, *, url: _URL, script: Optional[str] = None, scrolldown: Union[int, bool] = False, sleep: Union[int, float] = 0, wait: float, reload: bool, content: Optional[_BaseHTML], timeout: float, keep_page: bool, cookies: List[Dict[str, Any]] = [{}]) -> Optional[Tuple[str, Any, Any]]:
        """ Handle page creation and js rendering. Internal use for render/arender methods. """
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
                for _ in range(scrolldown):  # type: ignore[arg-type]
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
        """
        Convert HTMLSession.cookies:cookiejar[] for browser.newPage().setCookie
        """
        cookie_render: Dict[str, Any] = {}

        def __convert(cookiejar: http.cookiejar.Cookie, key: str) -> Dict[str, Any]:
            try:
                v = eval('cookiejar.' + key)
                if not v:
                    kv: Dict[str, Any] = ''
                else:
                    kv = {key: v}
            except:
                kv = ''
            return kv  # type: ignore[return-value]
        keys = ['name', 'value', 'url', 'domain', 'path', 'sameSite', 'expires', 'httpOnly', 'secure']
        for key in keys:
            cookie_render.update(__convert(session_cookiejar, key))
        return cookie_render

    def _convert_cookiesjar_to_render(self) -> List[Dict[str, Any]]:
        """
        Convert HTMLSession.cookies for browser.newPage().setCookie
        Return a list of dict
        """
        cookies_render: List[Dict[str, Any]] = []
        if isinstance(self.session.cookies, http.cookiejar.CookieJar):
            for cookie in self.session.cookies:
                cookies_render.append(self._convert_cookiejar_to_render(cookie))
        return cookies_render

    def render(self, retries: int = 8, script: Optional[str] = None, wait: float = 0.2, scrolldown: Union[int, bool] = False, sleep: Union[int, float] = 0, reload: bool = True, timeout: float = 8.0, keep_page: bool = False, cookies: List[Dict[str, Any]] = [{}], send_cookies_session: bool = False) -> Optional[Any]:
        '''Reloads the response in Chromium, and replaces HTML content
        with an updated version, with JavaScript executed.
        '''
        self.browser = self.session.browser  # type: ignore[assignment]
        content: Optional[str] = None
        if self.url == DEFAULT_URL:
            reload = False
        if send_cookies_session:
            cookies = self._convert_cookiesjar_to_render()
        for i in range(retries):
            if not content:
                try:
                    content, result, page = self.session.loop.run_until_complete(self._async_render(url=self.url, script=script, sleep=sleep, wait=wait, content=self.html, reload=reload, scrolldown=scrolldown, timeout=timeout, keep_page=keep_page, cookies=cookies))  # type: ignore[attr-defined]
                except TypeError:
                    pass
            else:
                break
        if not content:
            raise MaxRetries('Unable to render the page. Try increasing timeout')
        html = HTML(url=self.url, html=content.encode(DEFAULT_ENCODING), default_encoding=DEFAULT_ENCODING)
        self.__dict__.update(html.__dict__)
        self.page = page  # type: ignore[has-type]
        return result  # type: ignore[has-type]

    async def arender(self, retries: int = 8, script: Optional[str] = None, wait: float = 0.2, scrolldown: Union[int, bool] = False, sleep: Union[int, float] = 0, reload: bool = True, timeout: float = 8.0, keep_page: bool = False, cookies: List[Dict[str, Any]] = [{}], send_cookies_session: bool = False) -> Optional[Any]:
        """ Async version of render. Takes same parameters. """
        self.browser = await self.session.browser  # type: ignore[assignment]
        content: Optional[str] = None
        if self.url == DEFAULT_URL:
            reload = False
        if send_cookies_session:
            cookies = self._convert_cookiesjar_to_render()
        for _ in range(retries):
            if not content:
                try:
                    content, result, page = await self._async_render(url=self.url, script=script, sleep=sleep, wait=wait, content=self.html, reload=reload, scrolldown=scrolldown, timeout=timeout, keep_page=keep_page, cookies=cookies)
                except TypeError:
                    pass
            else:
                break
        if not content:
            raise MaxRetries('Unable to render the page. Try increasing timeout')
        html = HTML(url=self.url, html=content.encode(DEFAULT_ENCODING), default_encoding=DEFAULT_ENCODING)
        self.__dict__.update(html.__dict__)
        self.page = page  # type: ignore[has-type]
        return result  # type: ignore[has-type]


class HTMLResponse(requests.Response):
    """An HTML-enabled :class:`requests.Response <requests.Response>` object.
    Effectively the same, but with an intelligent ``.html`` property added.
    """

    def __init__(self, session: 'BaseSession') -> None:
        super(HTMLResponse, self).__init__()
        self._html: Optional[HTML] = None
        self.session: BaseSession = session

    @property
    def html(self) -> HTML:
        if not self._html:
            self._html = HTML(session=self.session, url=self.url, html=self.content, default_encoding=self.encoding)  # type: ignore[arg-type]
        return self._html

    @classmethod
    def _from_response(cls, response: requests.Response, session: 'BaseSession') -> 'HTMLResponse':
        html_r = cls(session=session)
        html_r.__dict__.update(response.__dict__)
        return html_r


def user_agent(style: Optional[_UserAgent] = None) -> str:
    """Returns an apparently legit user-agent, if not requested one of a specific
    style. Defaults to a Chrome-style User-Agent.
    """
    global useragent
    if not useragent and style:
        useragent = UserAgent()
    return useragent[style] if style else DEFAULT_USER_AGENT  # type: ignore[index]


def _get_first_or_list(l: List[Any], first: bool = False) -> Optional[Any]:
    if first:
        try:
            return l[0]
        except IndexError:
            return None
    else:
        return l


class BaseSession(requests.Session):
    """ A consumable session, for cookie persistence and connection pooling,
    amongst other things.
    """

    def __init__(self, mock_browser: bool = True, verify: bool = True, browser_args: List[str] = ['--no-sandbox']) -> None:
        super().__init__()
        if mock_browser:
            self.headers['User-Agent'] = user_agent()
        self.hooks['response'].append(self.response_hook)
        self.verify: bool = verify
        self.__browser_args: List[str] = browser_args

    def response_hook(self, response: requests.Response, **kwargs: Any) -> 'HTMLResponse':
        """ Change response encoding and replace it by a HTMLResponse. """
        if not response.encoding:
            response.encoding = DEFAULT_ENCODING
        return HTMLResponse._from_response(response, self)

    @property
    async def browser(self) -> Any:
        if not hasattr(self, '_browser'):
            self._browser = await pyppeteer.launch(ignoreHTTPSErrors=not self.verify, headless=True, args=self.__browser_args)
        return self._browser


class HTMLSession(BaseSession):

    def __init__(self, **kwargs: Any) -> None:
        super(HTMLSession, self).__init__(**kwargs)

    @property
    def browser(self) -> Any:
        if not hasattr(self, '_browser'):
            self.loop = asyncio.get_event_loop()
            if self.loop.is_running():
                raise RuntimeError('Cannot use HTMLSession within an existing event loop. Use AsyncHTMLSession instead.')
            self._browser = self.loop.run_until_complete(super().browser)  # type: ignore[misc]
        return self._browser

    def close(self) -> None:
        """ If a browser was created close it first. """
        if hasattr(self, '_browser'):
            self.loop.run_until_complete(self._browser.close())
        super().close()


class AsyncHTMLSession(BaseSession):
    """ An async consumable session. """

    def __init__(self, loop: Optional[asyncio.AbstractEventLoop] = None, workers: Optional[int] = None, mock_browser: bool = True, *args: Any, **kwargs: Any) -> None:
        """ Set or create an event loop and a thread pool.

            :param loop: Asyncio loop to use.
            :param workers: Amount of threads to use for executing async calls.
                If not pass it will default to the number of processors on the
                machine, multiplied by 5. """
        super().__init__(*args, mock_browser=mock_browser, **kwargs)
        self.loop: asyncio.AbstractEventLoop = loop or asyncio.get_event_loop()
        self.thread_pool: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=workers)

    def request(self, *args: Any, **kwargs: Any) -> asyncio.Future:
        """ Partial original request func and run it in a thread. """
        func = partial(super().request, *args, **kwargs)
        return self.loop.run_in_executor(self.thread_pool, func)  # type: ignore[return-value]

    async def close(self) -> None:
        """ If a browser was created close it first. """
        if hasattr(self, '_browser'):
            await self._browser.close()
        super().close()

    def run(self, *coros: Callable[[], Awaitable[Any]]) -> List[Any]:
        """ Pass in all the coroutines you want to run, it will wrap each one
            in a task, run it and wait for the result. Return a list with all
            results, this is returned in the same order coros are passed in. """
        tasks = [asyncio.ensure_future(coro()) for coro in coros]
        done, _ = self.loop.run_until_complete(asyncio.wait(tasks))
        return [t.result() for t in done]