import sys
import asyncio
from urllib.parse import urlparse, urlunparse, urljoin
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures._base import TimeoutError
from functools import partial
from typing import Set, Union, List, MutableMapping, Optional, Any, Dict, Coroutine
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
DEFAULT_USER_AGENT: str = (
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/603.3.8 (KHTML, like Gecko) Version/10.1.2 Safari/603.3.8'
    )
DEFAULT_NEXT_SYMBOL: List[str] = ['next', 'more', 'older']
cleaner: Cleaner = Cleaner()
cleaner.javascript = True
cleaner.style = True
useragent: Optional[UserAgent] = None
_Find = Union[List['Element'], 'Element']
_XPath = Union[List[str], List[Any], str, Any]
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
_Attrs = MutableMapping
_Next = Union['HTML', List[str], None]
_NextSymbol = List[str]
try:
    assert sys.version_info.major == 3
    assert sys.version_info.minor > 5
except AssertionError:
    raise RuntimeError('Requests-HTML requires Python 3.6+!')


class MaxRetries(Exception):

    def __init__(self, message):
        self.message: str = message


class BaseParser:
    """A basic HTML/Element Parser, for Humans.

    :param element: The element from which to base the parsing upon.
    :param default_encoding: Which encoding to default to.
    :param html: HTML from which to base the parsing upon (optional).
    :param url: The URL from which the HTML originated, used for ``absolute_links``.

    """

    def __init__(self, *, element: Any, default_encoding: Optional[
        _DefaultEncoding]=None, html: Optional[_HTML]=None, url: _URL):
        self.element: Any = element
        self.url: str = url
        self.skip_anchors: bool = True
        self.default_encoding: Optional[str] = default_encoding
        self._encoding: Optional[str] = None
        self._html: Optional[bytes] = html.encode(DEFAULT_ENCODING
            ) if isinstance(html, str) else html
        self._lxml: Optional[HtmlElement] = None
        self._pq: Optional[PyQuery] = None

    @property
    def raw_html(self):
        """Bytes representation of the HTML content.
        (`learn more <http://www.diveintopython3.net/strings.html>`_).
        """
        if self._html:
            return self._html
        else:
            return etree.tostring(self.element, encoding='unicode').strip(
                ).encode(self.encoding)

    @property
    def html(self):
        """Unicode representation of the HTML content
        (`learn more <http://www.diveintopython3.net/strings.html>`_).
        """
        if self._html:
            return self.raw_html.decode(self.encoding, errors='replace')
        else:
            return etree.tostring(self.element, encoding='unicode').strip()

    @html.setter
    def html(self, html):
        self._html = html.encode(self.encoding)

    @raw_html.setter
    def raw_html(self, html):
        """Property setter for self.html."""
        self._html = html

    @property
    def encoding(self):
        """The encoding string to be used, extracted from the HTML and
        :class:`HTMLResponse <HTMLResponse>` headers.
        """
        if self._encoding:
            return self._encoding
        if self._html:
            self._encoding = html_to_unicode(self.default_encoding, self._html
                )[0]
            try:
                self.raw_html.decode(self.encoding, errors='replace')
            except UnicodeDecodeError:
                self._encoding = self.default_encoding
        return self._encoding if self._encoding else self.default_encoding

    @encoding.setter
    def encoding(self, enc):
        """Property setter for self.encoding."""
        self._encoding = enc

    @property
    def pq(self):
        """`PyQuery <https://pythonhosted.org/pyquery/>`_ representation
        of the :class:`Element <Element>` or :class:`HTML <HTML>`.
        """
        if self._pq is None:
            self._pq = PyQuery(self.lxml)
        return self._pq

    @property
    def lxml(self):
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
    def text(self):
        """The text content of the
        :class:`Element <Element>` or :class:`HTML <HTML>`.
        """
        return self.pq.text()

    @property
    def full_text(self):
        """The full text content (including links) of the
        :class:`Element <Element>` or :class:`HTML <HTML>`.
        """
        return self.lxml.text_content()

    def find(self, selector='*', *, containing: _Containing=None, clean:
        bool=False, first: bool=False, _encoding: Optional[str]=None):
        """Given a CSS Selector, returns a list of
        :class:`Element <Element>` objects or a single one.

        :param selector: CSS Selector to use.
        :param clean: Whether or not to sanitize the found HTML of ``<script>`` and ``<style>`` tags.
        :param containing: If specified, only return elements that contain the provided text.
        :param first: Whether or not to return just the first result.
        :param _encoding: The encoding format.

        Example CSS Selectors:

        - ``a``
        - ``a.someClass``
        - ``a#someID``
        - ``a[target=_blank]``

        See W3School's `CSS Selectors Reference
        <https://www.w3schools.com/cssref/css_selectors.asp>`_
        for more details.

        If ``first`` is ``True``, only returns the first
        :class:`Element <Element>` found.
        """
        if isinstance(containing, str):
            containing = [containing]
        encoding = _encoding or self.encoding
        elements: List[Element] = [Element(element=found, url=self.url,
            default_encoding=encoding) for found in self.pq(selector)]
        if containing:
            elements_copy = elements.copy()
            elements = []
            for element in elements_copy:
                if any([(c.lower() in element.full_text.lower()) for c in
                    containing]):
                    elements.append(element)
            elements.reverse()
        if clean:
            elements_copy = elements.copy()
            elements = []
            for element in elements_copy:
                element.raw_html = lxml_html_tostring(cleaner.clean_html(
                    element.lxml))
                elements.append(element)
        return _get_first_or_list(elements, first)

    def xpath(self, selector, *, clean: bool=False, first: bool=False,
        _encoding: Optional[str]=None):
        """Given an XPath selector, returns a list of
        :class:`Element <Element>` objects or a single one.

        :param selector: XPath Selector to use.
        :param clean: Whether or not to sanitize the found HTML of ``<script>`` and ``<style>`` tags.
        :param first: Whether or not to return just the first result.
        :param _encoding: The encoding format.

        If a sub-selector is specified (e.g. ``//a/@href``), a simple
        list of results is returned.

        See W3School's `XPath Examples
        <https://www.w3schools.com/xml/xpath_examples.asp>`_
        for more details.

        If ``first`` is ``True``, only returns the first
        :class:`Element <Element>` found.
        """
        selected = self.lxml.xpath(selector)
        elements: List[Union[Element, str]] = [(Element(element=selection,
            url=self.url, default_encoding=_encoding or self.encoding) if 
            not isinstance(selection, etree._ElementUnicodeResult) else str
            (selection)) for selection in selected]
        if clean:
            elements_copy = elements.copy()
            elements = []
            for element in elements_copy:
                if isinstance(element, Element):
                    element.raw_html = lxml_html_tostring(cleaner.
                        clean_html(element.lxml))
                elements.append(element)
        return _get_first_or_list(elements, first)

    def search(self, template):
        """Search the :class:`Element <Element>` for the given Parse template.

        :param template: The Parse template to use.
        """
        return parse_search(template, self.html)

    def search_all(self, template):
        """Search the :class:`Element <Element>` (multiple times) for the given parse
        template.

        :param template: The Parse template to use.
        """
        return [r for r in findall(template, self.html)]

    @property
    def links(self):
        """All found links on page, in as–is form."""

        def gen():
            for link in self.find('a'):
                try:
                    href: str = link.attrs['href'].strip()
                    if href and not (href.startswith('#') and self.skip_anchors
                        ) and not href.startswith(('javascript:', 'mailto:')):
                        yield href
                except KeyError:
                    pass
        return set(gen())

    def _make_absolute(self, link):
        """Makes a given link absolute."""
        parsed = urlparse(link)._asdict()
        if not parsed['netloc']:
            return urljoin(self.base_url, link)
        if not parsed['scheme']:
            parsed['scheme'] = urlparse(self.base_url).scheme
            parsed_values = [parsed[key] for key in ['scheme', 'netloc',
                'path', 'params', 'query', 'fragment']]
            return urlunparse(parsed_values)
        return link

    @property
    def absolute_links(self):
        """All found links on page, in absolute form
        (`learn more <https://www.navegabem.com/absolute-or-relative-links.html>`_).
        """

        def gen():
            for link in self.links:
                yield self._make_absolute(link)
        return set(gen())

    @property
    def base_url(self):
        """The base URL for the page. Supports the ``<base>`` tag
        (`learn more <https://www.w3schools.com/tags/tag_base.asp>`_)."""
        base = self.find('base', first=True)
        if base:
            result = base.attrs.get('href', '').strip()
            if result:
                return result
        parsed = urlparse(self.url)._asdict()
        parsed['path'] = '/'.join(parsed['path'].split('/')[:-1]) + '/'
        parsed_values = [parsed[key] for key in ['scheme', 'netloc', 'path',
            'params', 'query', 'fragment']]
        url = urlunparse(parsed_values)
        return url


class Element(BaseParser):
    """An element of HTML.

    :param element: The element from which to base the parsing upon.
    :param url: The URL from which the HTML originated, used for ``absolute_links``.
    :param default_encoding: Which encoding to default to.
    """
    __slots__ = ['element', 'url', 'skip_anchors', 'default_encoding',
        '_encoding', '_html', '_lxml', '_pq', '_attrs', 'session', 'tag',
        'lineno']

    def __init__(self, *, element: Any, url: _URL, default_encoding:
        Optional[_DefaultEncoding]=None):
        super(Element, self).__init__(element=element, url=url,
            default_encoding=default_encoding)
        self.element: Any = element
        self.tag: str = element.tag
        self.lineno: Optional[int] = element.sourceline
        self._attrs: Optional[_Attrs] = None

    def __repr__(self):
        attrs_repr = ['{}={}'.format(attr, repr(self.attrs[attr])) for attr in
            self.attrs]
        return '<Element {} {}>'.format(repr(self.element.tag), ' '.join(
            attrs_repr))

    @property
    def attrs(self):
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

    def __init__(self, *, session: Optional[Union['HTMLSession',
        'AsyncHTMLSession']]=None, url: str=DEFAULT_URL, html: _HTML,
        default_encoding: str=DEFAULT_ENCODING, async_: bool=False):
        if isinstance(html, str):
            html = html.encode(DEFAULT_ENCODING)
        pq = PyQuery(html)
        super(HTML, self).__init__(element=pq('html') or pq.wrapAll(
            '<html></html>')('html'), html=html, url=url, default_encoding=
            default_encoding)
        self.session: Union['HTMLSession', 'AsyncHTMLSession'
            ] = session or async_ and AsyncHTMLSession() or HTMLSession()
        self.page: Optional[Any] = None
        self.next_symbol: List[str] = DEFAULT_NEXT_SYMBOL

    def __repr__(self):
        return f'<HTML url={self.url!r}>'

    def next(self, fetch=False, next_symbol=None):
        """Attempts to find the next page, if there is one. If ``fetch``
        is ``True`` (default), returns :class:`HTML <HTML>` object of
        next page. If ``fetch`` is ``False``, simply returns the next URL.

        """
        if next_symbol is None:
            next_symbol = DEFAULT_NEXT_SYMBOL

        def get_next():
            candidates: List[Element] = self.find('a', containing=next_symbol)
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
            return self.session.get(url)
        else:
            return url

    def __iter__(self):
        next_obj: 'HTML' = self
        while True:
            yield next_obj
            try:
                next_obj = next_obj.next(fetch=True, next_symbol=self.
                    next_symbol).html
            except AttributeError:
                break

    def __next__(self):
        next_html: 'HTML' = self.next(fetch=True, next_symbol=self.next_symbol
            ).html
        return next_html

    def __aiter__(self):
        return self

    async def __anext__(self) ->'HTML':
        while True:
            url: Optional[str] = self.next(fetch=False, next_symbol=self.
                next_symbol)
            if not url:
                raise StopAsyncIteration
            response: HTMLResponse = await self.session.get(url)
            return response.html

    def add_next_symbol(self, next_symbol):
        self.next_symbol.append(next_symbol)

    async def _async_render(self, *, url: str, script: Optional[str]=None,
        scrolldown: Optional[int]=None, sleep: int, wait: float, reload:
        bool, content: Optional[str], timeout: Union[float, int], keep_page:
        bool, cookies: list=[{}]) ->Optional[tuple]:
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
                await page.goto(f'data:text/html,{self.html}', options={
                    'timeout': int(timeout * 1000)})
            result: Optional[Any] = None
            if script:
                result = await page.evaluate(script)
            if scrolldown:
                for _ in range(scrolldown):
                    await page.keyboard.down('PageDown')
                    await asyncio.sleep(sleep)
            else:
                await asyncio.sleep(sleep)
            if scrolldown:
                await page.keyboard.up('PageDown')
            content_rendered: str = await page.content()
            if not keep_page:
                await page.close()
                page = None
            return content_rendered, result, page
        except TimeoutError:
            if 'page' in locals() and page:
                await page.close()
            return None

    def _convert_cookiejar_to_render(self, session_cookiejar):
        """
        Convert HTMLSession.cookies:cookiejar[] for browser.newPage().setCookie
        """
        cookie_render: Dict[str, Any] = {}

        def __convert(cookiejar, key):
            try:
                v = getattr(cookiejar, key)
                if not v:
                    kv = ''
                else:
                    kv = {key: v}
            except AttributeError:
                kv = ''
            return kv
        keys = ['name', 'value', 'url', 'domain', 'path', 'sameSite',
            'expires', 'httpOnly', 'secure']
        for key in keys:
            kv = __convert(session_cookiejar, key)
            if isinstance(kv, dict):
                cookie_render.update(kv)
        return cookie_render

    def _convert_cookiesjar_to_render(self):
        """
        Convert HTMLSession.cookies for browser.newPage().setCookie
        Return a list of dict
        """
        cookies_render: List[Dict[str, Any]] = []
        if isinstance(self.session.cookies, http.cookiejar.CookieJar):
            for cookie in self.session.cookies:
                cookies_render.append(self._convert_cookiejar_to_render(cookie)
                    )
        return cookies_render

    def render(self, retries=8, script=None, wait=0.2, scrolldown=False,
        sleep=0, reload=True, timeout=8.0, keep_page=False, cookies=[{}],
        send_cookies_session=False):
        """Reloads the response in Chromium, and replaces HTML content
        with an updated version, with JavaScript executed.

        :param retries: The number of times to retry loading the page in Chromium.
        :param script: JavaScript to execute upon page load (optional).
        :param wait: The number of seconds to wait before loading the page, preventing timeouts (optional).
        :param scrolldown: Integer, if provided, of how many times to page down.
        :param sleep: Integer, if provided, of how many seconds to sleep after initial render.
        :param reload: If ``False``, content will not be loaded from the browser, but will be provided from memory.
        :param keep_page: If ``True`` will allow you to interact with the browser page through ``r.html.page``.

        :param send_cookies_session: If ``True`` send ``HTMLSession.cookies`` convert.
        :param cookies: If not ``empty`` send ``cookies``.

        If ``scrolldown`` is specified, the page will scrolldown the specified
        number of times, after sleeping the specified amount of time
        (e.g. ``scrolldown=10, sleep=1``).

        If just ``sleep`` is provided, the rendering will wait *n* seconds, before
        returning.

        If ``script`` is specified, it will execute the provided JavaScript at
        runtime. Example:

        .. code-block:: python

            script = ""\"
                () => {
                    return {
                        width: document.documentElement.clientWidth,
                        height: document.documentElement.clientHeight,
                        deviceScaleFactor: window.devicePixelRatio,
                    }
                }
            ""\"

        Returns the return value of the executed  ``script``, if any is provided:

        .. code-block:: python

            >>> r.html.render(script=script)
            {'width': 800, 'height': 600, 'deviceScaleFactor': 1}

        Warning: the first time you run this method, it will download
        Chromium into your home directory (``~/.pyppeteer``).
        """
        self.browser = self.session.browser
        content: Optional[str] = None
        if self.url == DEFAULT_URL:
            reload = False
        if send_cookies_session:
            cookies = self._convert_cookiesjar_to_render()
        for i in range(retries):
            if not content:
                try:
                    render_result = self.session.loop.run_until_complete(self
                        ._async_render(url=self.url, script=script, sleep=
                        sleep, wait=wait, content=self.html, reload=reload,
                        scrolldown=scrolldown, timeout=timeout, keep_page=
                        keep_page, cookies=cookies))
                    if render_result:
                        content, result, page = render_result
                except TypeError:
                    pass
            else:
                break
        if not content:
            raise MaxRetries(
                'Unable to render the page. Try increasing timeout')
        html = HTML(url=self.url, html=content.encode(DEFAULT_ENCODING),
            default_encoding=DEFAULT_ENCODING)
        self.__dict__.update(html.__dict__)
        self.page = page
        return result

    async def arender(self, retries: int=8, script: Optional[str]=None,
        wait: float=0.2, scrolldown: Optional[int]=False, sleep: int=0,
        reload: bool=True, timeout: Union[float, int]=8.0, keep_page: bool=
        False, cookies: list=[{}], send_cookies_session: bool=False
        ) ->Optional[Any]:
        """ Async version of render. Takes same parameters. """
        self.browser = await self.session.browser
        content: Optional[str] = None
        if self.url == DEFAULT_URL:
            reload = False
        if send_cookies_session:
            cookies = self._convert_cookiesjar_to_render()
        for _ in range(retries):
            if not content:
                try:
                    render_result = await self._async_render(url=self.url,
                        script=script, sleep=sleep, wait=wait, content=self
                        .html, reload=reload, scrolldown=scrolldown,
                        timeout=timeout, keep_page=keep_page, cookies=cookies)
                    if render_result:
                        content, result, page = render_result
                except TypeError:
                    pass
            else:
                break
        if not content:
            raise MaxRetries(
                'Unable to render the page. Try increasing timeout')
        html = HTML(url=self.url, html=content.encode(DEFAULT_ENCODING),
            default_encoding=DEFAULT_ENCODING)
        self.__dict__.update(html.__dict__)
        self.page = page
        return result


class HTMLResponse(requests.Response):
    """An HTML-enabled :class:`requests.Response <requests.Response>` object.
    Effectively the same, but with an intelligent ``.html`` property added.
    """

    def __init__(self, session):
        super(HTMLResponse, self).__init__()
        self._html: Optional[HTML] = None
        self.session: Union['HTMLSession', 'AsyncHTMLSession'] = session

    @property
    def html(self):
        if not self._html:
            self._html = HTML(session=self.session, url=self.url, html=self
                .content, default_encoding=self.encoding)
        return self._html

    @classmethod
    def _from_response(cls, response, session):
        html_r: 'HTMLResponse' = cls(session=session)
        html_r.__dict__.update(response.__dict__)
        return html_r


def user_agent(style=None):
    """Returns an apparently legit user-agent, if not requested one of a specific
    style. Defaults to a Chrome-style User-Agent.
    """
    global useragent
    if not useragent and style:
        useragent = UserAgent()
    return useragent[style] if style else DEFAULT_USER_AGENT


def _get_first_or_list(l, first=False):
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

    def __init__(self, mock_browser=True, verify=True, browser_args=None):
        super().__init__()
        if browser_args is None:
            browser_args = ['--no-sandbox']
        if mock_browser:
            self.headers['User-Agent'] = user_agent()
        self.hooks['response'].append(self.response_hook)
        self.verify: bool = verify
        self.__browser_args: List[str] = browser_args

    def response_hook(self, response, **kwargs: Any):
        """ Change response encoding and replace it by a HTMLResponse. """
        if not response.encoding:
            response.encoding = DEFAULT_ENCODING
        return HTMLResponse._from_response(response, self)

    @property
    async def browser(self) ->pyppeteer.browser.Browser:
        if not hasattr(self, '_browser'):
            self._browser = await pyppeteer.launch(ignoreHTTPSErrors=not
                self.verify, headless=True, args=self.__browser_args)
        return self._browser


class HTMLSession(BaseSession):

    def __init__(self, **kwargs: Any):
        super(HTMLSession, self).__init__(**kwargs)
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    @property
    def browser(self):
        if not hasattr(self, '_browser'):
            self.loop = asyncio.get_event_loop()
            if self.loop.is_running():
                raise RuntimeError(
                    'Cannot use HTMLSession within an existing event loop. Use AsyncHTMLSession instead.'
                    )
            self._browser = self.loop.run_until_complete(super().browser)
        return self._browser

    def close(self):
        """ If a browser was created close it first. """
        if hasattr(self, '_browser'):
            self.loop.run_until_complete(self._browser.close())
        super().close()


class AsyncHTMLSession(BaseSession):
    """ An async consumable session. """

    def __init__(self, loop=None, workers=None, mock_browser=True, *args:
        Any, **kwargs: Any):
        """ Set or create an event loop and a thread pool.

            :param loop: Asyncio loop to use.
            :param workers: Amount of threads to use for executing async calls.
                If not pass it will default to the number of processors on the
                machine, multiplied by 5. """
        super().__init__(mock_browser=mock_browser, **kwargs)
        self.loop: asyncio.AbstractEventLoop = loop or asyncio.get_event_loop()
        self.thread_pool: ThreadPoolExecutor = ThreadPoolExecutor(max_workers
            =workers)

    def request(self, *args: Any, **kwargs: Any):
        """ Partial original request func and run it in a thread. """
        func = partial(super().request, *args, **kwargs)
        return self.loop.run_in_executor(self.thread_pool, func)

    async def close(self) ->None:
        """ If a browser was created close it first. """
        if hasattr(self, '_browser'):
            await self._browser.close()
        super().close()

    def run(self, *coros: Any):
        """ Pass in all the coroutines you want to run, it will wrap each one
            in a task, run it and wait for the result. Return a list with all
            results, this is returned in the same order coros are passed in. """
        tasks = [asyncio.ensure_future(coro()) for coro in coros]
        done, _ = self.loop.run_until_complete(asyncio.wait(tasks))
        return [t.result() for t in done]
