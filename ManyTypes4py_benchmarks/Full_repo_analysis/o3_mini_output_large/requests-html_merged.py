############################################
# File: docs/source/conf.py
############################################
from typing import List, Optional
import requests_html

project: str = 'requests-HTML'
copyright: str = u'MMXVIII. A <a href="http://kennethreitz.com/pages/open-projects.html">Kenneth Reitz</a> Project'
author: str = 'Kenneth Reitz'
version: str = ''
release: str = 'v0.3.4'
extensions: List[str] = ['sphinx.ext.autodoc',
                         'sphinx.ext.doctest',
                         'sphinx.ext.intersphinx',
                         'sphinx.ext.todo',
                         'sphinx.ext.coverage',
                         'sphinx.ext.viewcode',
                         'sphinx.ext.githubpages']
templates_path: List[str] = ['_templates']
source_suffix: str = '.rst'
master_doc: str = 'index'
language: Optional[str] = None
exclude_patterns: List[str] = []
pygments_style: str = 'sphinx'
html_theme: str = 'alabaster'
html_theme_options: dict = {'show_powered_by': False,
                            'github_user': 'psf',
                            'github_repo': 'requests-html',
                            'github_banner': True,
                            'show_related': False,
                            'note_bg': '#FFF59C'}
html_static_path: List[str] = ['_static']
html_sidebars: dict = {'index': ['sidebarintro.html', 'sourcelink.html', 'searchbox.html', 'hacks.html'],
                         '**': ['sidebarlogo.html', 'localtoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html', 'hacks.html']}
html_show_sphinx: bool = False
html_show_sourcelink: bool = False
htmlhelp_basename: str = 'requests-htmldoc'
latex_elements: dict = {}
latex_documents: List[tuple] = [(master_doc, 'requests-html.tex', 'requests-html Documentation', 'Kenneth Reitz', 'manual')]
man_pages: List[tuple] = [(master_doc, 'requests-html', 'requests-html Documentation', [author], 1)]
texinfo_documents: List[tuple] = [(master_doc, 'requests-html', 'requests-html Documentation', author, 'requests-html', 'One line description of project.', 'Miscellaneous')]
epub_title: str = project
epub_author: str = author
epub_publisher: str = author
epub_copyright: str = copyright
epub_exclude_files: List[str] = ['search.html']
intersphinx_mapping: dict = {'https://docs.python.org/': None}
todo_include_todos: bool = True

############################################
# File: requests_html.py
############################################
from __future__ import annotations
import sys
import asyncio
from urllib.parse import urlparse, urlunparse, urljoin
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures._base import TimeoutError
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union
import pyppeteer
import requests
import http.cookiejar
from pyquery import PyQuery
from fake_useragent import UserAgent
from lxml.html.clean import Cleaner
import lxml
from lxml import etree
from lxml.html import HtmlElement, tostring as lxml_html_tostring
from lxml.html.soupparser import fromstring as soup_parse
from parse import search as parse_search
from parse import findall, Result
from w3lib.encoding import html_to_unicode

T = TypeVar('T')

DEFAULT_ENCODING: str = 'utf-8'
DEFAULT_URL: str = 'https://example.org/'
DEFAULT_USER_AGENT: str = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/603.3.8 (KHTML, like Gecko) Version/10.1.2 Safari/603.3.8'
DEFAULT_NEXT_SYMBOL: List[str] = ['next', 'more', 'older']
cleaner: Cleaner = Cleaner()
cleaner.javascript = True
cleaner.style = True
useragent: Optional[UserAgent] = None
_Find = Union[List[Element], Element]
_XPath = Union[List[str], List[Element], str, Element]
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
_Attrs = Dict[str, Any]
_Next = Union[HTML, List[str]]
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
    """
    A basic HTML/Element Parser, for Humans.
    """

    def __init__(self, *, element: Any, default_encoding: Optional[str] = None, html: Optional[Union[str, bytes]] = None, url: str) -> None:
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
            self._encoding, _ = html_to_unicode(self.default_encoding, self._html)
            try:
                self.raw_html.decode(self.encoding, errors='replace')
            except UnicodeDecodeError:
                self._encoding = self.default_encoding  # type: ignore
        return self._encoding if self._encoding else self.default_encoding  # type: ignore

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

    def find(self, selector: str = '*', *, containing: Optional[Union[str, List[str]]] = None, clean: bool = False, first: bool = False, _encoding: Optional[str] = None) -> Union[Element, List[Element]]:
        encoding: str = _encoding or self.encoding
        elements: List[Element] = [Element(element=found, url=self.url, default_encoding=encoding) for found in self.pq(selector)]
        if containing:
            if isinstance(containing, str):
                containing = [containing]
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

    def xpath(self, selector: str, *, clean: bool = False, first: bool = False, _encoding: Optional[str] = None) -> Union[Element, List[Element], List[str]]:
        selected = self.lxml.xpath(selector)
        elements: List[Union[Element, str]] = [
            Element(element=selection, url=self.url, default_encoding=_encoding or self.encoding) if not isinstance(selection, etree._ElementUnicodeResult) else str(selection) for selection in selected
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

    def search(self, template: str) -> Optional[Result]:
        return parse_search(template, self.html)

    def search_all(self, template: str) -> List[Result]:
        return [r for r in findall(template, self.html)]

    @property
    def links(self) -> Set[str]:
        def gen() -> Any:
            for link in self.find('a'):
                try:
                    href: str = link.attrs['href'].strip()
                    if href and ((not (href.startswith('#') and self.skip_anchors))
                                 and (not href.startswith(('javascript:', 'mailto:')))):
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
            parsed_values = tuple(v for v in parsed.values())
            return urlunparse(parsed_values)
        return link

    @property
    def absolute_links(self) -> Set[str]:
        def gen() -> Any:
            for link in self.links:
                yield self._make_absolute(link)
        return set(gen())

    @property
    def base_url(self) -> str:
        base: Optional[Element] = self.find('base', first=True)
        if base:
            result: str = base.attrs.get('href', '').strip()
            if result:
                return result
        parsed = urlparse(self.url)._asdict()
        parsed['path'] = '/'.join(parsed['path'].split('/')[:-1]) + '/'
        parsed_values = tuple(v for v in parsed.values())
        url_str: str = urlunparse(parsed_values)
        return url_str

class Element(BaseParser):
    __slots__ = ['element', 'url', 'skip_anchors', 'default_encoding', '_encoding', '_html', '_lxml', '_pq', '_attrs', 'session', 'tag', 'lineno']

    def __init__(self, *, element: Any, url: str, default_encoding: Optional[str] = None) -> None:
        super(Element, self).__init__(element=element, url=url, default_encoding=default_encoding)
        self.element = element
        self.tag: str = element.tag
        self.lineno: Optional[int] = element.sourceline
        self._attrs: Optional[Dict[str, Any]] = None

    def __repr__(self) -> str:
        attrs = ['{}={}'.format(attr, repr(self.attrs[attr])) for attr in self.attrs]
        return '<Element {} {}>'.format(repr(self.element.tag), ' '.join(attrs))

    @property
    def attrs(self) -> Dict[str, Any]:
        if self._attrs is None:
            self._attrs = {k: v for (k, v) in self.element.items()}
            for attr in ['class', 'rel']:
                if attr in self._attrs:
                    self._attrs[attr] = tuple(self._attrs[attr].split())
        return self._attrs

class HTML(BaseParser):
    def __init__(self, *, session: Optional[Union[HTMLSession, AsyncHTMLSession]] = None, url: str = DEFAULT_URL, html: Union[str, bytes], default_encoding: str = DEFAULT_ENCODING, async_: bool = False) -> None:
        if isinstance(html, str):
            html = html.encode(DEFAULT_ENCODING)
        pq = PyQuery(html)
        element = pq('html') or pq.wrapAll('<html></html>')('html')
        super(HTML, self).__init__(element=element, html=html, url=url, default_encoding=default_encoding)
        from requests_html import HTMLSession, AsyncHTMLSession  # local import to avoid circular dependency
        self.session: Union[HTMLSession, AsyncHTMLSession] = session or (AsyncHTMLSession() if async_ else HTMLSession())
        self.page: Optional[Any] = None
        self.next_symbol: List[str] = DEFAULT_NEXT_SYMBOL

    def __repr__(self) -> str:
        return f'<HTML url={self.url!r}>'

    def next(self, fetch: bool = False, next_symbol: Optional[List[str]] = None) -> Union[str, Any, None]:
        if next_symbol is None:
            next_symbol = DEFAULT_NEXT_SYMBOL

        def get_next() -> Optional[str]:
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

    def __iter__(self) -> Any:
        next_page: Any = self
        while True:
            yield next_page
            try:
                next_page = next_page.next(fetch=True, next_symbol=self.next_symbol).html
            except AttributeError:
                break

    def __next__(self) -> Any:
        return self.next(fetch=True, next_symbol=self.next_symbol).html

    def __aiter__(self) -> Any:
        return self

    async def __anext__(self) -> Any:
        while True:
            url: Optional[str] = self.next(fetch=False, next_symbol=self.next_symbol)
            if not url:
                break
            response: Any = await self.session.get(url)
            return response.html

    def add_next_symbol(self, next_symbol: str) -> None:
        self.next_symbol.append(next_symbol)

    async def _async_render(self, *, url: str, script: Optional[str], scrolldown: Union[bool, int], sleep: float, wait: float, reload: bool, content: str, timeout: float, keep_page: bool, cookies: List[Dict[str, Any]]) -> Optional[Tuple[str, Optional[Any], Any]]:
        try:
            page: Any = await self.browser.newPage()
            await asyncio.sleep(wait)
            if cookies:
                for cookie in cookies:
                    if cookie:
                        await page.setCookie(cookie)
            if reload:
                await page.goto(url, options={'timeout': int(timeout * 1000)})
            else:
                await page.goto(f'data:text/html,{self.html}', options={'timeout': int(timeout * 1000)})
            result: Optional[Any] = None
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
            content_str: str = await page.content()
            if not keep_page:
                await page.close()
                page = None
            return (content_str, result, page)
        except TimeoutError:
            await page.close()
            page = None
            return None

    def _convert_cookiejar_to_render(self, session_cookiejar: http.cookiejar.Cookie) -> Dict[str, Any]:
        cookie_render: Dict[str, Any] = {}

        def __convert(cookiejar: Any, key: str) -> Dict[str, Any]:
            try:
                v = eval('cookiejar.' + key)
                kv: Dict[str, Any] = {key: v} if v else {key: ''}
            except Exception:
                kv = {}
            return kv
        keys: List[str] = ['name', 'value', 'url', 'domain', 'path', 'sameSite', 'expires', 'httpOnly', 'secure']
        for key in keys:
            cookie_render.update(__convert(session_cookiejar, key))
        return cookie_render

    def _convert_cookiesjar_to_render(self) -> List[Dict[str, Any]]:
        cookies_render: List[Dict[str, Any]] = []
        if isinstance(self.session.cookies, http.cookiejar.CookieJar):
            for cookie in self.session.cookies:
                cookies_render.append(self._convert_cookiejar_to_render(cookie))
        return cookies_render

    def render(self, retries: int = 8, script: Optional[str] = None, wait: float = 0.2, scrolldown: Union[bool, int] = False, sleep: float = 0, reload: bool = True, timeout: float = 8.0, keep_page: bool = False, cookies: List[Dict[str, Any]] = [{}], send_cookies_session: bool = False) -> Any:
        self.browser = self.session.browser
        content: Optional[str] = None
        if self.url == DEFAULT_URL:
            reload = False
        if send_cookies_session:
            cookies = self._convert_cookiesjar_to_render()
        for i in range(retries):
            if not content:
                try:
                    ret = self.session.loop.run_until_complete(
                        self._async_render(url=self.url, script=script, sleep=sleep, wait=wait, content=self.html, reload=reload, scrolldown=scrolldown, timeout=timeout, keep_page=keep_page, cookies=cookies)
                    )
                    if ret:
                        content, result, page = ret
                except TypeError:
                    pass
            else:
                break
        if not content:
            raise MaxRetries('Unable to render the page. Try increasing timeout')
        html_obj: HTML = HTML(url=self.url, html=content.encode(DEFAULT_ENCODING), default_encoding=DEFAULT_ENCODING)
        self.__dict__.update(html_obj.__dict__)
        self.page = page
        return result

    async def arender(self, retries: int = 8, script: Optional[str] = None, wait: float = 0.2, scrolldown: Union[bool, int] = False, sleep: float = 0, reload: bool = True, timeout: float = 8.0, keep_page: bool = False, cookies: List[Dict[str, Any]] = [{}], send_cookies_session: bool = False) -> Any:
        self.browser = await self.session.browser
        content: Optional[str] = None
        if self.url == DEFAULT_URL:
            reload = False
        if send_cookies_session:
            cookies = self._convert_cookiesjar_to_render()
        page: Any = None
        result: Optional[Any] = None
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
            raise MaxRetries('Unable to render the page. Try increasing timeout')
        html_obj: HTML = HTML(url=self.url, html=content.encode(DEFAULT_ENCODING), default_encoding=DEFAULT_ENCODING)
        self.__dict__.update(html_obj.__dict__)
        self.page = page
        return result

class HTMLResponse(requests.Response):
    def __init__(self, session: Union[HTMLSession, AsyncHTMLSession]) -> None:
        super(HTMLResponse, self).__init__()
        self._html: Optional[HTML] = None
        self.session: Union[HTMLSession, AsyncHTMLSession] = session

    @property
    def html(self) -> HTML:
        if not self._html:
            self._html = HTML(session=self.session, url=self.url, html=self.content, default_encoding=self.encoding)
        return self._html

    @classmethod
    def _from_response(cls, response: requests.Response, session: Union[HTMLSession, AsyncHTMLSession]) -> HTMLResponse:
        html_r: HTMLResponse = cls(session=session)
        html_r.__dict__.update(response.__dict__)
        return html_r

def user_agent(style: Optional[str] = None) -> str:
    global useragent
    if not useragent and style:
        useragent = UserAgent()
    return useragent[style] if style else DEFAULT_USER_AGENT

def _get_first_or_list(l: List[T], first: bool = False) -> Union[T, List[T], None]:
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

    def response_hook(self, response: requests.Response, **kwargs: Any) -> Any:
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
            self._browser = self.loop.run_until_complete(super().browser)
        return self._browser

    def close(self) -> None:
        if hasattr(self, '_browser'):
            self.loop.run_until_complete(self._browser.close())
        super().close()

class AsyncHTMLSession(BaseSession):
    def __init__(self, loop: Optional[asyncio.AbstractEventLoop] = None, workers: Optional[int] = None, mock_browser: bool = True, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.loop: asyncio.AbstractEventLoop = loop or asyncio.get_event_loop()
        self.thread_pool: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=workers)

    def request(self, *args: Any, **kwargs: Any) -> Any:
        func: Callable = partial(super().request, *args, **kwargs)
        return self.loop.run_in_executor(self.thread_pool, func)

    async def close(self) -> None:
        if hasattr(self, '_browser'):
            await self._browser.close()
        super().close()

    def run(self, *coros: Callable[[], Any]) -> List[Any]:
        tasks = [asyncio.ensure_future(coro()) for coro in coros]
        done, _ = self.loop.run_until_complete(asyncio.wait(tasks))
        return [t.result() for t in done]

############################################
# File: setup.py
############################################
import io
import os
import sys
from shutil import rmtree
from setuptools import setup, Command

NAME: str = 'requests-html'
DESCRIPTION: str = 'HTML Parsing for Humans.'
URL: str = 'https://github.com/psf/requests-html'
EMAIL: str = 'me@kennethreitz.org'
AUTHOR: str = 'Kenneth Reitz'
VERSION: str = '0.10.0'
REQUIRED: List[str] = ['requests', 'pyquery', 'fake-useragent', 'parse', 'beautifulsoup4', 'w3lib', 'pyppeteer>=0.0.14']

here: str = os.path.abspath(os.path.dirname(__file__))
with io.open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description: str = '\n' + f.read()

class UploadCommand(Command):
    description: str = 'Build and publish the package.'
    user_options: List[Any] = []

    @staticmethod
    def status(s: str) -> None:
        print('\x1b[1m{0}\x1b[0m'.format(s))

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass
        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))
        self.status('Uploading the package to PyPi via Twine…')
        os.system('twine upload dist/*')
        self.status('Publishing git tags…')
        os.system('git tag v{0}'.format(VERSION))
        os.system('git push --tags')
        sys.exit()

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    python_requires='>=3.6.0',
    py_modules=['requests_html'],
    install_requires=REQUIRED,
    include_package_data=True,
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
    cmdclass={'upload': UploadCommand}
)

############################################
# File: tests/test_internet.py
############################################
from typing import Any
import pytest
from requests_html import HTMLSession, AsyncHTMLSession, HTMLResponse

urls: list[str] = ['https://xkcd.com/1957/',
                   'https://www.reddit.com/',
                   'https://github.com/psf/requests-html/issues',
                   'https://discord.com/category/engineering',
                   'https://stackoverflow.com/',
                   'https://www.frontiersin.org/',
                   'https://azure.microsoft.com/en-us']

@pytest.mark.parametrize('url', urls)
@pytest.mark.internet
def test_pagination(url: str) -> None:
    session = HTMLSession()
    r: Any = session.get(url)
    assert next(r.html)

@pytest.mark.parametrize('url', urls)
@pytest.mark.internet
@pytest.mark.asyncio
async def test_async_pagination(event_loop: Any, url: str) -> None:
    asession = AsyncHTMLSession()
    r: Any = await asession.get(url)
    assert await r.html.__anext__()

@pytest.mark.internet
def test_async_run() -> None:
    asession = AsyncHTMLSession()
    async_list: list[Any] = []
    for url in urls:
        async def _test(url=url) -> Any:
            return await asession.get(url)
        async_list.append(_test)
    r = asession.run(*async_list)
    assert len(r) == len(urls)
    assert isinstance(r[0], HTMLResponse)

############################################
# File: tests/test_requests_html.py
############################################
import os
from functools import partial
import pytest
from pyppeteer.browser import Browser
from pyppeteer.page import Page
from requests_html import HTMLSession, AsyncHTMLSession, HTML
from requests_file import FileAdapter

session = HTMLSession()
session.mount('file://', FileAdapter())

def get() -> Any:
    path: str = os.path.sep.join((os.path.dirname(os.path.abspath(__file__)), 'python.html'))
    url: str = f'file://{path}'
    return session.get(url)

@pytest.fixture
def async_get(event_loop: Any) -> Any:
    from requests_html import AsyncHTMLSession
    async_session = AsyncHTMLSession()
    async_session.mount('file://', FileAdapter())
    path: str = os.path.sep.join((os.path.dirname(os.path.abspath(__file__)), 'python.html'))
    url: str = f'file://{path}'
    return partial(async_session.get, url)

def test_file_get() -> None:
    r = get()
    assert r.status_code == 200

@pytest.mark.asyncio
async def test_async_file_get(async_get: Any) -> None:
    r = await async_get()
    assert r.status_code == 200

def test_class_seperation() -> None:
    r = get()
    about = r.html.find('#about', first=True)
    assert len(about.attrs['class']) == 2

def test_css_selector() -> None:
    r = get()
    about = r.html.find('#about', first=True)
    for menu_item in ('About', 'Applications', 'Quotes', 'Getting Started', 'Help', 'Python Brochure'):
        assert menu_item in about.text.split('\n')
        assert menu_item in about.full_text.split('\n')

def test_containing() -> None:
    r = get()
    python_elements = r.html.find(containing='python')
    assert len(python_elements) == 192
    for e in python_elements:
        assert 'python' in e.full_text.lower()

def test_attrs() -> None:
    r = get()
    about = r.html.find('#about', first=True)
    assert 'aria-haspopup' in about.attrs
    assert len(about.attrs['class']) == 2

def test_links() -> None:
    r = get()
    about = r.html.find('#about', first=True)
    assert len(about.links) == 6
    assert len(about.absolute_links) == 6

@pytest.mark.asyncio
async def test_async_links(async_get: Any) -> None:
    r = await async_get()
    about = r.html.find('#about', first=True)
    assert len(about.links) == 6
    assert len(about.absolute_links) == 6

def test_search() -> None:
    r = get()
    style = r.html.search('Python is a {} language')[0]
    assert style == 'programming'

def test_xpath() -> None:
    r = get()
    html_elem = r.html.xpath('/html', first=True)
    assert 'no-js' in html_elem.attrs['class']
    a_hrefs = r.html.xpath('//a/@href')
    assert '#site-map' in a_hrefs

def test_html_loading() -> None:
    doc: str = "<a href='https://httpbin.org'>"
    html_obj = HTML(html=doc)
    assert 'https://httpbin.org' in html_obj.links
    assert isinstance(html_obj.raw_html, bytes)
    assert isinstance(html_obj.html, str)

def test_anchor_links() -> None:
    r = get()
    r.html.skip_anchors = False
    assert '#site-map' in r.html.links

@pytest.mark.parametrize('url,link,expected', [
    ('http://example.com/', 'test.html', 'http://example.com/test.html'),
    ('http://example.com', 'test.html', 'http://example.com/test.html'),
    ('http://example.com/foo/', 'test.html', 'http://example.com/foo/test.html'),
    ('http://example.com/foo/bar', 'test.html', 'http://example.com/foo/test.html'),
    ('http://example.com/foo/', '/test.html', 'http://example.com/test.html'),
    ('http://example.com/', 'http://xkcd.com/about/', 'http://xkcd.com/about/'),
    ('http://example.com/', '//xkcd.com/about/', 'http://xkcd.com/about/')
])
def test_absolute_links(url: str, link: str, expected: str) -> None:
    head_template: str = "<head><base href='{}'></head>"
    body_template: str = "<body><a href='{}'>Next</a></body>"
    html_obj = HTML(html=body_template.format(link), url=url)
    assert html_obj.absolute_links.pop() == expected
    html_obj = HTML(html=head_template.format(url) + body_template.format(link), url='http://example.com/foobar/')
    assert html_obj.absolute_links.pop() == expected

def test_parser() -> None:
    doc: str = "<a href='https://httpbin.org'>httpbin.org\n</a>"
    html_obj = HTML(html=doc)
    assert html_obj.find('html')
    # Assuming element() and text() methods exist for compatibility.
    assert html_obj.element('a').text().strip() == 'httpbin.org'  # type: ignore

@pytest.mark.render
def test_render() -> None:
    r = get()
    script: str = '''
    () => {
        return {
            width: document.documentElement.clientWidth,
            height: document.documentElement.clientHeight,
            deviceScaleFactor: window.devicePixelRatio,
        }
    }
    '''
    val = r.html.render(script=script)
    for value in ('width', 'height', 'deviceScaleFactor'):
        assert value in val
    about = r.html.find('#about', first=True)
    assert len(about.links) == 6

@pytest.mark.render
@pytest.mark.asyncio
async def test_async_render(async_get: Any) -> None:
    r = await async_get()
    script: str = '''
    () => {
        return {
            width: document.documentElement.clientWidth,
            height: document.documentElement.clientHeight,
            deviceScaleFactor: window.devicePixelRatio,
        }
    }
    '''
    val = await r.html.arender(script=script)
    for value in ('width', 'height', 'deviceScaleFactor'):
        assert value in val
    about = r.html.find('#about', first=True)
    assert len(about.links) == 6
    await r.html.browser.close()

@pytest.mark.render
def test_bare_render() -> None:
    doc: str = "<a href='https://httpbin.org'>"
    html_obj = HTML(html=doc)
    script: str = '''
        () => {
            return {
                width: document.documentElement.clientWidth,
                height: document.documentElement.clientHeight,
                deviceScaleFactor: window.devicePixelRatio,
            }
        }
    '''
    val = html_obj.render(script=script, reload=False)
    for value in ('width', 'height', 'deviceScaleFactor'):
        assert value in val
    assert html_obj.find('html')
    assert 'https://httpbin.org' in html_obj.links

@pytest.mark.render
@pytest.mark.asyncio
async def test_bare_arender() -> None:
    doc: str = "<a href='https://httpbin.org'>"
    html_obj = HTML(html=doc, async_=True)
    script: str = '''
        () => {
            return {
                width: document.documentElement.clientWidth,
                height: document.documentElement.clientHeight,
                deviceScaleFactor: window.devicePixelRatio,
            }
        }
    '''
    val = await html_obj.arender(script=script, reload=False)
    for value in ('width', 'height', 'deviceScaleFactor'):
        assert value in val
    assert html_obj.find('html')
    assert 'https://httpbin.org' in html_obj.links
    await html_obj.browser.close()

def test_bare_js_eval() -> None:
    doc: str = '''
    <!DOCTYPE html>
    <html>
    <body>
    <div id="replace">This gets replaced</div>

    <script type="text/javascript">
      document.getElementById("replace").innerHTML = "yolo";
    </script>
    </body>
    </html>
    '''
    html_obj = HTML(html=doc)
    html_obj.render()
    assert html_obj.find('#replace', first=True).text == 'yolo'

@pytest.mark.render
@pytest.mark.asyncio
async def test_bare_js_async_eval() -> None:
    doc: str = '''
    <!DOCTYPE html>
    <html>
    <body>
    <div id="replace">This gets replaced</div>

    <script type="text/javascript">
      document.getElementById("replace").innerHTML = "yolo";
    </script>
    </body>
    </html>
    '''
    html_obj = HTML(html=doc, async_=True)
    await html_obj.arender()
    assert html_obj.find('#replace', first=True).text == 'yolo'
    await html_obj.browser.close()

def test_browser_session() -> None:
    session_obj = HTMLSession()
    assert isinstance(session_obj.browser, Browser)
    assert hasattr(session_obj, 'loop')
    session_obj.close()

def test_browser_process() -> None:
    for _ in range(3):
        r = get()
        r.html.render()
        assert r.html.page is None

@pytest.mark.asyncio
async def test_browser_session_fail() -> None:
    session_obj = HTMLSession()
    with pytest.raises(RuntimeError):
        _ = session_obj.browser

@pytest.mark.asyncio
async def test_async_browser_session() -> None:
    session_obj = AsyncHTMLSession()
    browser_obj = await session_obj.browser
    assert isinstance(browser_obj, Browser)
    await session_obj.close()