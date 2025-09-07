# Merged Python files from requests-html
# Total files: 5


# File: docs\source\conf.py

import requests_html
project = 'requests-HTML'
copyright = u'MMXVIII. A <a href="http://kennethreitz.com/pages/open-projects.html">Kenneth Reitz</a> Project'
author = 'Kenneth Reitz'
version = ''
release = 'v0.3.4'
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.doctest', 'sphinx.ext.intersphinx', 'sphinx.ext.todo', 'sphinx.ext.coverage', 'sphinx.ext.viewcode', 'sphinx.ext.githubpages']
templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
language = None
exclude_patterns = []
pygments_style = 'sphinx'
html_theme = 'alabaster'
html_theme_options = {'show_powered_by': False, 'github_user': 'psf', 'github_repo': 'requests-html', 'github_banner': True, 'show_related': False, 'note_bg': '#FFF59C'}
html_static_path = ['_static']
html_sidebars = {'index': ['sidebarintro.html', 'sourcelink.html', 'searchbox.html', 'hacks.html'], '**': ['sidebarlogo.html', 'localtoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html', 'hacks.html']}
html_show_sphinx = False
html_show_sourcelink = False
htmlhelp_basename = 'requests-htmldoc'
latex_elements = {}
latex_documents = [(master_doc, 'requests-html.tex', 'requests-html Documentation', 'Kenneth Reitz', 'manual')]
man_pages = [(master_doc, 'requests-html', 'requests-html Documentation', [author], 1)]
texinfo_documents = [(master_doc, 'requests-html', 'requests-html Documentation', author, 'requests-html', 'One line description of project.', 'Miscellaneous')]
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright
epub_exclude_files = ['search.html']
intersphinx_mapping = {'https://docs.python.org/': None}
todo_include_todos = True

# File: requests_html.py

import sys
import asyncio
from urllib.parse import urlparse, urlunparse, urljoin
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures._base import TimeoutError
from functools import partial
from typing import Set, Union, List, MutableMapping, Optional
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
DEFAULT_NEXT_SYMBOL = ['next', 'more', 'older']
cleaner = Cleaner()
cleaner.javascript = True
cleaner.style = True
useragent = None
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

    def __init__(self, message):
        self.message = message

class BaseParser:
    """A basic HTML/Element Parser, for Humans.

    :param element: The element from which to base the parsing upon.
    :param default_encoding: Which encoding to default to.
    :param html: HTML from which to base the parsing upon (optional).
    :param url: The URL from which the HTML originated, used for ``absolute_links``.

    """

    def __init__(self, *, element, default_encoding=None, html=None, url):
        self.element = element
        self.url = url
        self.skip_anchors = True
        self.default_encoding = default_encoding
        self._encoding = None
        self._html = html.encode(DEFAULT_ENCODING) if isinstance(html, str) else html
        self._lxml = None
        self._pq = None

    @property
    def raw_html(self):
        """Bytes representation of the HTML content.
        (`learn more <http://www.diveintopython3.net/strings.html>`_).
        """
        if self._html:
            return self._html
        else:
            return etree.tostring(self.element, encoding='unicode').strip().encode(self.encoding)

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
            self._encoding = html_to_unicode(self.default_encoding, self._html)[0]
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

    def find(self, selector='*', *, containing=None, clean=False, first=False, _encoding=None):
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
        elements = [Element(element=found, url=self.url, default_encoding=encoding) for found in self.pq(selector)]
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

    def xpath(self, selector, *, clean=False, first=False, _encoding=None):
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
        elements = [Element(element=selection, url=self.url, default_encoding=_encoding or self.encoding) if not isinstance(selection, etree._ElementUnicodeResult) else str(selection) for selection in selected]
        if clean:
            elements_copy = elements.copy()
            elements = []
            for element in elements_copy:
                element.raw_html = lxml_html_tostring(cleaner.clean_html(element.lxml))
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
                    href = link.attrs['href'].strip()
                    if href and (not (href.startswith('#') and self.skip_anchors)) and (not href.startswith(('javascript:', 'mailto:'))):
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
            parsed = (v for v in parsed.values())
            return urlunparse(parsed)
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
        parsed = (v for v in parsed.values())
        url = urlunparse(parsed)
        return url

class Element(BaseParser):
    """An element of HTML.

    :param element: The element from which to base the parsing upon.
    :param url: The URL from which the HTML originated, used for ``absolute_links``.
    :param default_encoding: Which encoding to default to.
    """
    __slots__ = ['element', 'url', 'skip_anchors', 'default_encoding', '_encoding', '_html', '_lxml', '_pq', '_attrs', 'session']

    def __init__(self, *, element, url, default_encoding=None):
        super(Element, self).__init__(element=element, url=url, default_encoding=default_encoding)
        self.element = element
        self.tag = element.tag
        self.lineno = element.sourceline
        self._attrs = None

    def __repr__(self):
        attrs = ['{}={}'.format(attr, repr(self.attrs[attr])) for attr in self.attrs]
        return '<Element {} {}>'.format(repr(self.element.tag), ' '.join(attrs))

    @property
    def attrs(self):
        """Returns a dictionary of the attributes of the :class:`Element <Element>`
        (`learn more <https://www.w3schools.com/tags/ref_attributes.asp>`_).
        """
        if self._attrs is None:
            self._attrs = {k: v for (k, v) in self.element.items()}
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

    def __init__(self, *, session=None, url=DEFAULT_URL, html, default_encoding=DEFAULT_ENCODING, async_=False):
        if isinstance(html, str):
            html = html.encode(DEFAULT_ENCODING)
        pq = PyQuery(html)
        super(HTML, self).__init__(element=pq('html') or pq.wrapAll('<html></html>')('html'), html=html, url=url, default_encoding=default_encoding)
        self.session = session or (async_ and AsyncHTMLSession()) or HTMLSession()
        self.page = None
        self.next_symbol = DEFAULT_NEXT_SYMBOL

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

    def __iter__(self):
        next = self
        while True:
            yield next
            try:
                next = next.next(fetch=True, next_symbol=self.next_symbol).html
            except AttributeError:
                break

    def __next__(self):
        return self.next(fetch=True, next_symbol=self.next_symbol).html

    def __aiter__(self):
        return self

    async def __anext__(self):
        while True:
            url = self.next(fetch=False, next_symbol=self.next_symbol)
            if not url:
                break
            response = await self.session.get(url)
            return response.html

    def add_next_symbol(self, next_symbol):
        self.next_symbol.append(next_symbol)

    async def _async_render(self, *, url, script=None, scrolldown, sleep, wait, reload, content, timeout, keep_page, cookies=[{}]):
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

    def _convert_cookiejar_to_render(self, session_cookiejar):
        """
        Convert HTMLSession.cookies:cookiejar[] for browser.newPage().setCookie
        """
        cookie_render = {}

        def __convert(cookiejar, key):
            try:
                v = eval('cookiejar.' + key)
                if not v:
                    kv = ''
                else:
                    kv = {key: v}
            except:
                kv = ''
            return kv
        keys = ['name', 'value', 'url', 'domain', 'path', 'sameSite', 'expires', 'httpOnly', 'secure']
        for key in keys:
            cookie_render.update(__convert(session_cookiejar, key))
        return cookie_render

    def _convert_cookiesjar_to_render(self):
        """
        Convert HTMLSession.cookies for browser.newPage().setCookie
        Return a list of dict
        """
        cookies_render = []
        if isinstance(self.session.cookies, http.cookiejar.CookieJar):
            for cookie in self.session.cookies:
                cookies_render.append(self._convert_cookiejar_to_render(cookie))
        return cookies_render

    def render(self, retries=8, script=None, wait=0.2, scrolldown=False, sleep=0, reload=True, timeout=8.0, keep_page=False, cookies=[{}], send_cookies_session=False):
        '''Reloads the response in Chromium, and replaces HTML content
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

            script = """
                () => {
                    return {
                        width: document.documentElement.clientWidth,
                        height: document.documentElement.clientHeight,
                        deviceScaleFactor: window.devicePixelRatio,
                    }
                }
            """

        Returns the return value of the executed  ``script``, if any is provided:

        .. code-block:: python

            >>> r.html.render(script=script)
            {'width': 800, 'height': 600, 'deviceScaleFactor': 1}

        Warning: the first time you run this method, it will download
        Chromium into your home directory (``~/.pyppeteer``).
        '''
        self.browser = self.session.browser
        content = None
        if self.url == DEFAULT_URL:
            reload = False
        if send_cookies_session:
            cookies = self._convert_cookiesjar_to_render()
        for i in range(retries):
            if not content:
                try:
                    (content, result, page) = self.session.loop.run_until_complete(self._async_render(url=self.url, script=script, sleep=sleep, wait=wait, content=self.html, reload=reload, scrolldown=scrolldown, timeout=timeout, keep_page=keep_page, cookies=cookies))
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

    async def arender(self, retries=8, script=None, wait=0.2, scrolldown=False, sleep=0, reload=True, timeout=8.0, keep_page=False, cookies=[{}], send_cookies_session=False):
        """ Async version of render. Takes same parameters. """
        self.browser = await self.session.browser
        content = None
        if self.url == DEFAULT_URL:
            reload = False
        if send_cookies_session:
            cookies = self._convert_cookiesjar_to_render()
        for _ in range(retries):
            if not content:
                try:
                    (content, result, page) = await self._async_render(url=self.url, script=script, sleep=sleep, wait=wait, content=self.html, reload=reload, scrolldown=scrolldown, timeout=timeout, keep_page=keep_page, cookies=cookies)
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

class HTMLResponse(requests.Response):
    """An HTML-enabled :class:`requests.Response <requests.Response>` object.
    Effectively the same, but with an intelligent ``.html`` property added.
    """

    def __init__(self, session):
        super(HTMLResponse, self).__init__()
        self._html = None
        self.session = session

    @property
    def html(self):
        if not self._html:
            self._html = HTML(session=self.session, url=self.url, html=self.content, default_encoding=self.encoding)
        return self._html

    @classmethod
    def _from_response(cls, response, session):
        html_r = cls(session=session)
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

    def __init__(self, mock_browser=True, verify=True, browser_args=['--no-sandbox']):
        super().__init__()
        if mock_browser:
            self.headers['User-Agent'] = user_agent()
        self.hooks['response'].append(self.response_hook)
        self.verify = verify
        self.__browser_args = browser_args

    def response_hook(self, response, **kwargs):
        """ Change response encoding and replace it by a HTMLResponse. """
        if not response.encoding:
            response.encoding = DEFAULT_ENCODING
        return HTMLResponse._from_response(response, self)

    @property
    async def browser(self):
        if not hasattr(self, '_browser'):
            self._browser = await pyppeteer.launch(ignoreHTTPSErrors=not self.verify, headless=True, args=self.__browser_args)
        return self._browser

class HTMLSession(BaseSession):

    def __init__(self, **kwargs):
        super(HTMLSession, self).__init__(**kwargs)

    @property
    def browser(self):
        if not hasattr(self, '_browser'):
            self.loop = asyncio.get_event_loop()
            if self.loop.is_running():
                raise RuntimeError('Cannot use HTMLSession within an existing event loop. Use AsyncHTMLSession instead.')
            self._browser = self.loop.run_until_complete(super().browser)
        return self._browser

    def close(self):
        """ If a browser was created close it first. """
        if hasattr(self, '_browser'):
            self.loop.run_until_complete(self._browser.close())
        super().close()

class AsyncHTMLSession(BaseSession):
    """ An async consumable session. """

    def __init__(self, loop=None, workers=None, mock_browser=True, *args, **kwargs):
        """ Set or create an event loop and a thread pool.

            :param loop: Asyncio loop to use.
            :param workers: Amount of threads to use for executing async calls.
                If not pass it will default to the number of processors on the
                machine, multiplied by 5. """
        super().__init__(*args, **kwargs)
        self.loop = loop or asyncio.get_event_loop()
        self.thread_pool = ThreadPoolExecutor(max_workers=workers)

    def request(self, *args, **kwargs):
        """ Partial original request func and run it in a thread. """
        func = partial(super().request, *args, **kwargs)
        return self.loop.run_in_executor(self.thread_pool, func)

    async def close(self):
        """ If a browser was created close it first. """
        if hasattr(self, '_browser'):
            await self._browser.close()
        super().close()

    def run(self, *coros):
        """ Pass in all the coroutines you want to run, it will wrap each one
            in a task, run it and wait for the result. Return a list with all
            results, this is returned in the same order coros are passed in. """
        tasks = [asyncio.ensure_future(coro()) for coro in coros]
        (done, _) = self.loop.run_until_complete(asyncio.wait(tasks))
        return [t.result() for t in done]

# File: setup.py

import io
import os
import sys
from shutil import rmtree
from setuptools import setup, Command
NAME = 'requests-html'
DESCRIPTION = 'HTML Parsing for Humans.'
URL = 'https://github.com/psf/requests-html'
EMAIL = 'me@kennethreitz.org'
AUTHOR = 'Kenneth Reitz'
VERSION = '0.10.0'
REQUIRED = ['requests', 'pyquery', 'fake-useragent', 'parse', 'beautifulsoup4', 'w3lib', 'pyppeteer>=0.0.14']
here = os.path.abspath(os.path.dirname(__file__))
with io.open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = '\n' + f.read()

class UploadCommand(Command):
    """Support setup.py upload."""
    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\x1b[1m{0}\x1b[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
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
setup(name=NAME, version=VERSION, description=DESCRIPTION, long_description=long_description, author=AUTHOR, author_email=EMAIL, url=URL, python_requires='>=3.6.0', py_modules=['requests_html'], install_requires=REQUIRED, include_package_data=True, license='MIT', classifiers=['License :: OSI Approved :: MIT License', 'Programming Language :: Python', 'Programming Language :: Python :: 3.6', 'Programming Language :: Python :: Implementation :: CPython', 'Programming Language :: Python :: Implementation :: PyPy'], cmdclass={'upload': UploadCommand})

# File: tests\test_internet.py

import pytest
from requests_html import HTMLSession, AsyncHTMLSession, HTMLResponse
urls = ['https://xkcd.com/1957/', 'https://www.reddit.com/', 'https://github.com/psf/requests-html/issues', 'https://discord.com/category/engineering', 'https://stackoverflow.com/', 'https://www.frontiersin.org/', 'https://azure.microsoft.com/en-us']

@pytest.mark.parametrize('url', urls)
@pytest.mark.internet
def test_pagination(url):
    session = HTMLSession()
    r = session.get(url)
    assert next(r.html)

@pytest.mark.parametrize('url', urls)
@pytest.mark.internet
@pytest.mark.asyncio
async def test_async_pagination(event_loop, url):
    asession = AsyncHTMLSession()
    r = await asession.get(url)
    assert await r.html.__anext__()

@pytest.mark.internet
def test_async_run():
    asession = AsyncHTMLSession()
    async_list = []
    for url in urls:

        async def _test():
            return await asession.get(url)
        async_list.append(_test)
    r = asession.run(*async_list)
    assert len(r) == len(urls)
    assert isinstance(r[0], HTMLResponse)

# File: tests\test_requests_html.py

import os
from functools import partial
import pytest
from pyppeteer.browser import Browser
from pyppeteer.page import Page
from requests_html import HTMLSession, AsyncHTMLSession, HTML
from requests_file import FileAdapter
session = HTMLSession()
session.mount('file://', FileAdapter())

def get():
    path = os.path.sep.join((os.path.dirname(os.path.abspath(__file__)), 'python.html'))
    url = f'file://{path}'
    return session.get(url)

@pytest.fixture
def async_get(event_loop):
    """AsyncSession cannot be created global since it will create
        a different loop from pytest-asyncio. """
    async_session = AsyncHTMLSession()
    async_session.mount('file://', FileAdapter())
    path = os.path.sep.join((os.path.dirname(os.path.abspath(__file__)), 'python.html'))
    url = 'file://{}'.format(path)
    return partial(async_session.get, url)

def test_file_get():
    r = get()
    assert r.status_code == 200

@pytest.mark.asyncio
async def test_async_file_get(async_get):
    r = await async_get()
    assert r.status_code == 200

def test_class_seperation():
    r = get()
    about = r.html.find('#about', first=True)
    assert len(about.attrs['class']) == 2

def test_css_selector():
    r = get()
    about = r.html.find('#about', first=True)
    for menu_item in ('About', 'Applications', 'Quotes', 'Getting Started', 'Help', 'Python Brochure'):
        assert menu_item in about.text.split('\n')
        assert menu_item in about.full_text.split('\n')

def test_containing():
    r = get()
    python = r.html.find(containing='python')
    assert len(python) == 192
    for e in python:
        assert 'python' in e.full_text.lower()

def test_attrs():
    r = get()
    about = r.html.find('#about', first=True)
    assert 'aria-haspopup' in about.attrs
    assert len(about.attrs['class']) == 2

def test_links():
    r = get()
    about = r.html.find('#about', first=True)
    assert len(about.links) == 6
    assert len(about.absolute_links) == 6

@pytest.mark.asyncio
async def test_async_links(async_get):
    r = await async_get()
    about = r.html.find('#about', first=True)
    assert len(about.links) == 6
    assert len(about.absolute_links) == 6

def test_search():
    r = get()
    style = r.html.search('Python is a {} language')[0]
    assert style == 'programming'

def test_xpath():
    r = get()
    html = r.html.xpath('/html', first=True)
    assert 'no-js' in html.attrs['class']
    a_hrefs = r.html.xpath('//a/@href')
    assert '#site-map' in a_hrefs

def test_html_loading():
    doc = "<a href='https://httpbin.org'>"
    html = HTML(html=doc)
    assert 'https://httpbin.org' in html.links
    assert isinstance(html.raw_html, bytes)
    assert isinstance(html.html, str)

def test_anchor_links():
    r = get()
    r.html.skip_anchors = False
    assert '#site-map' in r.html.links

@pytest.mark.parametrize('url,link,expected', [('http://example.com/', 'test.html', 'http://example.com/test.html'), ('http://example.com', 'test.html', 'http://example.com/test.html'), ('http://example.com/foo/', 'test.html', 'http://example.com/foo/test.html'), ('http://example.com/foo/bar', 'test.html', 'http://example.com/foo/test.html'), ('http://example.com/foo/', '/test.html', 'http://example.com/test.html'), ('http://example.com/', 'http://xkcd.com/about/', 'http://xkcd.com/about/'), ('http://example.com/', '//xkcd.com/about/', 'http://xkcd.com/about/')])
def test_absolute_links(url, link, expected):
    head_template = "<head><base href='{}'></head>"
    body_template = "<body><a href='{}'>Next</a></body>"
    html = HTML(html=body_template.format(link), url=url)
    assert html.absolute_links.pop() == expected
    html = HTML(html=head_template.format(url) + body_template.format(link), url='http://example.com/foobar/')
    assert html.absolute_links.pop() == expected

def test_parser():
    doc = "<a href='https://httpbin.org'>httpbin.org\n</a>"
    html = HTML(html=doc)
    assert html.find('html')
    assert html.element('a').text().strip() == 'httpbin.org'

@pytest.mark.render
def test_render():
    r = get()
    script = '\n    () => {\n        return {\n            width: document.documentElement.clientWidth,\n            height: document.documentElement.clientHeight,\n            deviceScaleFactor: window.devicePixelRatio,\n        }\n    }\n    '
    val = r.html.render(script=script)
    for value in ('width', 'height', 'deviceScaleFactor'):
        assert value in val
    about = r.html.find('#about', first=True)
    assert len(about.links) == 6

@pytest.mark.render
@pytest.mark.asyncio
async def test_async_render(async_get):
    r = await async_get()
    script = '\n    () => {\n        return {\n            width: document.documentElement.clientWidth,\n            height: document.documentElement.clientHeight,\n            deviceScaleFactor: window.devicePixelRatio,\n        }\n    }\n    '
    val = await r.html.arender(script=script)
    for value in ('width', 'height', 'deviceScaleFactor'):
        assert value in val
    about = r.html.find('#about', first=True)
    assert len(about.links) == 6
    await r.html.browser.close()

@pytest.mark.render
def test_bare_render():
    doc = "<a href='https://httpbin.org'>"
    html = HTML(html=doc)
    script = '\n        () => {\n            return {\n                width: document.documentElement.clientWidth,\n                height: document.documentElement.clientHeight,\n                deviceScaleFactor: window.devicePixelRatio,\n            }\n        }\n    '
    val = html.render(script=script, reload=False)
    for value in ('width', 'height', 'deviceScaleFactor'):
        assert value in val
    assert html.find('html')
    assert 'https://httpbin.org' in html.links

@pytest.mark.render
@pytest.mark.asyncio
async def test_bare_arender():
    doc = "<a href='https://httpbin.org'>"
    html = HTML(html=doc, async_=True)
    script = '\n        () => {\n            return {\n                width: document.documentElement.clientWidth,\n                height: document.documentElement.clientHeight,\n                deviceScaleFactor: window.devicePixelRatio,\n            }\n        }\n    '
    val = await html.arender(script=script, reload=False)
    for value in ('width', 'height', 'deviceScaleFactor'):
        assert value in val
    assert html.find('html')
    assert 'https://httpbin.org' in html.links
    await html.browser.close()

@pytest.mark.render
def test_bare_js_eval():
    doc = '\n    <!DOCTYPE html>\n    <html>\n    <body>\n    <div id="replace">This gets replaced</div>\n\n    <script type="text/javascript">\n      document.getElementById("replace").innerHTML = "yolo";\n    </script>\n    </body>\n    </html>\n    '
    html = HTML(html=doc)
    html.render()
    assert html.find('#replace', first=True).text == 'yolo'

@pytest.mark.render
@pytest.mark.asyncio
async def test_bare_js_async_eval():
    doc = '\n    <!DOCTYPE html>\n    <html>\n    <body>\n    <div id="replace">This gets replaced</div>\n\n    <script type="text/javascript">\n      document.getElementById("replace").innerHTML = "yolo";\n    </script>\n    </body>\n    </html>\n    '
    html = HTML(html=doc, async_=True)
    await html.arender()
    assert html.find('#replace', first=True).text == 'yolo'
    await html.browser.close()

def test_browser_session():
    """ Test browser instances is created and properly close when session is closed.
        Note: session.close method need to be tested together with browser creation,
            since not doing that will leave the browser running. """
    session = HTMLSession()
    assert isinstance(session.browser, Browser)
    assert hasattr(session, 'loop')
    session.close()

def test_browser_process():
    for _ in range(3):
        r = get()
        r.html.render()
        assert r.html.page is None

@pytest.mark.asyncio
async def test_browser_session_fail():
    """ HTMLSession.browser should not be call within an existing event loop> """
    session = HTMLSession()
    with pytest.raises(RuntimeError):
        session.browser

@pytest.mark.asyncio
async def test_async_browser_session():
    session = AsyncHTMLSession()
    browser = await session.browser
    assert isinstance(browser, Browser)
    await session.close()