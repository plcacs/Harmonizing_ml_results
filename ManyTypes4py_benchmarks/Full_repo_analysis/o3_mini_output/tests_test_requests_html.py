import os
from functools import partial
from typing import Callable, Awaitable, Any
import pytest
from pyppeteer.browser import Browser
from pyppeteer.page import Page
from requests_html import HTMLSession, AsyncHTMLSession, HTML
from requests_file import FileAdapter
from requests.models import Response

session: HTMLSession = HTMLSession()
session.mount('file://', FileAdapter())

def get() -> Response:
    path: str = os.path.sep.join(
        (os.path.dirname(os.path.abspath(__file__)), 'python.html')
    )
    url: str = f'file://{path}'
    return session.get(url)

@pytest.fixture
def async_get(event_loop: Any) -> Callable[[], Awaitable[Response]]:
    """AsyncSession cannot be created global since it will create
        a different loop from pytest-asyncio. """
    async_session: AsyncHTMLSession = AsyncHTMLSession()
    async_session.mount('file://', FileAdapter())
    path: str = os.path.sep.join(
        (os.path.dirname(os.path.abspath(__file__)), 'python.html')
    )
    url: str = 'file://{}'.format(path)
    return partial(async_session.get, url)

def test_file_get() -> None:
    r: Response = get()
    assert r.status_code == 200

@pytest.mark.asyncio
async def test_async_file_get(async_get: Callable[[], Awaitable[Response]]) -> None:
    r: Response = await async_get()
    assert r.status_code == 200

def test_class_seperation() -> None:
    r: Response = get()
    about: HTML = r.html.find('#about', first=True)
    assert len(about.attrs['class']) == 2

def test_css_selector() -> None:
    r: Response = get()
    about: HTML = r.html.find('#about', first=True)
    for menu_item in ('About', 'Applications', 'Quotes', 'Getting Started', 'Help', 'Python Brochure'):
        assert menu_item in about.text.split('\n')
        assert menu_item in about.full_text.split('\n')

def test_containing() -> None:
    r: Response = get()
    python_elements = r.html.find(containing='python')
    assert len(python_elements) == 192
    for e in python_elements:
        assert 'python' in e.full_text.lower()

def test_attrs() -> None:
    r: Response = get()
    about: HTML = r.html.find('#about', first=True)
    assert 'aria-haspopup' in about.attrs
    assert len(about.attrs['class']) == 2

def test_links() -> None:
    r: Response = get()
    about: HTML = r.html.find('#about', first=True)
    assert len(about.links) == 6
    assert len(about.absolute_links) == 6

@pytest.mark.asyncio
async def test_async_links(async_get: Callable[[], Awaitable[Response]]) -> None:
    r: Response = await async_get()
    about: HTML = r.html.find('#about', first=True)
    assert len(about.links) == 6
    assert len(about.absolute_links) == 6

def test_search() -> None:
    r: Response = get()
    style: str = r.html.search('Python is a {} language')[0]
    assert style == 'programming'

def test_xpath() -> None:
    r: Response = get()
    html_element: HTML = r.html.xpath('/html', first=True)
    assert 'no-js' in html_element.attrs['class']
    a_hrefs = r.html.xpath('//a/@href')
    assert '#site-map' in a_hrefs

def test_html_loading() -> None:
    doc: str = "<a href='https://httpbin.org'>"
    html: HTML = HTML(html=doc)
    assert 'https://httpbin.org' in html.links
    assert isinstance(html.raw_html, bytes)
    assert isinstance(html.html, str)

def test_anchor_links() -> None:
    r: Response = get()
    r.html.skip_anchors = False
    assert '#site-map' in r.html.links

@pytest.mark.parametrize(
    'url,link,expected', [
        ('http://example.com/', 'test.html', 'http://example.com/test.html'),
        ('http://example.com', 'test.html', 'http://example.com/test.html'),
        ('http://example.com/foo/', 'test.html', 'http://example.com/foo/test.html'),
        ('http://example.com/foo/bar', 'test.html', 'http://example.com/foo/test.html'),
        ('http://example.com/foo/', '/test.html', 'http://example.com/test.html'),
        ('http://example.com/', 'http://xkcd.com/about/', 'http://xkcd.com/about/'),
        ('http://example.com/', '//xkcd.com/about/', 'http://xkcd.com/about/')
    ]
)
def test_absolute_links(url: str, link: str, expected: str) -> None:
    head_template: str = "<head><base href='{}'></head>"
    body_template: str = "<body><a href='{}'>Next</a></body>"
    html_obj: HTML = HTML(html=body_template.format(link), url=url)
    assert html_obj.absolute_links.pop() == expected
    html_obj = HTML(html=head_template.format(url) + body_template.format(link), url='http://example.com/foobar/')
    assert html_obj.absolute_links.pop() == expected

def test_parser() -> None:
    doc: str = "<a href='https://httpbin.org'>httpbin.org\n</a>"
    html_obj: HTML = HTML(html=doc)
    assert html_obj.find('html')
    assert html_obj.element('a').text().strip() == 'httpbin.org'

@pytest.mark.render
def test_render() -> None:
    r: Response = get()
    script: str = (
        '\n    () => {\n'
        '        return {\n'
        '            width: document.documentElement.clientWidth,\n'
        '            height: document.documentElement.clientHeight,\n'
        '            deviceScaleFactor: window.devicePixelRatio,\n'
        '        }\n'
        '    }\n    '
    )
    val = r.html.render(script=script)
    for value in ('width', 'height', 'deviceScaleFactor'):
        assert value in val
    about: HTML = r.html.find('#about', first=True)
    assert len(about.links) == 6

@pytest.mark.render
@pytest.mark.asyncio
async def test_async_render(async_get: Callable[[], Awaitable[Response]]) -> None:
    r: Response = await async_get()
    script: str = (
        '\n    () => {\n'
        '        return {\n'
        '            width: document.documentElement.clientWidth,\n'
        '            height: document.documentElement.clientHeight,\n'
        '            deviceScaleFactor: window.devicePixelRatio,\n'
        '        }\n'
        '    }\n    '
    )
    val = await r.html.arender(script=script)
    for value in ('width', 'height', 'deviceScaleFactor'):
        assert value in val
    about: HTML = r.html.find('#about', first=True)
    assert len(about.links) == 6
    await r.html.browser.close()

@pytest.mark.render
def test_bare_render() -> None:
    doc: str = "<a href='https://httpbin.org'>"
    html_obj: HTML = HTML(html=doc)
    script: str = (
        '\n        () => {\n'
        '            return {\n'
        '                width: document.documentElement.clientWidth,\n'
        '                height: document.documentElement.clientHeight,\n'
        '                deviceScaleFactor: window.devicePixelRatio,\n'
        '            }\n'
        '        }\n    '
    )
    val = html_obj.render(script=script, reload=False)
    for value in ('width', 'height', 'deviceScaleFactor'):
        assert value in val
    assert html_obj.find('html')
    assert 'https://httpbin.org' in html_obj.links

@pytest.mark.render
@pytest.mark.asyncio
async def test_bare_arender() -> None:
    doc: str = "<a href='https://httpbin.org'>"
    html_obj: HTML = HTML(html=doc, async_=True)
    script: str = (
        '\n        () => {\n'
        '            return {\n'
        '                width: document.documentElement.clientWidth,\n'
        '                height: document.documentElement.clientHeight,\n'
        '                deviceScaleFactor: window.devicePixelRatio,\n'
        '            }\n'
        '        }\n    '
    )
    val = await html_obj.arender(script=script, reload=False)
    for value in ('width', 'height', 'deviceScaleFactor'):
        assert value in val
    assert html_obj.find('html')
    assert 'https://httpbin.org' in html_obj.links
    await html_obj.browser.close()

@pytest.mark.render
def test_bare_js_eval() -> None:
    doc: str = (
        '\n    <!DOCTYPE html>\n'
        '    <html>\n'
        '    <body>\n'
        '    <div id="replace">This gets replaced</div>\n\n'
        '    <script type="text/javascript">\n'
        '      document.getElementById("replace").innerHTML = "yolo";\n'
        '    </script>\n'
        '    </body>\n'
        '    </html>\n    '
    )
    html_obj: HTML = HTML(html=doc)
    html_obj.render()
    assert html_obj.find('#replace', first=True).text == 'yolo'

@pytest.mark.render
@pytest.mark.asyncio
async def test_bare_js_async_eval() -> None:
    doc: str = (
        '\n    <!DOCTYPE html>\n'
        '    <html>\n'
        '    <body>\n'
        '    <div id="replace">This gets replaced</div>\n\n'
        '    <script type="text/javascript">\n'
        '      document.getElementById("replace").innerHTML = "yolo";\n'
        '    </script>\n'
        '    </body>\n'
        '    </html>\n    '
    )
    html_obj: HTML = HTML(html=doc, async_=True)
    await html_obj.arender()
    assert html_obj.find('#replace', first=True).text == 'yolo'
    await html_obj.browser.close()

def test_browser_session() -> None:
    """ Test browser instances is created and properly close when session is closed.
        Note: session.close method need to be tested together with browser creation,
            since not doing that will leave the browser running. """
    session_obj: HTMLSession = HTMLSession()
    assert isinstance(session_obj.browser, Browser)
    assert hasattr(session_obj, 'loop')
    session_obj.close()

def test_browser_process() -> None:
    for _ in range(3):
        r: Response = get()
        r.html.render()
        assert r.html.page is None

@pytest.mark.asyncio
async def test_browser_session_fail() -> None:
    """ HTMLSession.browser should not be call within an existing event loop> """
    session_obj: HTMLSession = HTMLSession()
    with pytest.raises(RuntimeError):
        _ = session_obj.browser

@pytest.mark.asyncio
async def test_async_browser_session() -> None:
    session_obj: AsyncHTMLSession = AsyncHTMLSession()
    browser: Browser = await session_obj.browser
    assert isinstance(browser, Browser)
    await session_obj.close()