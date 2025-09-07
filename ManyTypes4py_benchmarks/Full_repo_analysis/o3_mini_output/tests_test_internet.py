import pytest
from requests_html import HTMLSession, AsyncHTMLSession, HTMLResponse
from typing import Any, List, Callable, Coroutine

urls: List[str] = [
    'https://xkcd.com/1957/',
    'https://www.reddit.com/',
    'https://github.com/psf/requests-html/issues',
    'https://discord.com/category/engineering',
    'https://stackoverflow.com/',
    'https://www.frontiersin.org/',
    'https://azure.microsoft.com/en-us'
]

@pytest.mark.parametrize('url', urls)
@pytest.mark.internet
def test_pagination(url: str) -> None:
    session: HTMLSession = HTMLSession()
    r: HTMLResponse = session.get(url)
    assert next(r.html)

@pytest.mark.parametrize('url', urls)
@pytest.mark.internet
@pytest.mark.asyncio
async def test_async_pagination(event_loop: Any, url: str) -> None:
    asession: AsyncHTMLSession = AsyncHTMLSession()
    r: HTMLResponse = await asession.get(url)
    assert await r.html.__anext__()

@pytest.mark.internet
def test_async_run() -> None:
    asession: AsyncHTMLSession = AsyncHTMLSession()
    async_list: List[Callable[[], Coroutine[Any, Any, HTMLResponse]]] = []
    for url in urls:
        async def _test(url: str = url) -> HTMLResponse:
            return await asession.get(url)
        async_list.append(_test)
    r: List[HTMLResponse] = asession.run(*async_list)
    assert len(r) == len(urls)
    assert isinstance(r[0], HTMLResponse)