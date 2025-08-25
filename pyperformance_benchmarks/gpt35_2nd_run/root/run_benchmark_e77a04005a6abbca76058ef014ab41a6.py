from typing import List, Tuple, Any, Callable

def bench_mako(runner: Any, table_size: int, nparagraph: int, img_count: int) -> None:
    lookup: Any = TemplateLookup()
    lookup.put_string('base.mako', BASE_TEMPLATE)
    lookup.put_string('page.mako', PAGE_TEMPLATE)
    template: Any = Template(CONTENT_TEMPLATE, lookup=lookup)
    table: List[List[int]] = [list(range(table_size)) for i in range(table_size)]
    paragraphs: List[int] = list(range(nparagraph))
    title: str = 'Hello world!'
    func: Callable = functools.partial(template.render, table=table, paragraphs=paragraphs, lorem=LOREM_IPSUM, title=title, img_count=img_count, range=range)
    runner.bench_func('mako', func)
