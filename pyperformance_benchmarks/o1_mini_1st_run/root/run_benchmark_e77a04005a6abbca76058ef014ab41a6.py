"""
Benchmark for test the performance of Mako templates engine.
Includes:
    -two template inherences
    -HTML escaping, XML escaping, URL escaping, whitespace trimming
    -function defitions and calls
    -forloops
"""
import functools
import sys
import pyperf
from typing import List, Callable
from mako.template import Template
from mako.lookup import TemplateLookup

__author__: str = 'virhilo@gmail.com (Lukasz Fidosz)'

LOREM_IPSUM: str = (
    'Quisque lobortis hendrerit posuere. Curabitur\n'
    'aliquet consequat sapien molestie pretium. Nunc adipiscing luc\n'
    'tus mi, viverra porttitor lorem vulputate et. Ut at purus sem,\n'
    'sed tincidunt ante. Vestibulum ante ipsum primis in faucibus\n'
    'orci luctus et ultrices posuere cubilia Curae; Praesent pulvinar\n'
    'sodales justo at congue. Praesent aliquet facilisis nisl a\n'
    'molestie. Sed tempus nisl ut augue eleifend tincidunt. Sed a\n'
    'lacinia nulla. Cras tortor est, mollis et consequat at,\n'
    'vulputate et orci. Nulla sollicitudin'
)

BASE_TEMPLATE: str = (
    '\n<%def name="render_table(table)">\n'
    '    <table>\n'
    '    % for row in table:\n'
    '        <tr>\n'
    '        % for col in row:\n'
    '            <td>${col|h}</td>\n'
    '        % endfor\n'
    '        </tr>\n'
    '    % endfor\n'
    '    </table>\n'
    '</%def>\n'
    '<%def name="img(src, alt)">\n'
    '    <img src="${src|u}" alt="${alt}" />\n'
    '</%def>\n'
    '<html>\n'
    '    <head><title>${title|h,trim}</title></head>\n'
    '    <body>\n'
    '        ${next.body()}\n'
    '    </body>\n'
    '<html>\n'
)

PAGE_TEMPLATE: str = (
    '\n<%inherit file="base.mako"/>\n'
    '<table>\n'
    '    % for row in table:\n'
    '        <tr>\n'
    '            % for col in row:\n'
    '                <td>${col}</td>\n'
    '            % endfor\n'
    '        </tr>\n'
    '    % endfor\n'
    '</table>\n'
    '% for nr in range(img_count):\n'
    '    ${parent.img(\'/foo/bar/baz.png\', \'no image :o\')}\n'
    '% endfor\n'
    '${next.body()}\n'
    '% for nr in paragraphs:\n'
    '    <p>${lorem|x}</p>\n'
    '% endfor\n'
    '${parent.render_table(table)}\n'
)

CONTENT_TEMPLATE: str = (
    '\n<%inherit file="page.mako"/>\n'
    '<%def name="fun1()">\n'
    '    <span>fun1</span>\n'
    '</%def>\n'
    '<%def name="fun2()">\n'
    '    <span>fun2</span>\n'
    '</%def>\n'
    '<%def name="fun3()">\n'
    '    <span>foo3</span>\n'
    '</%def>\n'
    '<%def name="fun4()">\n'
    '    <span>foo4</span>\n'
    '</%def>\n'
    '<%def name="fun5()">\n'
    '    <span>foo5</span>\n'
    '</%def>\n'
    '<%def name="fun6()">\n'
    '    <span>foo6</span>\n'
    '</%def>\n'
    '<p>Lorem ipsum dolor sit amet, consectetur adipiscing elit.\n'
    'Nam laoreet justo in velit faucibus lobortis. Sed dictum sagittis\n'
    'volutpat. Sed adipiscing vestibulum consequat. Nullam laoreet, ante\n'
    'nec pretium varius, libero arcu porttitor orci, id cursus odio nibh\n'
    'nec leo. Vestibulum dapibus pellentesque purus, sed bibendum tortor\n'
    'laoreet id. Praesent quis sodales ipsum. Fusce ut ligula sed diam\n'
    'pretium sagittis vel at ipsum. Nulla sagittis sem quam, et volutpat\n'
    'velit. Fusce dapibus ligula quis lectus ultricies tempor. Pellente</p>\n'
    '${fun1()}\n'
    '${fun2()}\n'
    '${fun3()}\n'
    '${fun4()}\n'
    '${fun5()}\n'
    '${fun6()}\n'
)

def bench_mako(
    runner: pyperf.Runner,
    table_size: int,
    nparagraph: int,
    img_count: int
) -> None:
    lookup: TemplateLookup = TemplateLookup()
    lookup.put_string('base.mako', BASE_TEMPLATE)
    lookup.put_string('page.mako', PAGE_TEMPLATE)
    template: Template = Template(CONTENT_TEMPLATE, lookup=lookup)
    table: List[range] = [range(table_size) for _ in range(table_size)]
    paragraphs: range = range(nparagraph)
    title: str = 'Hello world!'
    func: Callable[[], str] = functools.partial(
        template.render,
        table=table,
        paragraphs=paragraphs,
        lorem=LOREM_IPSUM,
        title=title,
        img_count=img_count,
        range=range
    )
    runner.bench_func('mako', func)

if __name__ == '__main__':
    runner: pyperf.Runner = pyperf.Runner()
    runner.metadata['description'] = 'Mako templates'
    runner.metadata['mako_version'] = mako.__version__
    table_size: int = 150
    nparagraph: int = 50
    img_count: int = 50
    bench_mako(runner, table_size, nparagraph, img_count)
