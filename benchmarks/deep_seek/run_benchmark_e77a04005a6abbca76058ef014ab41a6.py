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
from typing import Any, List, Callable
sys.modules['markupsafe'] = None
import mako
from mako.template import Template
from mako.lookup import TemplateLookup
__author__ = 'virhilo@gmail.com (Lukasz Fidosz)'
LOREM_IPSUM: str = 'Quisque lobortis hendrerit posuere. Curabitur\naliquet consequat sapien molestie pretium. Nunc adipiscing luc\ntus mi, viverra porttitor lorem vulputate et. Ut at purus sem,\nsed tincidunt ante. Vestibulum ante ipsum primis in faucibus\norci luctus et ultrices posuere cubilia Curae; Praesent pulvinar\nsodales justo at congue. Praesent aliquet facilisis nisl a\nmolestie. Sed tempus nisl ut augue eleifend tincidunt. Sed a\nlacinia nulla. Cras tortor est, mollis et consequat at,\nvulputate et orci. Nulla sollicitudin'
BASE_TEMPLATE: str = '\n<%def name="render_table(table)">\n    <table>\n    % for row in table:\n        <tr>\n        % for col in row:\n            <td>${col|h}</td>\n        % endfor\n        </tr>\n    % endfor\n    </table>\n</%def>\n<%def name="img(src, alt)">\n    <img src="${src|u}" alt="${alt}" />\n</%def>\n<html>\n    <head><title>${title|h,trim}</title></head>\n    <body>\n        ${next.body()}\n    </body>\n<html>\n'
PAGE_TEMPLATE: str = '\n<%inherit file="base.mako"/>\n<table>\n    % for row in table:\n        <tr>\n            % for col in row:\n                <td>${col}</td>\n            % endfor\n        </tr>\n    % endfor\n</table>\n% for nr in range(img_count):\n    ${parent.img(\'/foo/bar/baz.png\', \'no image :o\')}\n% endfor\n${next.body()}\n% for nr in paragraphs:\n    <p>${lorem|x}</p>\n% endfor\n${parent.render_table(table)}\n'
CONTENT_TEMPLATE: str = '\n<%inherit file="page.mako"/>\n<%def name="fun1()">\n    <span>fun1</span>\n</%def>\n<%def name="fun2()">\n    <span>fun2</span>\n</%def>\n<%def name="fun3()">\n    <span>foo3</span>\n</%def>\n<%def name="fun4()">\n    <span>foo4</span>\n</%def>\n<%def name="fun5()">\n    <span>foo5</span>\n</%def>\n<%def name="fun6()">\n    <span>foo6</span>\n</%def>\n<p>Lorem ipsum dolor sit amet, consectetur adipiscing elit.\nNam laoreet justo in velit faucibus lobortis. Sed dictum sagittis\nvolutpat. Sed adipiscing vestibulum consequat. Nullam laoreet, ante\nnec pretium varius, libero arcu porttitor orci, id cursus odio nibh\nnec leo. Vestibulum dapibus pellentesque purus, sed bibendum tortor\nlaoreet id. Praesent quis sodales ipsum. Fusce ut ligula sed diam\npretium sagittis vel at ipsum. Nulla sagittis sem quam, et volutpat\nvelit. Fusce dapibus ligula quis lectus ultricies tempor. Pellente</p>\n${fun1()}\n${fun2()}\n${fun3()}\n${fun4()}\n${fun5()}\n${fun6()}\n'

def bench_mako(runner: pyperf.Runner, table_size: int, nparagraph: int, img_count: int) -> None:
    lookup: TemplateLookup = TemplateLookup()
    lookup.put_string('base.mako', BASE_TEMPLATE)
    lookup.put_string('page.mako', PAGE_TEMPLATE)
    template: Template = Template(CONTENT_TEMPLATE, lookup=lookup)
    table: List[List[int]] = [list(range(table_size)) for i in range(table_size)]
    paragraphs: range = range(nparagraph)
    title: str = 'Hello world!'
    func: Callable[[], Any] = functools.partial(template.render, table=table, paragraphs=paragraphs, lorem=LOREM_IPSUM, title=title, img_count=img_count, range=range)
    runner.bench_func('mako', func)

if __name__ == '__main__':
    runner: pyperf.Runner = pyperf.Runner()
    runner.metadata['description'] = 'Mako templates'
    runner.metadata['mako_version'] = mako.__version__
    table_size: int = 150
    nparagraph: int = 50
    img_count: int = 50
    bench_mako(runner, table_size, nparagraph, img_count)
