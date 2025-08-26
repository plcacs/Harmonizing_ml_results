"\nConvert Docutils' documentation from reStructuredText to <format>.\n"
import contextlib
from pathlib import Path
import docutils
from docutils import core
import pyperf
from typing import Union

try:
    from docutils.utils.math.math2html import Trace
except ImportError:
    pass
else:
    Trace.show = (lambda message, channel: ...)

DOC_ROOT: Path = ((Path(__file__).parent / 'data') / 'docs').resolve()

def build_html(doc_root: Path) -> float:
    elapsed: float = 0.0
    for file in doc_root.rglob('*.txt'):
        file_contents: str = file.read_text(encoding='utf-8')
        t0: float = pyperf.perf_counter()
        with contextlib.suppress(docutils.ApplicationError):
            core.publish_string(
                source=file_contents,
                reader_name='standalone',
                parser_name='restructuredtext',
                writer_name='html5',
                settings_overrides={
                    'input_encoding': 'unicode',
                    'output_encoding': 'unicode',
                    'report_level': 5
                }
            )
        elapsed += (pyperf.perf_counter() - t0)
    return elapsed

def bench_docutils(loops: int, doc_root: Path) -> float:
    runs_total: float = 0.0
    for _ in range(loops):
        runs_total += build_html(doc_root)
    return runs_total

if (__name__ == '__main__'):
    runner: pyperf.Runner = pyperf.Runner()
    runner.metadata['description'] = 'Render documentation with Docutils'
    args: Union[None, pyperf.Runner] = runner.parse_args()
    runner.bench_time_func('docutils', bench_docutils, DOC_ROOT)
