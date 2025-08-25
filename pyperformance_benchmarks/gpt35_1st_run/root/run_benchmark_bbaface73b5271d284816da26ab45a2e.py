from pathlib import Path
import docutils
from docutils import core
import pyperf
from typing import Union

DOC_ROOT: Path = ((Path(__file__).parent / 'data') / 'docs').resolve()

def build_html(doc_root: Path) -> float:
    elapsed: float = 0
    for file in doc_root.rglob('*.txt'):
        file_contents: str = file.read_text(encoding='utf-8')
        t0: float = pyperf.perf_counter()
        with contextlib.suppress(docutils.ApplicationError):
            core.publish_string(source=file_contents, reader_name='standalone', parser_name='restructuredtext', writer_name='html5', settings_overrides={'input_encoding': 'unicode', 'output_encoding': 'unicode', 'report_level': 5})
        elapsed += (pyperf.perf_counter() - t0)
    return elapsed

def bench_docutils(loops: int, doc_root: Path) -> float:
    runs_total: float = 0
    for _ in range(loops):
        runs_total += build_html(doc_root)
    return runs_total
