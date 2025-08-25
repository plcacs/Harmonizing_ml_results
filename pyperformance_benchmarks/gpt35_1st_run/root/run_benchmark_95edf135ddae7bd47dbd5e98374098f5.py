from typing import IO
import io
import os.path
import html5lib
import pyperf

def bench_html5lib(html_file: IO[bytes]):
    html_file.seek(0)
    html5lib.parse(html_file)

if (__name__ == '__main__'):
    runner: pyperf.Runner = pyperf.Runner()
    runner.metadata['description'] = 'Test the performance of the html5lib parser.'
    runner.metadata['html5lib_version'] = html5lib.__version__
    filename: str = os.path.join(os.path.dirname(__file__), 'data', 'w3_tr_html5.html')
    with open(filename, 'rb') as fp:
        html_file: IO[bytes] = io.BytesIO(fp.read())
    runner.bench_func('html5lib', bench_html5lib, html_file)
