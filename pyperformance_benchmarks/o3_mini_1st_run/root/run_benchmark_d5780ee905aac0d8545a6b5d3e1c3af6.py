#!/usr/bin/env python3
"""
Build a subset of Python's documentation using Sphinx
"""
import io
import os
from pathlib import Path
import shutil
import pyperf
from sphinx.cmd.build import main as sphinx_main
from typing import IO, BinaryIO, TextIO, Optional, Union, Callable, Any, Dict

# DOC_ROOT is the root path for documentation data.
DOC_ROOT: Path = ((Path(__file__).parent / 'data') / 'Doc').resolve()
_orig_open = open
preloaded_files: Dict[str, Union[bytes, io.BytesIO, io.StringIO]] = {}

def read_all_files() -> None:
    for filename in DOC_ROOT.glob('**/*'):
        if filename.is_file():
            preloaded_files[str(filename)] = filename.read_bytes()

def open(file: Union[str, Path],
         mode: str = 'r',
         buffering: int = -1,
         encoding: Optional[str] = None,
         errors: Optional[str] = None,
         newline: Optional[str] = None,
         closefd: bool = True,
         opener: Optional[Callable[..., Any]] = None) -> Union[TextIO, BinaryIO]:
    if isinstance(file, Path):
        file = str(file)
    if isinstance(file, str):
        if ('r' in mode) and (file in preloaded_files):
            if 'b' in mode:
                return io.BytesIO(preloaded_files[file])  # type: ignore
            else:
                return io.StringIO(preloaded_files[file].decode(encoding or 'utf-8'))  # type: ignore
        elif ('w' in mode) and (DOC_ROOT in Path(file).parents):
            if 'b' in mode:
                newfile = io.BytesIO()
            else:
                newfile = io.StringIO()
            preloaded_files[file] = newfile
            return newfile
    return _orig_open(file, mode=mode, buffering=buffering, encoding=encoding,
                      errors=errors, newline=newline, closefd=closefd, opener=opener)

__builtins__.open = open

def replace(src: str, dst: str) -> None:
    pass

os.replace = replace

def build_doc(doc_root: Path) -> float:
    t0: float = pyperf.perf_counter()
    sphinx_main(['--builder', 'dummy',
                 '--doctree-dir', str((doc_root / 'build') / 'doctrees'),
                 '--jobs', '1',
                 '--silent',
                 '--fresh-env',
                 '--write-all',
                 str(doc_root),
                 str((doc_root / 'build') / 'html')])
    return pyperf.perf_counter() - t0

def bench_sphinx(loops: int, doc_root: Path) -> float:
    build_dir: Path = DOC_ROOT / 'build'
    if build_dir.is_dir():
        shutil.rmtree(build_dir)
    read_all_files()
    runs_total: float = 0.0
    for _ in range(loops):
        runs_total += build_doc(doc_root)
        if build_dir.is_dir():
            shutil.rmtree(build_dir)
    return runs_total

if __name__ == '__main__':
    runner = pyperf.Runner()
    runner.metadata['description'] = 'Render documentation with Sphinx, like the CPython docs'
    args = runner.parse_args()
    runner.bench_time_func('sphinx', bench_sphinx, DOC_ROOT)