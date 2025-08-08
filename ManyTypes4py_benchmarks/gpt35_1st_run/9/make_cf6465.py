import argparse
import csv
import importlib
import os
import shutil
import subprocess
import sys
import webbrowser
import docutils
import docutils.parsers.rst
from typing import Optional

DOC_PATH: str = os.path.dirname(os.path.abspath(__file__))
SOURCE_PATH: str = os.path.join(DOC_PATH, 'source')
BUILD_PATH: str = os.path.join(DOC_PATH, 'build')
REDIRECTS_FILE: str = os.path.join(DOC_PATH, 'redirects.csv')

class DocBuilder:
    def __init__(self, num_jobs: str = 'auto', include_api: bool = True, whatsnew: bool = False, single_doc: Optional[str] = None, verbosity: int = 0, warnings_are_errors: bool = False, no_browser: bool = False) -> None:
        ...

    def _process_single_doc(self, single_doc: str) -> str:
        ...

    @staticmethod
    def _run_os(*args: str) -> None:
        ...

    def _sphinx_build(self, kind: str) -> int:
        ...

    def _open_browser(self, single_doc_html: str) -> None:
        ...

    def _get_page_title(self, page: str) -> str:
        ...

    def _add_redirects(self) -> None:
        ...

    def html(self) -> int:
        ...

    def latex(self, force: bool = False) -> int:
        ...

    def latex_forced(self) -> int:
        ...

    @staticmethod
    def clean() -> None:
        ...

    def zip_html(self) -> None:
        ...

    def linkcheck(self) -> int:
        ...

def main() -> int:
    ...

if __name__ == '__main__':
    sys.exit(main())
