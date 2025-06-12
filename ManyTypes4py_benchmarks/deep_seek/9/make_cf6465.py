"""
Python script for building documentation.

To build the docs you must have all optional dependencies for pandas
installed. See the installation instructions for a list of these.

Usage
-----
    $ python make.py clean
    $ python make.py html
    $ python make.py latex
"""
import argparse
import csv
import importlib
import os
import shutil
import subprocess
import sys
import webbrowser
from typing import Any, List, Optional, Union, cast
import docutils
import docutils.parsers.rst
from docutils.nodes import Node, section, title
from docutils.utils import new_document
from docutils.frontend import get_default_settings

DOC_PATH: str = os.path.dirname(os.path.abspath(__file__))
SOURCE_PATH: str = os.path.join(DOC_PATH, 'source')
BUILD_PATH: str = os.path.join(DOC_PATH, 'build')
REDIRECTS_FILE: str = os.path.join(DOC_PATH, 'redirects.csv')

class DocBuilder:
    """
    Class to wrap the different commands of this script.

    All public methods of this class can be called as parameters of the
    script.
    """

    def __init__(
        self,
        num_jobs: str = 'auto',
        include_api: bool = True,
        whatsnew: bool = False,
        single_doc: Optional[str] = None,
        verbosity: int = 0,
        warnings_are_errors: bool = False,
        no_browser: bool = False
    ) -> None:
        self.num_jobs: str = num_jobs
        self.include_api: bool = include_api
        self.whatsnew: bool = whatsnew
        self.verbosity: int = verbosity
        self.warnings_are_errors: bool = warnings_are_errors
        self.no_browser: bool = no_browser
        if single_doc:
            single_doc = self._process_single_doc(single_doc)
            os.environ['SPHINX_PATTERN'] = single_doc
        elif not include_api:
            os.environ['SPHINX_PATTERN'] = '-api'
        elif whatsnew:
            os.environ['SPHINX_PATTERN'] = 'whatsnew'
        self.single_doc_html: Optional[str] = None
        if single_doc and single_doc.endswith('.rst'):
            self.single_doc_html = os.path.splitext(single_doc)[0] + '.html'
        elif single_doc:
            self.single_doc_html = f'reference/api/pandas.{single_doc}.html'

    def _process_single_doc(self, single_doc: str) -> str:
        """
        Make sure the provided value for --single is a path to an existing
        .rst/.ipynb file, or a pandas object that can be imported.

        For example, categorial.rst or pandas.DataFrame.head. For the latter,
        return the corresponding file path
        (e.g. reference/api/pandas.DataFrame.head.rst).
        """
        base_name, extension = os.path.splitext(single_doc)
        if extension in ('.rst', '.ipynb'):
            if os.path.exists(os.path.join(SOURCE_PATH, single_doc)):
                return single_doc
            else:
                raise FileNotFoundError(f'File {single_doc} not found')
        elif single_doc.startswith('pandas.'):
            try:
                obj: Any = pandas
                for name in single_doc.split('.'):
                    obj = getattr(obj, name)
            except AttributeError as err:
                raise ImportError(f'Could not import {single_doc}') from err
            else:
                return single_doc[len('pandas.'):]
        else:
            raise ValueError(f'--single={single_doc} not understood. Value should be a valid path to a .rst or .ipynb file, or a valid pandas object (e.g. categorical.rst or pandas.DataFrame.head)')

    @staticmethod
    def _run_os(*args: str) -> None:
        """
        Execute a command as a OS terminal.

        Parameters
        ----------
        *args : list of str
            Command and parameters to be executed

        Examples
        --------
        >>> DocBuilder()._run_os("python", "--version")
        """
        subprocess.check_call(args, stdout=sys.stdout, stderr=sys.stderr)

    def _sphinx_build(self, kind: str) -> int:
        """
        Call sphinx to build documentation.

        Attribute `num_jobs` from the class is used.

        Parameters
        ----------
        kind : {'html', 'latex', 'linkcheck'}

        Examples
        --------
        >>> DocBuilder(num_jobs=4)._sphinx_build("html")
        """
        if kind not in ('html', 'latex', 'linkcheck'):
            raise ValueError(f'kind must be html, latex or linkcheck, not {kind}')
        cmd: List[str] = ['sphinx-build', '-b', kind]
        if self.num_jobs:
            cmd += ['-j', self.num_jobs]
        if self.warnings_are_errors:
            cmd += ['-W', '--keep-going']
        if self.verbosity:
            cmd.append(f'-{"v" * self.verbosity}')
        cmd += ['-d', os.path.join(BUILD_PATH, 'doctrees'), SOURCE_PATH, os.path.join(BUILD_PATH, kind)]
        return subprocess.call(cmd)

    def _open_browser(self, single_doc_html: str) -> None:
        """
        Open a browser tab showing single
        """
        url: str = os.path.join('file://', DOC_PATH, 'build', 'html', single_doc_html)
        webbrowser.open(url, new=2)

    def _get_page_title(self, page: str) -> str:
        """
        Open the rst file `page` and extract its title.
        """
        fname: str = os.path.join(SOURCE_PATH, f'{page}.rst')
        doc: Any = new_document('<doc>', get_default_settings(docutils.parsers.rst.Parser))
        with open(fname, encoding='utf-8') as f:
            data: str = f.read()
        parser: Any = docutils.parsers.rst.Parser()
        with open(os.devnull, 'a', encoding='utf-8') as f:
            doc.reporter.stream = f
            parser.parse(data, doc)
        section_node: Node = next((node for node in doc.children if isinstance(node, section)))
        title_node: Node = next((node for node in section_node.children if isinstance(node, title)))
        return cast(str, title_node.astext())

    def _add_redirects(self) -> None:
        """
        Create in the build directory an html file with a redirect,
        for every row in REDIRECTS_FILE.
        """
        with open(REDIRECTS_FILE, encoding='utf-8') as mapping_fd:
            reader: csv.reader = csv.reader(mapping_fd)
            for row in reader:
                if not row or row[0].strip().startswith('#'):
                    continue
                html_path: str = os.path.join(BUILD_PATH, 'html')
                path: str = os.path.join(html_path, *row[0].split('/')) + '.html'
                if not self.include_api and (os.path.join(html_path, 'reference') in path or os.path.join(html_path, 'generated') in path):
                    continue
                try:
                    title: str = self._get_page_title(row[1])
                except Exception:
                    title = 'this page'
                with open(path, 'w', encoding='utf-8') as moved_page_fd:
                    html: str = f'<html>\n    <head>\n        <meta http-equiv="refresh" content="0;URL={row[1]}.html"/>\n    </head>\n    <body>\n        <p>\n            The page has been moved to <a href="{row[1]}.html">{title}</a>\n        </p>\n    </body>\n<html>'
                    moved_page_fd.write(html)

    def html(self) -> int:
        """
        Build HTML documentation.
        """
        ret_code: int = self._sphinx_build('html')
        zip_fname: str = os.path.join(BUILD_PATH, 'html', 'pandas.zip')
        if os.path.exists(zip_fname):
            os.remove(zip_fname)
        if ret_code == 0:
            if self.single_doc_html is not None:
                if not self.no_browser:
                    self._open_browser(self.single_doc_html)
            else:
                self._add_redirects()
                if self.whatsnew and (not self.no_browser):
                    self._open_browser(os.path.join('whatsnew', 'index.html'))
        return ret_code

    def latex(self, force: bool = False) -> int:
        """
        Build PDF documentation.
        """
        if sys.platform == 'win32':
            sys.stderr.write('latex build has not been tested on windows\n')
            return 1
        else:
            ret_code: int = self._sphinx_build('latex')
            os.chdir(os.path.join(BUILD_PATH, 'latex'))
            if force:
                for i in range(3):
                    self._run_os('pdflatex', '-interaction=nonstopmode', 'pandas.tex')
                raise SystemExit('You should check the file "build/latex/pandas.pdf" for problems.')
            self._run_os('make')
            return ret_code

    def latex_forced(self) -> int:
        """
        Build PDF documentation with retries to find missing references.
        """
        return self.latex(force=True)

    @staticmethod
    def clean() -> None:
        """
        Clean documentation generated files.
        """
        shutil.rmtree(BUILD_PATH, ignore_errors=True)
        shutil.rmtree(os.path.join(SOURCE_PATH, 'reference', 'api'), ignore_errors=True)

    def zip_html(self) -> None:
        """
        Compress HTML documentation into a zip file.
        """
        zip_fname: str = os.path.join(BUILD_PATH, 'html', 'pandas.zip')
        if os.path.exists(zip_fname):
            os.remove(zip_fname)
        dirname: str = os.path.join(BUILD_PATH, 'html')
        fnames: List[str] = os.listdir(dirname)
        os.chdir(dirname)
        self._run_os('zip', zip_fname, '-r', '-q', *fnames)

    def linkcheck(self) -> int:
        """
        Check for broken links in the documentation.
        """
        return self._sphinx_build('linkcheck')

def main() -> int:
    cmds: List[str] = [method for method in dir(DocBuilder) if not method.startswith('_')]
    joined: str = ','.join(cmds)
    argparser: argparse.ArgumentParser = argparse.ArgumentParser(description='pandas documentation builder', epilog=f'Commands: {joined}')
    joined = ', '.join(cmds)
    argparser.add_argument('command', nargs='?', default='html', help=f'command to run: {joined}')
    argparser.add_argument('--num-jobs', default='auto', help='number of jobs used by sphinx-build')
    argparser.add_argument('--no-api', default=False, help='omit api and autosummary', action='store_true')
    argparser.add_argument('--whatsnew', default=False, help='only build whatsnew (and api for links)', action='store_true')
    argparser.add_argument('--single', metavar='FILENAME', type=str, default=None, help="filename (relative to the 'source' folder) of section or method name to compile, e.g. 'development/contributing.rst', 'pandas.DataFrame.join'")
    argparser.add_argument('--python-path', type=str, default=os.path.dirname(DOC_PATH), help='path')
    argparser.add_argument('-v', action='count', dest='verbosity', default=0, help='increase verbosity (can be repeated), passed to the sphinx build command')
    argparser.add_argument('--warnings-are-errors', '-W', action='store_true', help='fail if warnings are raised')
    argparser.add_argument('--no-browser', help="Don't open browser", default=False, action='store_true')
    args: argparse.Namespace = argparser.parse_args()
    if args.command not in cmds:
        joined = ', '.join(cmds)
        raise ValueError(f'Unknown command {args.command}. Available options: {joined}')
    os.environ['PYTHONPATH'] = args.python_path
    sys.path.insert(0, args.python_path)
    globals()['pandas'] = importlib.import_module('pandas')
    os.environ['MPLBACKEND'] = 'module://matplotlib.backends.backend_agg'
    builder: DocBuilder = DocBuilder(args.num_jobs, not args.no_api, args.whatsnew, args.single, args.verbosity, args.warnings_are_errors, args.no_browser)
    return getattr(builder, args.command)()

if __name__ == '__main__':
    sys.exit(main())
