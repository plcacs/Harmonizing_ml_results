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
        self.num_jobs = num_jobs
        self.include_api = include_api
        self.whatsnew = whatsnew
        self.verbosity = verbosity
        self.warnings_are_errors = warnings_are_errors
        self.no_browser = no_browser
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
        base_name, extension = os.path.splitext(single_doc)
        if extension in ('.rst', '.ipynb'):
            if os.path.exists(os.path.join(SOURCE_PATH, single_doc)):
                return single_doc
            else:
                raise FileNotFoundError(f'File {single_doc} not found')
        elif single_doc.startswith('pandas.'):
            try:
                obj = pandas
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
        subprocess.check_call(args, stdout=sys.stdout, stderr=sys.stderr)

    def _sphinx_build(self, kind: str) -> int:
        if kind not in ('html', 'latex', 'linkcheck'):
            raise ValueError(f'kind must be html, latex or linkcheck, not {kind}')
        cmd = ['sphinx-build', '-b', kind]
        if self.num_jobs:
            cmd += ['-j', self.num_jobs]
        if self.warnings_are_errors:
            cmd += ['-W', '--keep-going']
        if self.verbosity:
            cmd.append(f'-{"v" * self.verbosity}')
        cmd += ['-d', os.path.join(BUILD_PATH, 'doctrees'), SOURCE_PATH, os.path.join(BUILD_PATH, kind)]
        return subprocess.call(cmd)

    def _open_browser(self, single_doc_html: str) -> None:
        url = os.path.join('file://', DOC_PATH, 'build', 'html', single_doc_html)
        webbrowser.open(url, new=2)

    def _get_page_title(self, page: str) -> str:
        fname = os.path.join(SOURCE_PATH, f'{page}.rst')
        doc = docutils.utils.new_document('<doc>', docutils.frontend.get_default_settings(docutils.parsers.rst.Parser))
        with open(fname, encoding='utf-8') as f:
            data = f.read()
        parser = docutils.parsers.rst.Parser()
        with open(os.devnull, 'a', encoding='utf-8') as f:
            doc.reporter.stream = f
            parser.parse(data, doc)
        section = next((node for node in doc.children if isinstance(node, docutils.nodes.section)))
        title = next((node for node in section.children if isinstance(node, docutils.nodes.title)))
        return title.astext()

    def _add_redirects(self) -> None:
        with open(REDIRECTS_FILE, encoding='utf-8') as mapping_fd:
            reader = csv.reader(mapping_fd)
            for row in reader:
                if not row or row[0].strip().startswith('#'):
                    continue
                html_path = os.path.join(BUILD_PATH, 'html')
                path = os.path.join(html_path, *row[0].split('/')) + '.html'
                if not self.include_api and (os.path.join(html_path, 'reference') in path or os.path.join(html_path, 'generated') in path):
                    continue
                try:
                    title = self._get_page_title(row[1])
                except Exception:
                    title = 'this page'
                with open(path, 'w', encoding='utf-8') as moved_page_fd:
                    html = f'<html>\n    <head>\n        <meta http-equiv="refresh" content="0;URL={row[1]}.html"/>\n    </head>\n    <body>\n        <p>\n            The page has been moved to <a href="{row[1]}.html">{title}</a>\n        </p>\n    </body>\n<html>'
                    moved_page_fd.write(html)

    def html(self) -> int:
        ret_code = self._sphinx_build('html')
        zip_fname = os.path.join(BUILD_PATH, 'html', 'pandas.zip')
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

    def latex(self, force: bool = False) -> Optional[int]:
        if sys.platform == 'win32':
            sys.stderr.write('latex build has not been tested on windows\n')
        else:
            ret_code = self._sphinx_build('latex')
            os.chdir(os.path.join(BUILD_PATH, 'latex'))
            if force:
                for i in range(3):
                    self._run_os('pdflatex', '-interaction=nonstopmode', 'pandas.tex')
                raise SystemExit('You should check the file "build/latex/pandas.pdf" for problems.')
            self._run_os('make')
            return ret_code

    def latex_forced(self) -> Optional[int]:
        return self.latex(force=True)

    @staticmethod
    def clean() -> None:
        shutil.rmtree(BUILD_PATH, ignore_errors=True)
        shutil.rmtree(os.path.join(SOURCE_PATH, 'reference', 'api'), ignore_errors=True)

    def zip_html(self) -> None:
        zip_fname = os.path.join(BUILD_PATH, 'html', 'pandas.zip')
        if os.path.exists(zip_fname):
            os.remove(zip_fname)
        dirname = os.path.join(BUILD_PATH, 'html')
        fnames = os.listdir(dirname)
        os.chdir(dirname)
        self._run_os('zip', zip_fname, '-r', '-q', *fnames)

    def linkcheck(self) -> int:
        return self._sphinx_build('linkcheck')

def main() -> int:
    cmds = [method for method in dir(DocBuilder) if not method.startswith('_')]
    joined = ','.join(cmds)
    argparser = argparse.ArgumentParser(description='pandas documentation builder', epilog=f'Commands: {joined}')
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
    args = argparser.parse_args()
    if args.command not in cmds:
        joined = ', '.join(cmds)
        raise ValueError(f'Unknown command {args.command}. Available options: {joined}')
    os.environ['PYTHONPATH'] = args.python_path
    sys.path.insert(0, args.python_path)
    globals()['pandas'] = importlib.import_module('pandas')
    os.environ['MPLBACKEND'] = 'module://matplotlib.backends.backend_agg'
    builder = DocBuilder(args.num_jobs, not args.no_api, args.whatsnew, args.single, args.verbosity, args.warnings_are_errors, args.no_browser)
    return getattr(builder, args.command)()

if __name__ == '__main__':
    sys.exit(main())
