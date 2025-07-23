"""
.. _hypothesis-cli:

----------------
hypothesis[cli]
----------------

::

    $ hypothesis --help
    Usage: hypothesis [OPTIONS] COMMAND [ARGS]...

    Options:
      --version   Show the version and exit.
      -h, --help  Show this message and exit.

    Commands:
      codemod  `hypothesis codemod` refactors deprecated or inefficient code.
      fuzz     [hypofuzz] runs tests with an adaptive coverage-guided fuzzer.
      write    `hypothesis write` writes property-based tests for you!

This module requires the :pypi:`click` package, and provides Hypothesis' command-line
interface, for e.g. :doc:`'ghostwriting' tests <ghostwriter>` via the terminal.
It's also where `HypoFuzz <https://hypofuzz.com/>`__ adds the :command:`hypothesis fuzz`
command (`learn more about that here <https://hypofuzz.com/docs/quickstart.html>`__).
"""
import builtins
import importlib
import inspect
import sys
import types
from difflib import get_close_matches
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Callable, List, Optional, Set, Tuple, Union

try:
    import pytest
except ImportError:
    pytest = None

MESSAGE: str = "\nThe Hypothesis command-line interface requires the `{}` package,\nwhich you do not have installed.  Run:\n\n    python -m pip install --upgrade 'hypothesis[cli]'\n\nand try again.\n"

try:
    import click
except ImportError:

    def main() -> None:
        """If `click` is not installed, tell the user to install it then exit."""
        sys.stderr.write(MESSAGE.format('click'))
        sys.exit(1)
else:
    sys.path.append('.')

    @click.group(context_settings={'help_option_names': ('-h', '--help')})
    @click.version_option()
    def main() -> None:
        pass

    def obj_name(s: str) -> Any:
        """This "type" imports whatever object is named by a dotted string."""
        s = s.strip()
        if '/' in s or '\\' in s:
            raise click.UsageError('Remember that the ghostwriter should be passed the name of a module, not a path.') from None
        try:
            return importlib.import_module(s)
        except ImportError:
            pass
        classname: Optional[str] = None
        if '.' not in s:
            modulename, module, funcname = ('builtins', builtins, s)
        else:
            modulename, funcname = s.rsplit('.', 1)
            try:
                module = importlib.import_module(modulename)
            except ImportError as err:
                try:
                    modulename, classname = modulename.rsplit('.', 1)
                    module = importlib.import_module(modulename)
                except (ImportError, ValueError):
                    if s.endswith('.py'):
                        raise click.UsageError('Remember that the ghostwriter should be passed the name of a module, not a file.') from None
                    raise click.UsageError(f'Failed to import the {modulename} module for introspection.  Check spelling and your Python import path, or use the Python API?') from err

        def describe_close_matches(module_or_class: Any, objname: str) -> str:
            public_names = [name for name in vars(module_or_class) if not name.startswith('_')]
            matches = get_close_matches(objname, public_names)
            if matches:
                return f'  Closest matches: {matches!r}'
            else:
                return ''

        if classname is None:
            try:
                return getattr(module, funcname)
            except AttributeError as err:
                if funcname == 'py':
                    raise click.UsageError(f'Remember that the ghostwriter should be passed the name of a module, not a file.\n\tTry: hypothesis write {s[:-3]}') from None
                raise click.UsageError(f"Found the {modulename!r} module, but it doesn't have a {funcname!r} attribute." + describe_close_matches(module, funcname)) from err
        else:
            try:
                func_class = getattr(module, classname)
            except AttributeError as err:
                raise click.UsageError(f"Found the {modulename!r} module, but it doesn't have a {classname!r} class." + describe_close_matches(module, classname)) from err
            try:
                return getattr(func_class, funcname)
            except AttributeError as err:
                if inspect.isclass(func_class):
                    func_class_is = 'class'
                else:
                    func_class_is = 'attribute'
                raise click.UsageError(f"Found the {modulename!r} module and {classname!r} {func_class_is}, but it doesn't have a {funcname!r} attribute." + describe_close_matches(func_class, funcname)) from err

    def _refactor(func: Callable[[str], Optional[str]], fname: str) -> Optional[str]:
        try:
            oldcode: str = Path(fname).read_text(encoding='utf-8')
        except (OSError, UnicodeError) as err:
            return f'skipping {fname!r} due to {err}'
        if 'hypothesis' not in oldcode:
            return None
        try:
            newcode: str = func(oldcode)
        except Exception as err:
            from libcst import ParserSyntaxError
            if isinstance(err, ParserSyntaxError):
                from hypothesis.extra._patching import indent
                msg: str = indent(str(err).replace('\n\n', '\n'), '    ').strip()
                return f'skipping {fname!r} due to {msg}'
            raise
        if newcode != oldcode:
            Path(fname).write_text(newcode, encoding='utf-8')
        return None

    @main.command()
    @click.argument('path', type=str, required=True, nargs=-1)
    def codemod(path: Tuple[str, ...]) -> int:
        """`hypothesis codemod` refactors deprecated or inefficient code.

        It adapts `python -m libcst.tool`, removing many features and config options
        which are rarely relevant for this purpose.  If you need more control, we
        encourage you to use the libcst CLI directly; if not this one is easier.

        PATH is the file(s) or directories of files to format in place, or
        "-" to read from stdin and write to stdout.
        """
        try:
            from libcst.codemod import gather_files
            from hypothesis.extra import codemods
        except ImportError:
            sys.stderr.write('You are missing required dependencies for this option.  Run:\n\n    python -m pip install --upgrade hypothesis[codemods]\n\nand try again.')
            sys.exit(1)
        if '-' in path:
            if len(path) > 1:
                raise Exception('Cannot specify multiple paths when reading from stdin!')
            print('Codemodding from stdin', file=sys.stderr)
            print(codemods.refactor(sys.stdin.read()))
            return 0
        files: List[str] = gather_files(path)
        errors: Set[Optional[str]] = set()
        if len(files) <= 1:
            errors.add(_refactor(codemods.refactor, files[0]))
        else:
            with Pool() as pool:
                for msg in pool.imap_unordered(partial(_refactor, codemods.refactor), files):
                    errors.add(msg)
        errors.discard(None)
        for msg in errors:
            print(msg, file=sys.stderr)
        return 1 if errors else 0

    @main.command()
    @click.argument('func', type=obj_name, required=True, nargs=-1)
    @click.option('--roundtrip', 'writer', flag_value='roundtrip', help='start by testing write/read or encode/decode!')
    @click.option('--equivalent', 'writer', flag_value='equivalent', help='very useful when optimising or refactoring code')
    @click.option('--errors-equivalent', 'writer', flag_value='errors-equivalent', help='--equivalent, but also allows consistent errors')
    @click.option('--idempotent', 'writer', flag_value='idempotent', help='check that f(x) == f(f(x))')
    @click.option('--binary-op', 'writer', flag_value='binary_operation', help='associativity, commutativity, identity element')
    @click.option('--style', type=click.Choice(['pytest', 'unittest']), default='pytest' if pytest else 'unittest', help='pytest-style function, or unittest-style method?')
    @click.option('-e', '--except', 'except_', type=obj_name, multiple=True, help='dotted name of exception(s) to ignore')
    @click.option('--annotate/--no-annotate', default=None, help='force ghostwritten tests to be type-annotated (or not).  By default, match the code to test.')
    def write(
        func: Tuple[Any, ...],
        writer: Optional[str],
        except_: Tuple[Any, ...],
        style: str,
        annotate: Optional[bool]
    ) -> None:
        """`hypothesis write` writes property-based tests for you!

        Type annotations are helpful but not required for our advanced introspection
        and templating logic.  Try running the examples below to see how it works:

        \x08
            hypothesis write gzip
            hypothesis write numpy.matmul
            hypothesis write pandas.from_dummies
            hypothesis write re.compile --except re.error
            hypothesis write --equivalent ast.literal_eval eval
            hypothesis write --roundtrip json.dumps json.loads
            hypothesis write --style=unittest --idempotent sorted
            hypothesis write --binary-op operator.add
        """
        kwargs: dict = {'except_': except_ or (), 'style': style, 'annotate': annotate}
        if writer is None:
            writer = 'magic'
        elif writer == 'idempotent' and len(func) > 1:
            raise click.UsageError('Test functions for idempotence one at a time.')
        elif writer == 'roundtrip' and len(func) == 1:
            writer = 'idempotent'
        elif 'equivalent' in writer and len(func) == 1:
            writer = 'fuzz'
        if writer == 'errors-equivalent':
            writer = 'equivalent'
            kwargs['allow_same_errors'] = True
        try:
            from hypothesis.extra import ghostwriter
        except ImportError:
            sys.stderr.write(MESSAGE.format('black'))
            sys.exit(1)
        code: str = getattr(ghostwriter, writer)(*func, **kwargs)
        try:
            from rich.console import Console
            from rich.syntax import Syntax
            from hypothesis.utils.terminal import guess_background_color
        except ImportError:
            print(code)
        else:
            try:
                theme: str = 'default' if guess_background_color() == 'light' else 'monokai'
                syntax: Syntax = Syntax(code, 'python', background_color='default', theme=theme)
                Console().print(syntax, soft_wrap=True)
            except Exception:
                print('# Error while syntax-highlighting code', file=sys.stderr)
                print(code)
