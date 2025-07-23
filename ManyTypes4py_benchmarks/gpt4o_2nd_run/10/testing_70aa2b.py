import re
import os
import warnings
import platform
import unittest
import inspect
import contextlib
from pathlib import Path
import typing as tp
import numpy as np
from . import errors
try:
    import pytest
except ImportError:
    pass

@contextlib.contextmanager
def suppress_nevergrad_warnings() -> tp.Generator[None, None, None]:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=errors.NevergradWarning)
        yield

def assert_set_equal(estimate: tp.Iterable[tp.Any], reference: tp.Iterable[tp.Any], err_msg: str = '') -> None:
    estimate, reference = (set(x) for x in [estimate, reference])
    elements = [('additional', estimate - reference), ('missing', reference - estimate)]
    messages = ['  - {} element(s): {}.'.format(name, s) for name, s in elements if s]
    if messages:
        messages = ([err_msg] if err_msg else []) + ['Sets are not equal:'] + messages
        raise AssertionError('\n'.join(messages))

def printed_assert_equal(actual: tp.Any, desired: tp.Any, err_msg: str = '') -> None:
    try:
        np.testing.assert_equal(actual, desired, err_msg=err_msg)
    except AssertionError as e:
        print('\n' + '# ' * 12 + 'DEBUG MESSAGE ' + '# ' * 12)
        print(f'Expected: {desired}\nbut got:  {actual}')
        raise e

def assert_markdown_links_not_broken(folder: tp.Union[str, Path]) -> None:
    links = _get_all_markdown_links(folder)
    broken = [l for l in links if not l.exists()]
    if broken:
        text = '\n - '.join([str(l) for l in broken])
        raise AssertionError(f'Broken markdown links:\n - {text}')

class _MarkdownLink:
    def __init__(self, folder: Path, filepath: Path, string: str, link: str) -> None:
        self._folder = folder
        self._filepath = filepath
        self._string = string
        self._link = link

    def exists(self) -> bool:
        if self._link.startswith('http'):
            return True
        fullpath = self._folder / self._filepath.parent / self._link
        return fullpath.exists()

    def __repr__(self) -> str:
        return f'{self._link} ({self._string}) from file {self._filepath}'

def _get_all_markdown_links(folder: tp.Union[str, Path]) -> tp.List[_MarkdownLink]:
    pattern = re.compile(r'\[(?P<string>.+?)\]\((?P<link>\S+?)\)')
    folder = Path(folder).expanduser().absolute()
    links = []
    for rfilepath in folder.glob('**/*.md'):
        if ('/site-packages/' if os.name != 'nt' else '\\site-packages\\') not in str(rfilepath):
            filepath = folder / rfilepath
            with filepath.open('r') as f:
                text = f.read()
            for match in pattern.finditer(text):
                links.append(_MarkdownLink(folder, rfilepath, match.group('string'), match.group('link')))
    return links

class parametrized:
    def __init__(self, **kwargs: tp.Tuple[tp.Any, ...]) -> None:
        self.ids = sorted(kwargs)
        self.params = tuple((kwargs[name] for name in self.ids))
        assert self.params
        self.num_params = len(self.params[0])
        assert all((isinstance(p, (tuple, list)) for p in self.params))
        assert all((self.num_params == len(p) for p in self.params[1:]))

    def __call__(self, func: tp.Callable) -> tp.Callable:
        names = list(inspect.signature(func).parameters.keys())
        assert len(names) == self.num_params, f'Parameter names: {names}'
        return pytest.mark.parametrize(','.join(names), self.params if self.num_params > 1 else [p[0] for p in self.params], ids=self.ids)(func)

@contextlib.contextmanager
def skip_error_on_systems(error_type: tp.Type[BaseException], systems: tp.List[str]) -> tp.Generator[None, None, None]:
    try:
        yield
    except error_type as e:
        system = platform.system()
        if system in systems:
            raise unittest.SkipTest(f'Skipping on system {system}')
        if systems:
            print(f'This is system "{system}" (should it be skipped for the test?)')
        raise e
