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
    import pytest  # type: tp.Any
except ImportError:
    pass

@contextlib.contextmanager
def suppress_nevergrad_warnings() -> tp.Iterator[None]:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=errors.NevergradWarning)
        yield

def assert_set_equal(estimate: tp.Iterable[tp.Any], reference: tp.Iterable[tp.Any], err_msg: str = '') -> None:
    """Asserts that both sets are equals, with comprehensive error message.
    This function should only be used in tests.
    Parameters
    ----------
    estimate: iterable
        sequence of elements to compare with the reference set of elements
    reference: iterable
        reference sequence of elements
    """
    estimate_set, reference_set = (set(x) for x in [estimate, reference])
    elements = [('additional', estimate_set - reference_set), ('missing', reference_set - estimate_set)]
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
    """Asserts that all relative hyperlinks are valid in markdown files of the folder
    and its subfolders.

    Note
    ----
    http hyperlinks are not tested.
    """
    links = _get_all_markdown_links(folder)
    broken = [l for l in links if not l.exists()]
    if broken:
        text = '\n - '.join([str(l) for l in broken])
        raise AssertionError(f'Broken markdown links:\n - {text}')

class _MarkdownLink:
    """Handle to a markdown link, for easy existence test and printing
    (external links are not tested)
    """
    def __init__(self, folder: Path, filepath: Path, string: str, link: str) -> None:
        self._folder: Path = folder
        self._filepath: Path = filepath
        self._string: str = string
        self._link: str = link

    def exists(self) -> bool:
        if self._link.startswith('http'):
            return True
        fullpath = self._folder / self._filepath.parent / self._link
        return fullpath.exists()

    def __repr__(self) -> str:
        return f'{self._link} ({self._string}) from file {self._filepath}'

def _get_all_markdown_links(folder: tp.Union[str, Path]) -> tp.List[_MarkdownLink]:
    """Returns a list of all existing markdown links"""
    pattern = re.compile(r'\[(?P<string>.+?)\]\((?P<link>\S+?)\)')
    folder_path: Path = Path(folder).expanduser().absolute()
    links: tp.List[_MarkdownLink] = []
    for rfilepath in folder_path.glob('**/*.md'):
        if ('/site-packages/' if os.name != 'nt' else '\\site-packages\\') not in str(rfilepath):
            filepath: Path = folder_path / rfilepath
            with filepath.open('r') as f:
                text: str = f.read()
            for match in pattern.finditer(text):
                links.append(_MarkdownLink(folder_path, rfilepath, match.group('string'), match.group('link')))
    return links

class parametrized:
    """Simplified decorator API for specifying named parametrized test with pytests
    (like with old "genty" package)
    See example of use in test_testing

    Parameters
    ----------
    **kwargs:
        name of the argument is converted as id of the experiments, and the provided tuple
        contains a value for each of the arguments of the underlying function (in the definition order).
    """
    def __init__(self, **kwargs: tp.Sequence[tp.Any]) -> None:
        self.ids: tp.List[str] = sorted(kwargs)
        self.params: tp.Tuple[tp.Sequence[tp.Any], ...] = tuple((kwargs[name] for name in self.ids))
        assert self.params
        self.num_params: int = len(self.params[0])
        assert all(isinstance(p, (tuple, list)) for p in self.params)
        assert all(self.num_params == len(p) for p in self.params[1:])

    def __call__(self, func: tp.Callable[..., tp.Any]) -> tp.Callable[..., tp.Any]:
        names = list(inspect.signature(func).parameters.keys())
        assert len(names) == self.num_params, f'Parameter names: {names}'
        if self.num_params > 1:
            parameters = self.params  # type: tp.Any
        else:
            parameters = [p[0] for p in self.params]
        return pytest.mark.parametrize(','.join(names), parameters, ids=self.ids)(func)

@contextlib.contextmanager
def skip_error_on_systems(error_type: tp.Type[BaseException], systems: tp.Iterable[str]) -> tp.Iterator[None]:
    """Context manager for skipping a test upon a specific error on specific systems
    This is mostly used to skip some tests for features which are incompatible with Windows
    Eg. of systems (mind the capitalized first letter): Darwin, Windows
    """
    try:
        yield
    except error_type as e:
        system = platform.system()
        if system in systems:
            raise unittest.SkipTest(f'Skipping on system {system}')
        if systems:
            print(f'This is system "{system}" (should it be skipped for the test?)')
        raise e