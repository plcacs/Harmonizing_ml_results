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
def suppress_nevergrad_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=errors.NevergradWarning)
        yield

def assert_set_equal(estimate, reference, err_msg=''):
    """Asserts that both sets are equals, with comprehensive error message.
    This function should only be used in tests.
    Parameters
    ----------
    estimate: iterable
        sequence of elements to compare with the reference set of elements
    reference: iterable
        reference sequence of elements
    """
    estimate, reference = (set(x) for x in [estimate, reference])
    elements = [('additional', estimate - reference), ('missing', reference - estimate)]
    messages = ['  - {} element(s): {}.'.format(name, s) for name, s in elements if s]
    if messages:
        messages = ([err_msg] if err_msg else []) + ['Sets are not equal:'] + messages
        raise AssertionError('\n'.join(messages))

def printed_assert_equal(actual, desired, err_msg=''):
    try:
        np.testing.assert_equal(actual, desired, err_msg=err_msg)
    except AssertionError as e:
        print('\n' + '# ' * 12 + 'DEBUG MESSAGE ' + '# ' * 12)
        print(f'Expected: {desired}\nbut got:  {actual}')
        raise e

def assert_markdown_links_not_broken(folder):
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

    def __init__(self, folder, filepath, string, link):
        self._folder = folder
        self._filepath = filepath
        self._string = string
        self._link = link

    def exists(self):
        if self._link.startswith('http'):
            return True
        fullpath = self._folder / self._filepath.parent / self._link
        return fullpath.exists()

    def __repr__(self):
        return f'{self._link} ({self._string}) from file {self._filepath}'

def _get_all_markdown_links(folder):
    """Returns a list of all existing markdown links"""
    pattern = re.compile('\\[(?P<string>.+?)\\]\\((?P<link>\\S+?)\\)')
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
    """Simplified decorator API for specifying named parametrized test with pytests
    (like with old "genty" package)
    See example of use in test_testing

    Parameters
    ----------
    **kwargs:
        name of the argument is converted as id of the experiments, and the provided tuple
        contains a value for each of the arguments of the underlying function (in the definition order).
    """

    def __init__(self, **kwargs):
        self.ids = sorted(kwargs)
        self.params = tuple((kwargs[name] for name in self.ids))
        assert self.params
        self.num_params = len(self.params[0])
        assert all((isinstance(p, (tuple, list)) for p in self.params))
        assert all((self.num_params == len(p) for p in self.params[1:]))

    def __call__(self, func):
        names = list(inspect.signature(func).parameters.keys())
        assert len(names) == self.num_params, f'Parameter names: {names}'
        return pytest.mark.parametrize(','.join(names), self.params if self.num_params > 1 else [p[0] for p in self.params], ids=self.ids)(func)

@contextlib.contextmanager
def skip_error_on_systems(error_type, systems):
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