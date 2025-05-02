import inspect
import os
from types import FrameType
from typing import Any, List, Optional
from monkeytype.compat import cached_property

class Dummy:

    @staticmethod
    def a_static_method(foo):
        return inspect.currentframe()

    @classmethod
    def a_class_method(cls, foo):
        return inspect.currentframe()

    def an_instance_method(self, foo, bar):
        return inspect.currentframe()

    def has_complex_signature(self, a, b, /, c, d=0, *e, f, g=0, **h):
        return inspect.currentframe()

    @property
    def a_property(self):
        return inspect.currentframe()

    @property
    def a_settable_property(self):
        return inspect.currentframe()

    @a_settable_property.setter
    def a_settable_property(self, unused):
        return inspect.currentframe()
    if cached_property:

        @cached_property
        def a_cached_property(self):
            return inspect.currentframe()

class Outer:

    class Inner:

        def f(self):
            pass

def transform_path(path):
    """Transform tests/test_foo.py to monkeytype.foo"""
    path = 'monkeytype/' + path[len('tests/'):]
    *basepath, file_name = path.split('/')
    basename, _ext = os.path.splitext(file_name[len('test_'):])
    return '.'.join(basepath + [basename])

def smartcov_paths_hook(paths):
    """Given list of test files to run, return modules to measure coverage of."""
    if not paths:
        return ['monkeytype']
    return [transform_path(path) for path in paths]