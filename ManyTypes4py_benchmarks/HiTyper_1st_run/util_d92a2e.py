import inspect
import os
from types import FrameType
from typing import Any, List, Optional
from monkeytype.compat import cached_property

class Dummy:

    @staticmethod
    def a_static_method(foo: Union[typing.Sequence[typing.Any], None, str, os.DirEntry]):
        return inspect.currentframe()

    @classmethod
    def a_class_method(cls: Union[typing.Type, str, typing.Callable[..., T]], foo: Union[typing.Type, str, typing.Callable[..., T]]):
        return inspect.currentframe()

    def an_instance_method(self, foo: Union[str, typing.IO], bar: Union[str, typing.IO]):
        return inspect.currentframe()

    def has_complex_signature(self, a, b, /, c: Union[int, list], d: int=0, *e, f: Union[int, list], g: int=0, **h):
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

        def f(self) -> None:
            pass

def transform_path(path: str) -> str:
    """Transform tests/test_foo.py to monkeytype.foo"""
    path = 'monkeytype/' + path[len('tests/'):]
    *basepath, file_name = path.split('/')
    basename, _ext = os.path.splitext(file_name[len('test_'):])
    return '.'.join(basepath + [basename])

def smartcov_paths_hook(paths: Union[str, os.PathLike]) -> Union[list[typing.Text], list]:
    """Given list of test files to run, return modules to measure coverage of."""
    if not paths:
        return ['monkeytype']
    return [transform_path(path) for path in paths]