import inspect
import os
from types import FrameType
from typing import Any, List, Optional
from monkeytype.compat import cached_property

class Dummy:
    @staticmethod
    def a_static_method(foo: Any) -> FrameType: ...
    @classmethod
    def a_class_method(cls, foo: Any) -> FrameType: ...
    def an_instance_method(self, foo: Any, bar: Any) -> FrameType: ...
    def has_complex_signature(self, a: Any, b: Any, *, c: Any, d: int = 0, *e: Any, f: Any, g: int = 0, **h: Any) -> FrameType: ...
    @property
    def a_property(self) -> FrameType: ...
    @property
    def a_settable_property(self) -> FrameType: ...
    @a_settable_property.setter
    def a_settable_property(self, unused: Any) -> FrameType: ...
    if cached_property:
        @cached_property
        def a_cached_property(self) -> FrameType: ...

class Outer:
    class Inner:
        def f(self) -> None: ...

def transform_path(path: str) -> str:
    """Transform tests/test_foo.py to monkeytype.foo"""
    path = 'monkeytype/' + path[len('tests/'):]
    *basepath, file_name = path.split('/')
    basename, _ext = os.path.splitext(file_name[len('test_'):])
    return '.'.join(basepath + [basename])

def smartcov_paths_hook(paths: List[str]) -> List[str]:
    """Given list of test files to run, return modules to measure coverage of."""
    if not paths:
        return ['monkeytype']
    return [transform_path(path) for path in paths]
