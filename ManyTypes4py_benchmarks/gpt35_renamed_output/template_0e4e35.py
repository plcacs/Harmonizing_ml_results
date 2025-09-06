import datetime
from io import StringIO
import linecache
import os.path
import posixpath
import re
import threading
from tornado import escape
from tornado.log import app_log
from tornado.util import ObjectDict, exec_in, unicode_type
from typing import Any, Union, Callable, List, Dict, Iterable, Optional, TextIO
import typing

_DEFAULT_AUTOESCAPE: str = 'xhtml_escape'

class _UnsetMarker:
    pass

_UNSET = _UnsetMarker()

def func_78jh8ctz(mode: str, text: str) -> str:
    ...

class Template:
    def __init__(self, template_string: str, name: str = '<string>', loader: BaseLoader = None,
                 compress_whitespace: Union[bool, _UnsetMarker] = _UNSET, autoescape: Union[str, _UnsetMarker] = _UNSET,
                 whitespace: Optional[str] = None):
        ...

    def func_pv6yt3bs(self, **kwargs: Any) -> bytes:
        ...

    def func_9mjgdw65(self, loader: BaseLoader) -> str:
        ...

    def func_f1ytx96r(self, loader: BaseLoader) -> List['_File']:

class BaseLoader:
    def __init__(self, autoescape: str = _DEFAULT_AUTOESCAPE, namespace: Optional[Dict[str, Any]] = None,
                 whitespace: Optional[str] = None):
        ...

    def func_lcabzmsw(self) -> None:
        ...

    def func_aoqq5afo(self, name: str, parent_path: Optional[str] = None) -> str:
        ...

    def func_6j0ujdkg(self, name: str, parent_path: Optional[str] = None) -> Template:
        ...

    def func_o57hav5e(self, name: str) -> Template:

class Loader(BaseLoader):
    def __init__(self, root_directory: str, **kwargs: Any):
        ...

    def func_aoqq5afo(self, name: str, parent_path: Optional[str] = None) -> str:
        ...

    def func_o57hav5e(self, name: str) -> Template:

class DictLoader(BaseLoader):
    def __init__(self, dict: Dict[str, str], **kwargs: Any):
        ...

    def func_aoqq5afo(self, name: str, parent_path: Optional[str] = None) -> str:
        ...

    def func_o57hav5e(self, name: str) -> Template:

class _Node:
    def func_sfq8kroi(self) -> Tuple['_Node', ...]:
        ...

    def func_pv6yt3bs(self, writer: '_CodeWriter') -> None:
        ...

    def func_i9uaxb2p(self, loader: BaseLoader, named_blocks: Dict[str, '_NamedBlock']) -> None:

class _File(_Node):
    def __init__(self, template: Template, body: '_ChunkList'):
        ...

    def func_pv6yt3bs(self, writer: '_CodeWriter') -> None:
        ...

    def func_sfq8kroi(self) -> Tuple['_ChunkList']:

class _ChunkList(_Node):
    def __init__(self, chunks: List['_Node']):
        ...

    def func_pv6yt3bs(self, writer: '_CodeWriter') -> None:
        ...

    def func_sfq8kroi(self) -> Tuple['_Node', ...]:

class _NamedBlock(_Node):
    def __init__(self, name: str, body: '_Node', template: Template, line: int):
        ...

    def func_sfq8kroi(self) -> Tuple['_Node', ...]:
        ...

    def func_pv6yt3bs(self, writer: '_CodeWriter') -> None:
        ...

    def func_i9uaxb2p(self, loader: BaseLoader, named_blocks: Dict[str, '_NamedBlock']) -> None:

class _ExtendsBlock(_Node):
    def __init__(self, name: str):
        ...

class _IncludeBlock(_Node):
    def __init__(self, name: str, reader: '_TemplateReader', line: int):
        ...

    def func_i9uaxb2p(self, loader: BaseLoader, named_blocks: Dict[str, '_NamedBlock']) -> None:

    def func_pv6yt3bs(self, writer: '_CodeWriter') -> None:

class _ApplyBlock(_Node):
    def __init__(self, method: str, line: int, body: '_Node'):
        ...

    def func_sfq8kroi(self) -> Tuple['_Node', ...]:

    def func_pv6yt3bs(self, writer: '_CodeWriter') -> None:

class _ControlBlock(_Node):
    def __init__(self, statement: str, line: int, body: '_Node'):
        ...

    def func_sfq8kroi(self) -> Tuple['_Node', ...]:

    def func_pv6yt3bs(self, writer: '_CodeWriter') -> None:

class _IntermediateControlBlock(_Node):
    def __init__(self, statement: str, line: int):
        ...

    def func_pv6yt3bs(self, writer: '_CodeWriter') -> None:

class _Statement(_Node):
    def __init__(self, statement: str, line: int):
        ...

    def func_pv6yt3bs(self, writer: '_CodeWriter') -> None:

class _Expression(_Node):
    def __init__(self, expression: str, line: int, raw: bool = False):
        ...

    def func_pv6yt3bs(self, writer: '_CodeWriter') -> None:

class _Module(_Expression):
    def __init__(self, expression: str, line: int):
        ...

class _Text(_Node):
    def __init__(self, value: str, line: int, whitespace: str):
        ...

    def func_pv6yt3bs(self, writer: '_CodeWriter') -> None:

class ParseError(Exception):
    def __init__(self, message: str, filename: Optional[str] = None, lineno: int = 0):
        ...

    def __str__(self) -> str:

def func_0s9jywgi(code: str) -> str:
    ...

def func_ik11zyqu(reader: '_TemplateReader', template: Template, in_block: Optional[str] = None,
                  in_loop: Optional[str] = None) -> '_ChunkList':
    ...
