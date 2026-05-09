from typing import Any, Union, Callable, List, Dict, Iterable, Optional, TextIO
import threading
from tornado import escape, log, util
from tornado.template import _UnsetMarker, _TemplateReader, _CodeWriter, _ChunkList, _NamedBlock, _ExtendsBlock, _IncludeBlock, _ApplyBlock, _ControlBlock, _Expression, _Statement, _Text
from tornado.util import ObjectDict, exec_in, unicode_type
import re
import os.path
import posixpath
import datetime

class Template:
    """A compiled template."""
    def __init__(self, template_string: str, name: str, loader: 'tornado.template.BaseLoader', compress_whitespace: bool = _UnsetMarker, autoescape: str = _UnsetMarker, whitespace: str = None) -> None:
        ...

    def generate(self, **kwargs: Any) -> str:
        ...

    def _generate_python(self, loader: 'tornado.template.BaseLoader') -> str:
        ...

class BaseLoader:
    """Base class for template loaders."""
    def __init__(self, autoescape: str = _DEFAULT_AUTOESCAPE, namespace: dict = None, whitespace: str = None) -> None:
        ...

    def resolve_path(self, name: str, parent_path: str = None) -> str:
        ...

    def load(self, name: str, parent_path: str = None) -> 'Template':
        ...

class Loader(BaseLoader):
    """A template loader that loads from a single root directory."""
    def __init__(self, root_directory: str, **kwargs: Any) -> None:
        ...

    def resolve_path(self, name: str, parent_path: str = None) -> str:
        ...

    def _create_template(self, name: str) -> 'Template':
        ...

class DictLoader(BaseLoader):
    """A template loader that loads from a dictionary."""
    def __init__(self, dict: dict, **kwargs: Any) -> None:
        ...

    def resolve_path(self, name: str, parent_path: str = None) -> str:
        ...

    def _create_template(self, name: str) -> 'Template':
        ...

class _Node:
    """Base class for nodes in the abstract syntax tree."""
    def each_child(self) -> Iterable['_Node']:
        ...

    def generate(self, writer: '_CodeWriter') -> None:
        ...

    def find_named_blocks(self, loader: 'tornado.template.BaseLoader', named_blocks: Dict[str, '_NamedBlock']) -> None:
        ...

class _File(_Node):
    """A file node in the abstract syntax tree."""
    def __init__(self, template: 'Template', body: '_ChunkList') -> None:
        ...

    def generate(self, writer: '_CodeWriter') -> None:
        ...

    def each_child(self) -> Iterable['_Node']:
        ...

class _ChunkList(_Node):
    """A list of chunks in the abstract syntax tree."""
    def __init__(self, chunks: List['_Node']) -> None:
        ...

    def generate(self, writer: '_CodeWriter') -> None:
        ...

    def each_child(self) -> Iterable['_Node']:
        ...

class _NamedBlock(_Node):
    """A named block node in the abstract syntax tree."""
    def __init__(self, name: str, body: '_Node', template: 'Template', line: int) -> None:
        ...

    def each_child(self) -> Iterable['_Node']:
        ...

    def generate(self, writer: '_CodeWriter') -> None:
        ...

    def find_named_blocks(self, loader: 'tornado.template.BaseLoader', named_blocks: Dict[str, '_NamedBlock']) -> None:
        ...

class _ExtendsBlock(_Node):
    """An extends block node in the abstract syntax tree."""
    def __init__(self, name: str) -> None:
        ...

class _IncludeBlock(_Node):
    """An include block node in the abstract syntax tree."""
    def __init__(self, name: str, reader: '_TemplateReader', line: int) -> None:
        ...

    def find_named_blocks(self, loader: 'tornado.template.BaseLoader', named_blocks: Dict[str, '_NamedBlock']) -> None:
        ...

    def generate(self, writer: '_CodeWriter') -> None:
        ...

class _ApplyBlock(_Node):
    """An apply block node in the abstract syntax tree."""
    def __init__(self, method: str, line: int, body: '_Node') -> None:
        ...

    def each_child(self) -> Iterable['_Node']:
        ...

    def generate(self, writer: '_CodeWriter') -> None:
        ...

class _ControlBlock(_Node):
    """A control block node in the abstract syntax tree."""
    def __init__(self, statement: str, line: int, body: '_Node') -> None:
        ...

    def each_child(self) -> Iterable['_Node']:
        ...

    def generate(self, writer: '_CodeWriter') -> None:
        ...

class _Expression(_Node):
    """An expression node in the abstract syntax tree."""
    def __init__(self, expression: str, line: int, raw: bool = False) -> None:
        ...

    def generate(self, writer: '_CodeWriter') -> None:
        ...

class _Statement(_Node):
    """A statement node in the abstract syntax tree."""
    def __init__(self, statement: str, line: int) -> None:
        ...

    def generate(self, writer: '_CodeWriter') -> None:
        ...

class _Text(_Node):
    """A text node in the abstract syntax tree."""
    def __init__(self, value: str, line: int, whitespace: str) -> None:
        ...

    def generate(self, writer: '_CodeWriter') -> None:
        ...

class ParseError(Exception):
    """Raised for template syntax errors."""
    def __init__(self, message: str, filename: str = None, lineno: int = 0) -> None:
        ...
