from typing import Any, Union, Callable, List, Dict, Iterable, Optional, TextIO

_DEFAULT_AUTOESCAPE: str = 'xhtml_escape'

class _UnsetMarker:
    pass

_UNSET: _UnsetMarker = _UnsetMarker()

def filter_whitespace(mode: str, text: str) -> str:
    ...

class Template:
    def __init__(self, template_string: str, name: str = '<string>', loader: BaseLoader = None, compress_whitespace: Union[bool, _UnsetMarker] = _UNSET, autoescape: Union[str, _UnsetMarker] = _UNSET, whitespace: Optional[str] = None) -> None:
        ...

    def generate(self, **kwargs: Any) -> bytes:
        ...

    def _generate_python(self, loader: BaseLoader) -> str:
        ...

class BaseLoader:
    def __init__(self, autoescape: str = _DEFAULT_AUTOESCAPE, namespace: Optional[Dict[str, Any]] = None, whitespace: Optional[str] = None) -> None:
        ...

    def reset(self) -> None:
        ...

    def resolve_path(self, name: str, parent_path: Optional[str] = None) -> str:
        ...

    def load(self, name: str, parent_path: Optional[str] = None) -> Template:
        ...

    def _create_template(self, name: str) -> Template:
        ...

class Loader(BaseLoader):
    def __init__(self, root_directory: str, **kwargs: Any) -> None:
        ...

    def resolve_path(self, name: str, parent_path: Optional[str] = None) -> str:
        ...

    def _create_template(self, name: str) -> Template:
        ...

class DictLoader(BaseLoader):
    def __init__(self, dict: Dict[str, str], **kwargs: Any) -> None:
        ...

    def resolve_path(self, name: str, parent_path: Optional[str] = None) -> str:
        ...

    def _create_template(self, name: str) -> Template:
        ...

class _Node:
    ...

class _File(_Node):
    ...

class _ChunkList(_Node):
    ...

class _NamedBlock(_Node):
    ...

class _ExtendsBlock(_Node):
    ...

class _IncludeBlock(_Node):
    ...

class _ApplyBlock(_Node):
    ...

class _ControlBlock(_Node):
    ...

class _IntermediateControlBlock(_Node):
    ...

class _Statement(_Node):
    ...

class _Expression(_Node):
    ...

class _Module(_Expression):
    ...

class _Text(_Node):
    ...

class ParseError(Exception):
    ...

def _format_code(code: str) -> str:
    ...

def _parse(reader: _TemplateReader, template: Template, in_block: Optional[str] = None, in_loop: Optional[str] = None) -> _ChunkList:
    ...

class _CodeWriter:
    ...

class _TemplateReader:
    ...

def add_appropriate_type_annotations():
    ...
