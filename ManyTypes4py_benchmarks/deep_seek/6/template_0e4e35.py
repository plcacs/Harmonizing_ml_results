"""A simple template system that compiles templates to Python code."""
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
from typing import Any, Union, Callable, List, Dict, Iterable, Optional, TextIO, Tuple, Set, TypeVar, cast
import typing
from types import ModuleType

if typing.TYPE_CHECKING:
    from typing import ContextManager, IO

_DEFAULT_AUTOESCAPE = 'xhtml_escape'

T = TypeVar('T')

class _UnsetMarker:
    pass
_UNSET = _UnsetMarker()

def filter_whitespace(mode: str, text: str) -> str:
    """Transform whitespace in ``text`` according to ``mode``."""
    if mode == 'all':
        return text
    elif mode == 'single':
        text = re.sub('([\\t ]+)', ' ', text)
        text = re.sub('(\\s*\\n\\s*)', '\n', text)
        return text
    elif mode == 'oneline':
        return re.sub('(\\s+)', ' ', text)
    else:
        raise Exception('invalid whitespace mode %s' % mode)

class Template:
    """A compiled template."""

    def __init__(
        self,
        template_string: Union[str, bytes],
        name: str = '<string>',
        loader: Optional['BaseLoader'] = None,
        compress_whitespace: Union[bool, _UnsetMarker] = _UNSET,
        autoescape: Union[str, None, _UnsetMarker] = _UNSET,
        whitespace: Optional[str] = None
    ) -> None:
        self.name = escape.native_str(name)
        if compress_whitespace is not _UNSET:
            if whitespace is not None:
                raise Exception('cannot set both whitespace and compress_whitespace')
            whitespace = 'single' if compress_whitespace else 'all'
        if whitespace is None:
            if loader and loader.whitespace:
                whitespace = loader.whitespace
            elif name.endswith('.html') or name.endswith('.js'):
                whitespace = 'single'
            else:
                whitespace = 'all'
        assert whitespace is not None
        filter_whitespace(whitespace, '')
        if not isinstance(autoescape, _UnsetMarker):
            self.autoescape = autoescape
        elif loader:
            self.autoescape = loader.autoescape
        else:
            self.autoescape = _DEFAULT_AUTOESCAPE
        self.namespace = loader.namespace if loader else {}
        reader = _TemplateReader(name, escape.native_str(template_string), whitespace)
        self.file = _File(self, _parse(reader, self))
        self.code = self._generate_python(loader)
        self.loader = loader
        try:
            self.compiled = compile(escape.to_unicode(self.code), '%s.generated.py' % self.name.replace('.', '_'), 'exec', dont_inherit=True)
        except Exception:
            formatted_code = _format_code(self.code).rstrip()
            app_log.error('%s code:\n%s', self.name, formatted_code)
            raise

    def generate(self, **kwargs: Any) -> bytes:
        """Generate this template with the given arguments."""
        namespace = {
            'escape': escape.xhtml_escape,
            'xhtml_escape': escape.xhtml_escape,
            'url_escape': escape.url_escape,
            'json_encode': escape.json_encode,
            'squeeze': escape.squeeze,
            'linkify': escape.linkify,
            'datetime': datetime,
            '_tt_utf8': escape.utf8,
            '_tt_string_types': (unicode_type, bytes),
            '__name__': self.name.replace('.', '_'),
            '__loader__': ObjectDict(get_source=lambda name: self.code)
        }
        namespace.update(self.namespace)
        namespace.update(kwargs)
        exec_in(self.compiled, namespace)
        execute = cast(Callable[[], bytes], namespace['_tt_execute'])
        linecache.clearcache()
        return execute()

    def _generate_python(self, loader: Optional['BaseLoader']) -> str:
        buffer = StringIO()
        try:
            named_blocks: Dict[str, _NamedBlock] = {}
            ancestors = self._get_ancestors(loader)
            ancestors.reverse()
            for ancestor in ancestors:
                ancestor.find_named_blocks(loader, named_blocks)
            writer = _CodeWriter(buffer, named_blocks, loader, ancestors[0].template)
            ancestors[0].generate(writer)
            return buffer.getvalue()
        finally:
            buffer.close()

    def _get_ancestors(self, loader: Optional['BaseLoader']) -> List['_File']:
        ancestors = [self.file]
        for chunk in self.file.body.chunks:
            if isinstance(chunk, _ExtendsBlock):
                if not loader:
                    raise ParseError('{% extends %} block found, but no template loader')
                template = loader.load(chunk.name, self.name)
                ancestors.extend(template._get_ancestors(loader))
        return ancestors

class BaseLoader:
    """Base class for template loaders."""

    def __init__(
        self,
        autoescape: Union[str, None] = _DEFAULT_AUTOESCAPE,
        namespace: Optional[Dict[str, Any]] = None,
        whitespace: Optional[str] = None
    ) -> None:
        self.autoescape = autoescape
        self.namespace = namespace or {}
        self.whitespace = whitespace
        self.templates: Dict[str, Template] = {}
        self.lock = threading.RLock()

    def reset(self) -> None:
        """Resets the cache of compiled templates."""
        with self.lock:
            self.templates = {}

    def resolve_path(self, name: str, parent_path: Optional[str] = None) -> str:
        """Converts a possibly-relative path to absolute (used internally)."""
        raise NotImplementedError()

    def load(self, name: str, parent_path: Optional[str] = None) -> Template:
        """Loads a template."""
        name = self.resolve_path(name, parent_path=parent_path)
        with self.lock:
            if name not in self.templates:
                self.templates[name] = self._create_template(name)
            return self.templates[name]

    def _create_template(self, name: str) -> Template:
        raise NotImplementedError()

class Loader(BaseLoader):
    """A template loader that loads from a single root directory."""

    def __init__(self, root_directory: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.root = os.path.abspath(root_directory)

    def resolve_path(self, name: str, parent_path: Optional[str] = None) -> str:
        if parent_path and (not parent_path.startswith('<')) and (not parent_path.startswith('/')) and (not name.startswith('/')):
            current_path = os.path.join(self.root, parent_path)
            file_dir = os.path.dirname(os.path.abspath(current_path))
            relative_path = os.path.abspath(os.path.join(file_dir, name))
            if relative_path.startswith(self.root):
                name = relative_path[len(self.root) + 1:]
        return name

    def _create_template(self, name: str) -> Template:
        path = os.path.join(self.root, name)
        with open(path, 'rb') as f:
            template = Template(f.read(), name=name, loader=self)
            return template

class DictLoader(BaseLoader):
    """A template loader that loads from a dictionary."""

    def __init__(self, dict: Dict[str, Union[str, bytes]], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.dict = dict

    def resolve_path(self, name: str, parent_path: Optional[str] = None) -> str:
        if parent_path and (not parent_path.startswith('<')) and (not parent_path.startswith('/')) and (not name.startswith('/')):
            file_dir = posixpath.dirname(parent_path)
            name = posixpath.normpath(posixpath.join(file_dir, name))
        return name

    def _create_template(self, name: str) -> Template:
        return Template(self.dict[name], name=name, loader=self)

class _Node:
    def each_child(self) -> Iterable['_Node']:
        return ()

    def generate(self, writer: '_CodeWriter') -> None:
        raise NotImplementedError()

    def find_named_blocks(self, loader: Optional[BaseLoader], named_blocks: Dict[str, '_NamedBlock']) -> None:
        for child in self.each_child():
            child.find_named_blocks(loader, named_blocks)

class _File(_Node):
    def __init__(self, template: Template, body: '_ChunkList') -> None:
        self.template = template
        self.body = body
        self.line = 0

    def generate(self, writer: '_CodeWriter') -> None:
        writer.write_line('def _tt_execute():', self.line)
        with writer.indent():
            writer.write_line('_tt_buffer = []', self.line)
            writer.write_line('_tt_append = _tt_buffer.append', self.line)
            self.body.generate(writer)
            writer.write_line("return _tt_utf8('').join(_tt_buffer)", self.line)

    def each_child(self) -> Tuple['_ChunkList']:
        return (self.body,)

class _ChunkList(_Node):
    def __init__(self, chunks: List['_Node']) -> None:
        self.chunks = chunks

    def generate(self, writer: '_CodeWriter') -> None:
        for chunk in self.chunks:
            chunk.generate(writer)

    def each_child(self) -> List['_Node']:
        return self.chunks

class _NamedBlock(_Node):
    def __init__(self, name: str, body: '_ChunkList', template: Template, line: int) -> None:
        self.name = name
        self.body = body
        self.template = template
        self.line = line

    def each_child(self) -> Tuple['_ChunkList']:
        return (self.body,)

    def generate(self, writer: '_CodeWriter') -> None:
        block = writer.named_blocks[self.name]
        with writer.include(block.template, self.line):
            block.body.generate(writer)

    def find_named_blocks(self, loader: Optional[BaseLoader], named_blocks: Dict[str, '_NamedBlock']) -> None:
        named_blocks[self.name] = self
        _Node.find_named_blocks(self, loader, named_blocks)

class _ExtendsBlock(_Node):
    def __init__(self, name: str) -> None:
        self.name = name

class _IncludeBlock(_Node):
    def __init__(self, name: str, reader: '_TemplateReader', line: int) -> None:
        self.name = name
        self.template_name = reader.name
        self.line = line

    def find_named_blocks(self, loader: Optional[BaseLoader], named_blocks: Dict[str, '_NamedBlock']) -> None:
        assert loader is not None
        included = loader.load(self.name, self.template_name)
        included.file.find_named_blocks(loader, named_blocks)

    def generate(self, writer: '_CodeWriter') -> None:
        assert writer.loader is not None
        included = writer.loader.load(self.name, self.template_name)
        with writer.include(included, self.line):
            included.file.body.generate(writer)

class _ApplyBlock(_Node):
    def __init__(self, method: str, line: int, body: '_ChunkList') -> None:
        self.method = method
        self.line = line
        self.body = body

    def each_child(self) -> Tuple['_ChunkList']:
        return (self.body,)

    def generate(self, writer: '_CodeWriter') -> None:
        method_name = '_tt_apply%d' % writer.apply_counter
        writer.apply_counter += 1
        writer.write_line('def %s():' % method_name, self.line)
        with writer.indent():
            writer.write_line('_tt_buffer = []', self.line)
            writer.write_line('_tt_append = _tt_buffer.append', self.line)
            self.body.generate(writer)
            writer.write_line("return _tt_utf8('').join(_tt_buffer)", self.line)
        writer.write_line(f'_tt_append(_tt_utf8({self.method}({method_name}())))', self.line)

class _ControlBlock(_Node):
    def __init__(self, statement: str, line: int, body: '_ChunkList') -> None:
        self.statement = statement
        self.line = line
        self.body = body

    def each_child(self) -> Tuple['_ChunkList']:
        return (self.body,)

    def generate(self, writer: '_CodeWriter') -> None:
        writer.write_line('%s:' % self.statement, self.line)
        with writer.indent():
            self.body.generate(writer)
            writer.write_line('pass', self.line)

class _IntermediateControlBlock(_Node):
    def __init__(self, statement: str, line: int) -> None:
        self.statement = statement
        self.line = line

    def generate(self, writer: '_CodeWriter') -> None:
        writer.write_line('pass', self.line)
        writer.write_line('%s:' % self.statement, self.line, writer.indent_size() - 1)

class _Statement(_Node):
    def __init__(self, statement: str, line: int) -> None:
        self.statement = statement
        self.line = line

    def generate(self, writer: '_CodeWriter') -> None:
        writer.write_line(self.statement, self.line)

class _Expression(_Node):
    def __init__(self, expression: str, line: int, raw: bool = False) -> None:
        self.expression = expression
        self.line = line
        self.raw = raw

    def generate(self, writer: '_CodeWriter') -> None:
        writer.write_line('_tt_tmp = %s' % self.expression, self.line)
        writer.write_line('if isinstance(_tt_tmp, _tt_string_types): _tt_tmp = _tt_utf8(_tt_tmp)', self.line)
        writer.write_line('else: _tt_tmp = _tt_utf8(str(_tt_tmp))', self.line)
        if not self.raw and writer.current_template.autoescape is not None:
            writer.write_line('_tt_tmp = _tt_utf8(%s(_tt_tmp))' % writer.current_template.autoescape, self.line)
        writer.write_line('_tt_append(_tt_tmp)', self.line)

class _Module(_Expression):
    def __init__(self, expression: str, line: int) -> None:
        super().__init__('_tt_modules.' + expression, line, raw=True)

class _Text(_Node):
    def __init__(self, value: str, line: int, whitespace: str) -> None:
        self.value = value
        self.line = line
        self.whitespace = whitespace

    def generate(self, writer: '_CodeWriter') -> None:
        value = self.value
        if '<pre>' not in value:
            value = filter_whitespace(self.whitespace, value)
        if value:
            writer.write_line('_tt_append(%r)' % escape.utf8(value), self.line)

class ParseError(Exception):
    def __init__(self, message: str, filename: Optional[str] = None, lineno: int = 0) -> None:
        self.message = message
        self.filename = filename
        self.lineno = lineno

    def __str__(self) -> str:
        return '%s at %s:%d' % (self.message, self.filename, self.lineno)

class _CodeWriter:
    def __init__(
        self,
        file: TextIO,
        named_blocks: Dict[str, _NamedBlock],
        loader: Optional[BaseLoader],
        current_template: Template
    ) -> None:
        self.file = file
        self.named_blocks = named_blocks
        self.loader = loader
        self.current_template = current_template
        self.apply_counter = 0
        self.include_stack: List[Tuple[Template, int]] = []
        self._indent = 0

    def indent_size(self) -> int:
        return self._indent

    def indent(self) -> 'ContextManager[_CodeWriter]':
        class Indenter:
            def __enter__(self_: Any) -> '_CodeWriter':
                self._indent += 1
                return self

            def __exit__(self_: Any, *args: Any) -> None:
                assert self._indent > 0
                self._indent -= 1
        return Indenter()

    def include(self, template: Template, line: int) -> 'ContextManager[_CodeWriter]':
        self.include_stack.append((self.current_template, line))
        self.current_template = template

        class IncludeTemplate:
            def __enter__(self_: Any) -> '_CodeWriter':
                return self

            def __exit__(self_: Any, *args: Any) -> None:
                self.current_template = self.include_stack.pop()[0]
        return IncludeTemplate()

    def write_line(self, line: str, line_number: int, indent: Optional[int] = None) -> None:
        if indent is None:
            indent = self._indent
        line_comment = '  # %s:%d' % (self.current_template.name, line_number)
        if self.include_stack:
            ancestors = ['%s:%d' % (tmpl.name, lineno) for tmpl, lineno in self.include_stack]
            line_comment += ' (via %s)' % ', '.join(reversed(ancestors))
        print('    ' * indent + line + line_comment, file=self.file)

class _TemplateReader:
    def __init__(self, name: str, text: str, whitespace: str) -> None:
        self.name = name
        self.text = text
        self.whitespace = whitespace
        self.line = 1
        self.pos = 0

    def find(self, needle: str, start: int = 0,