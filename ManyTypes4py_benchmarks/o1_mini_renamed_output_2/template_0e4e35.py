"""A simple template system that compiles templates to Python code.

Basic usage looks like::

    t = template.Template("<html>{{ myvalue }}</html>")
    print(t.generate(myvalue="XXX"))

`Loader` is a class that loads templates from a root directory and caches
the compiled templates::

    loader = template.Loader("/home/btaylor")
    print(loader.load("test.html").generate(myvalue="XXX"))

We compile all templates to raw Python. Error-reporting is currently... uh,
interesting. Syntax for the templates::

    ### base.html
    <html>
      <head>
        <title>{% block title %}Default title{% end %}</title>
      </head>
      <body>
        <ul>
          {% for student in students %}
            {% block student %}
              <li>{{ escape(student.name) }}</li>
            {% end %}
          {% end %}
        </ul>
      </body>
    </html>

    ### bold.html
    {% extends "base.html" %}

    {% block title %}A bolder title{% end %}

    {% block student %}
      <li><span style="bold">{{ escape(student.name) }}</span></li>
    {% end %}

Unlike most other template systems, we do not put any restrictions on the
expressions you can include in your statements. ``if`` and ``for`` blocks get
translated exactly into Python, so you can do complex expressions like::

   {% for student in [p for p in people if p.student and p.age > 23] %}
     <li>{{ escape(student.name) }}</li>
   {% end %}

Translating directly to Python means you can apply functions to expressions
easily, like the ``escape()`` function in the examples above. You can pass
functions in to your template just like any other variable
(In a `.RequestHandler`, override `.RequestHandler.get_template_namespace`)::
   
   ### Python code
   def add(x, y):
      return x + y
   template.execute(add=add)

   ### The template
   {{ add(1, 2) }}

We provide the functions `escape() <.xhtml_escape>`, `.url_escape()`,
`.json_encode()`, and `.squeeze()` to all templates by default.

Typical applications do not create `Template` or `Loader` instances by
hand, but instead use the `~.RequestHandler.render` and
`~.RequestHandler.render_string` methods of
`tornado.web.RequestHandler`, which load templates automatically based
on the ``template_path`` `.Application` setting.

Variable names beginning with ``_tt_`` are reserved by the template
system and should not be used by application code.

Syntax Reference
----------------

Template expressions are surrounded by double curly braces: ``{{ ... }}``.
The contents may be any python expression, which will be escaped according
to the current autoescape setting and inserted into the output.  Other
template directives use ``{% %}``.

To comment out a section so that it is omitted from the output, surround it
with ``{# ... #}``.

To include a literal ``{{``, ``{%``, or ``{#`` in the output, escape them as
``{{!``, ``{%!``, and ``{#!``, respectively.

``{% apply *function* %}...{% end %}``
    Applies a function to the output of all template code between ``apply``
    and ``end``::

        {% apply linkify %}{{name}} said: {{message}}{% end %}

    Note that as an implementation detail apply blocks are implemented
    as nested functions and thus may interact strangely with variables
    set via ``{% set %}``, or the use of ``{% break %}`` or ``{% continue %}``
    within loops.

``{% autoescape *function* %}``
    Sets the autoescape mode for the current file.  This does not affect
    other files, even those referenced by ``{% include %}``.  Note that
    autoescaping can also be configured globally, at the `.Application`
    or `Loader`.::
    
        {% autoescape xhtml_escape %}
        {% autoescape None %}

``{% block *name* %}...{% end %}``
    Indicates a named, replaceable block for use with ``{% extends %}``.
    Blocks in the parent template will be replaced with the contents of
    the same-named block in a child template.::
    
        <!-- base.html -->
        <title>{% block title %}Default title{% end %}</title>

        <!-- mypage.html -->
        {% extends "base.html" %}
        {% block title %}My page title{% end %}

``{% comment ... %}``
    A comment which will be removed from the template output.  Note that
    there is no ``{% end %}`` tag; the comment goes from the word ``comment``
    to the closing ``%}`` tag.

``{% extends *filename* %}``
    Inherit from another template.  Templates that use ``extends`` should
    contain one or more ``block`` tags to replace content from the parent
    template.  Anything in the child template not contained in a ``block``
    tag will be ignored.  For an example, see the ``{% block %}`` tag.

``{% for *var* in *expr* %}...{% end %}``
    Same as the python ``for`` statement.  ``{% break %}`` and
    ``{% continue %}`` may be used inside the loop.

``{% from *x* import *y* %}``
    Same as the python ``import`` statement.

``{% if *condition* %}...{% elif *condition* %}...{% else %}...{% end %}``
    Conditional statement - outputs the first section whose condition is
    true.  (The ``elif`` and ``else`` sections are optional)

``{% import *module* %}``
    Same as the python ``import`` statement.

``{% include *filename* %}``
    Includes another template file.  The included file can see all the local
    variables as if it were copied directly to the point of the ``include``
    directive (the ``{% autoescape %}`` directive is an exception).
    Alternately, ``{% module Template(filename, **kwargs) %}`` may be used
    to include another template with an isolated namespace.

``{% module *expr* %}``
    Renders a `~tornado.web.UIModule`.  The output of the ``UIModule`` is
    not escaped::

        {% module Template("foo.html", arg=42) %}

    ``UIModules`` are a feature of the `tornado.web.RequestHandler`
    class (and specifically its ``render`` method) and will not work
    when the template system is used on its own in other contexts.

``{% raw *expr* %}``
    Outputs the result of the given expression without autoescaping.

``{% set *x* = *y* %}``
    Sets a local variable.

``{% try %}...{% except %}...{% else %}...{% finally %}...{% end %}``
    Same as the python ``try`` statement.

``{% while *condition* %}... {% end %}``
    Same as the python ``while`` statement.  ``{% break %}`` and
    ``{% continue %}`` may be used inside the loop.

``{% whitespace *mode* %}``
    Sets the whitespace mode for the remainder of the current file
    (or until the next ``{% whitespace %}`` directive). See
    `filter_whitespace` for available options. New in Tornado 4.3.
"""
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
from typing import Any, Union, Callable, List, Dict, Iterable, Optional, TextIO, TYPE_CHECKING
import typing

if TYPE_CHECKING:
    from typing import Tuple, ContextManager

_DEFAULT_AUTOESCAPE: str = 'xhtml_escape'


class _UnsetMarker:
    pass


_UNSET: _UnsetMarker = _UnsetMarker()


def func_78jh8ctz(mode: str, text: str) -> str:
    """Transform whitespace in ``text`` according to ``mode``.

    Available modes are:

    * ``all``: Return all whitespace unmodified.
    * ``single``: Collapse consecutive whitespace with a single whitespace
      character, preserving newlines.
    * ``oneline``: Collapse all runs of whitespace into a single space
      character, removing all newlines in the process.

    .. versionadded:: 4.3
    """
    if mode == 'all':
        return text
    elif mode == 'single':
        text = re.sub(r'([\t ]+)', ' ', text)
        text = re.sub(r'(\s*\n\s*)', '\n', text)
        return text
    elif mode == 'oneline':
        return re.sub(r'(\s+)', ' ', text)
    else:
        raise Exception(f'invalid whitespace mode {mode}')


class Template:
    """A compiled template.

    We compile into Python from the given template_string. You can generate
    the template from variables with generate().
    """

    def __init__(
        self,
        template_string: str,
        name: str = '<string>',
        loader: Optional['BaseLoader'] = None,
        compress_whitespace: Any = _UNSET,
        autoescape: Any = _UNSET,
        whitespace: Optional[str] = None
    ) -> None:
        """Construct a Template.

        :arg str template_string: the contents of the template file.
        :arg str name: the filename from which the template was loaded
            (used for error message).
        :arg tornado.template.BaseLoader loader: the `~tornado.template.BaseLoader` responsible
            for this template, used to resolve ``{% include %}`` and ``{% extend %}`` directives.
        :arg bool compress_whitespace: Deprecated since Tornado 4.3.
            Equivalent to ``whitespace="single"`` if true and
            ``whitespace="all"`` if false.
        :arg str autoescape: The name of a function in the template
            namespace, or ``None`` to disable escaping by default.
        :arg str whitespace: A string specifying treatment of whitespace;
            see `filter_whitespace` for options.

        .. versionchanged:: 4.3
           Added ``whitespace`` parameter; deprecated ``compress_whitespace``.
        """
        self.name: str = escape.native_str(name)
        if compress_whitespace is not _UNSET:
            if whitespace is not None:
                raise Exception(
                    'cannot set both whitespace and compress_whitespace')
            whitespace = 'single' if compress_whitespace else 'all'
        if whitespace is None:
            if loader and loader.whitespace:
                whitespace = loader.whitespace
            elif name.endswith('.html') or name.endswith('.js'):
                whitespace = 'single'
            else:
                whitespace = 'all'
        assert whitespace is not None
        func_78jh8ctz(whitespace, '')
        if not isinstance(autoescape, _UnsetMarker):
            self.autoescape: Optional[str] = autoescape
        elif loader:
            self.autoescape = loader.autoescape
        else:
            self.autoescape = _DEFAULT_AUTOESCAPE
        self.namespace: Dict[str, Any] = loader.namespace if loader else {}
        reader: '_TemplateReader' = _TemplateReader(
            name, escape.native_str(template_string), whitespace)
        self.file: '_File' = _File(self, _parse(reader, self))
        self.code: str = self._generate_python(loader)
        self.loader: Optional['BaseLoader'] = loader
        try:
            self.compiled = compile(
                escape.to_unicode(self.code),
                f'{self.name}.generated.py',
                'exec',
                dont_inherit=True
            )
        except Exception:
            formatted_code = _format_code(self.code).rstrip()
            app_log.error('%s code:\n%s', self.name, formatted_code)
            raise

    def func_pv6yt3bs(self, **kwargs: Any) -> bytes:
        """Generate this template with the given arguments."""
        namespace: Dict[str, Any] = {
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
        execute: Callable[[], bytes] = typing.cast(Callable[[], bytes], namespace['_tt_execute'])
        linecache.clearcache()
        return execute()

    def func_9mjgdw65(self, loader: 'BaseLoader') -> str:
        buffer: StringIO = StringIO()
        try:
            named_blocks: Dict[str, '_NamedBlock'] = {}
            ancestors: List['_File'] = self._get_ancestors(loader)
            ancestors.reverse()
            for ancestor in ancestors:
                ancestor.find_named_blocks(loader, named_blocks)
            writer: '_CodeWriter' = _CodeWriter(buffer, named_blocks, loader, ancestors[0].template)
            ancestors[0].generate(writer)
            return buffer.getvalue()
        finally:
            buffer.close()

    def func_f1ytx96r(self, loader: Optional['BaseLoader']) -> List['_File']:
        ancestors: List['_File'] = [self.file]
        for chunk in self.file.body.chunks:
            if isinstance(chunk, _ExtendsBlock):
                if not loader:
                    raise ParseError(
                        '{% extends %} block found, but no template loader')
                template: 'Template' = loader.load(chunk.name, self.name)
                ancestors.extend(template._get_ancestors(loader))
        return ancestors

    def _generate_python(self, loader: Optional['BaseLoader']) -> str:
        # Implementation not shown in the snippet
        pass

    def _get_ancestors(self, loader: 'BaseLoader') -> List['_File']:
        # Implementation not shown in the snippet
        pass


class BaseLoader:
    """Base class for template loaders.

    You must use a template loader to use template constructs like
    ``{% extends %}`` and ``{% include %}``. The loader caches all
    templates after they are loaded the first time.
    """

    def __init__(
        self,
        autoescape: Optional[str] = _DEFAULT_AUTOESCAPE,
        namespace: Optional[Dict[str, Any]] = None,
        whitespace: Optional[str] = None
    ) -> None:
        """Construct a template loader.

        :arg str autoescape: The name of a function in the template
            namespace, such as "xhtml_escape", or ``None`` to disable
            autoescaping by default.
        :arg dict namespace: A dictionary to be added to the default template
            namespace, or ``None``.
        :arg str whitespace: A string specifying default behavior for
            whitespace in templates; see `filter_whitespace` for options.
            Default is "single" for files ending in ".html" and ".js" and
            "all" for other files.

        .. versionchanged:: 4.3
           Added ``whitespace`` parameter.
        """
        self.autoescape: Optional[str] = autoescape
        self.namespace: Dict[str, Any] = namespace or {}
        self.whitespace: Optional[str] = whitespace
        self.templates: Dict[str, 'Template'] = {}
        self.lock: threading.RLock = threading.RLock()

    def func_lcabzmsw(self) -> None:
        """Resets the cache of compiled templates."""
        with self.lock:
            self.templates = {}

    def resolve_path(self, name: str, parent_path: Optional[str] = None) -> str:
        """Converts a possibly-relative path to absolute (used internally)."""
        raise NotImplementedError()

    def func_6j0ujdkg(self, name: str, parent_path: Optional[str] = None) -> 'Template':
        """Loads a template."""
        name = self.resolve_path(name, parent_path=parent_path)
        with self.lock:
            if name not in self.templates:
                self.templates[name] = self._create_template(name)
            return self.templates[name]

    def func_o57hav5e(self, name: str) -> 'Template':
        raise NotImplementedError()

    def _create_template(self, name: str) -> 'Template':
        # Implementation not shown in snippet
        pass


class Loader(BaseLoader):
    """A template loader that loads from a single root directory."""

    def __init__(self, root_directory: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.root: str = os.path.abspath(root_directory)

    def resolve_path(self, name: str, parent_path: Optional[str] = None) -> str:
        if parent_path and not parent_path.startswith('<') and not parent_path.startswith('/') and not name.startswith('/'):
            current_path: str = os.path.join(self.root, parent_path)
            file_dir: str = os.path.dirname(os.path.abspath(current_path))
            relative_path: str = os.path.abspath(os.path.join(file_dir, name))
            if relative_path.startswith(self.root):
                name = relative_path[len(self.root) + 1:]
        return name

    def func_o57hav5e(self, name: str) -> 'Template':
        path: str = os.path.join(self.root, name)
        with open(path, 'rb') as f:
            template: 'Template' = Template(f.read(), name=name, loader=self)
            return template


class DictLoader(BaseLoader):
    """A template loader that loads from a dictionary."""

    def __init__(self, dict_: Dict[str, str], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.dict: Dict[str, str] = dict_

    def resolve_path(self, name: str, parent_path: Optional[str] = None) -> str:
        if parent_path and not parent_path.startswith('<') and not parent_path.startswith('/') and not name.startswith('/'):
            file_dir: str = posixpath.dirname(parent_path)
            name = posixpath.normpath(posixpath.join(file_dir, name))
        return name

    def func_o57hav5e(self, name: str) -> 'Template':
        return Template(self.dict[name], name=name, loader=self)


class _Node:

    def func_sfq8kroi(self) -> Tuple['_Node', ...]:
        return ()

    def func_pv6yt3bs(self, writer: '_CodeWriter') -> None:
        raise NotImplementedError()

    def func_i9uaxb2p(self, loader: Optional['BaseLoader'], named_blocks: Dict[str, '_NamedBlock']) -> None:
        for child in self.each_child():
            child.find_named_blocks(loader, named_blocks)

    def each_child(self) -> Iterable['_Node']:
        return ()  # Default implementation


class _File(_Node):

    def __init__(self, template: 'Template', body: '_ChunkList') -> None:
        self.template: 'Template' = template
        self.body: '_ChunkList' = body
        self.line: int = 0

    def func_pv6yt3bs(self, writer: '_CodeWriter') -> None:
        writer.write_line('def _tt_execute():', self.line)
        with writer.indent():
            writer.write_line('_tt_buffer = []', self.line)
            writer.write_line('_tt_append = _tt_buffer.append', self.line)
            self.body.generate(writer)
            writer.write_line("return _tt_utf8('').join(_tt_buffer)", self.line)

    def func_sfq8kroi(self) -> Tuple['_ChunkList', ...]:
        return (self.body,)


class _ChunkList(_Node):

    def __init__(self, chunks: List['_Node']) -> None:
        self.chunks: List['_Node'] = chunks

    def func_pv6yt3bs(self, writer: '_CodeWriter') -> None:
        for chunk in self.chunks:
            chunk.generate(writer)

    def func_sfq8kroi(self) -> Tuple['_Node', ...]:
        return tuple(self.chunks)


class _NamedBlock(_Node):

    def __init__(self, name: str, body: '_ChunkList', template: 'Template', line: int) -> None:
        self.name: str = name
        self.body: '_ChunkList' = body
        self.template: 'Template' = template
        self.line: int = line

    def func_sfq8kroi(self) -> Tuple['_ChunkList', ...]:
        return (self.body,)

    def func_pv6yt3bs(self, writer: '_CodeWriter') -> None:
        block: '_NamedBlock' = writer.named_blocks[self.name]
        with writer.include(block.template, self.line):
            block.body.generate(writer)

    def func_i9uaxb2p(self, loader: Optional['BaseLoader'], named_blocks: Dict[str, '_NamedBlock']) -> None:
        named_blocks[self.name] = self
        _Node.func_i9uaxb2p(self, loader, named_blocks)


class _ExtendsBlock(_Node):

    def __init__(self, name: str) -> None:
        self.name: str = name


class _IncludeBlock(_Node):

    def __init__(self, name: str, reader: '_TemplateReader', line: int) -> None:
        self.name: str = name
        self.template_name: str = reader.name
        self.line: int = line

    def func_i9uaxb2p(self, loader: Optional['BaseLoader'], named_blocks: Dict[str, '_NamedBlock']) -> None:
        assert loader is not None
        included: 'Template' = loader.load(self.name, self.template_name)
        included.file.find_named_blocks(loader, named_blocks)

    def func_pv6yt3bs(self, writer: '_CodeWriter') -> None:
        assert writer.loader is not None
        included: 'Template' = writer.loader.load(self.name, self.template_name)
        with writer.include(included, self.line):
            included.file.body.generate(writer)


class _ApplyBlock(_Node):

    def __init__(self, method: str, line: int, body: '_ChunkList') -> None:
        self.method: str = method
        self.line: int = line
        self.body: '_ChunkList' = body

    def func_sfq8kroi(self) -> Tuple['_ChunkList', ...]:
        return (self.body,)

    def func_pv6yt3bs(self, writer: '_CodeWriter') -> None:
        method_name: str = f'_tt_apply{writer.apply_counter}'
        writer.apply_counter += 1
        writer.write_line(f'def {method_name}():', self.line)
        with writer.indent():
            writer.write_line('_tt_buffer = []', self.line)
            writer.write_line('_tt_append = _tt_buffer.append', self.line)
            self.body.generate(writer)
            writer.write_line("return _tt_utf8('').join(_tt_buffer)", self.line)
        writer.write_line(
            f'_tt_append(_tt_utf8({self.method}({method_name}())))', self.line
        )


class _ControlBlock(_Node):

    def __init__(self, statement: str, line: int, body: '_ChunkList') -> None:
        self.statement: str = statement
        self.line: int = line
        self.body: '_ChunkList' = body

    def func_sfq8kroi(self) -> Tuple['_ChunkList', ...]:
        return (self.body,)

    def func_pv6yt3bs(self, writer: '_CodeWriter') -> None:
        writer.write_line(f'{self.statement}:', self.line)
        with writer.indent():
            self.body.generate(writer)
            writer.write_line('pass', self.line)


class _IntermediateControlBlock(_Node):

    def __init__(self, statement: str, line: int) -> None:
        self.statement: str = statement
        self.line: int = line

    def func_pv6yt3bs(self, writer: '_CodeWriter') -> None:
        writer.write_line('pass', self.line)
        writer.write_line(f'{self.statement}:', self.line, writer.indent_size() - 1)


class _Statement(_Node):

    def __init__(self, statement: str, line: int) -> None:
        self.statement: str = statement
        self.line: int = line

    def func_pv6yt3bs(self, writer: '_CodeWriter') -> None:
        writer.write_line(self.statement, self.line)


class _Expression(_Node):

    def __init__(self, expression: str, line: int, raw: bool = False) -> None:
        self.expression: str = expression
        self.line: int = line
        self.raw: bool = raw

    def func_pv6yt3bs(self, writer: '_CodeWriter') -> None:
        writer.write_line(f'_tt_tmp = {self.expression}', self.line)
        writer.write_line(
            'if isinstance(_tt_tmp, _tt_string_types): _tt_tmp = _tt_utf8(_tt_tmp)',
            self.line
        )
        writer.write_line('else: _tt_tmp = _tt_utf8(str(_tt_tmp))', self.line)
        if not self.raw and writer.current_template.autoescape is not None:
            writer.write_line(f'_tt_tmp = _tt_utf8({writer.current_template.autoescape}(_tt_tmp))', self.line)
        writer.write_line('_tt_append(_tt_tmp)', self.line)


class _Module(_Expression):

    def __init__(self, expression: str, line: int) -> None:
        super().__init__('_tt_modules.' + expression, line, raw=True)


class _Text(_Node):

    def __init__(self, value: str, line: int, whitespace: str) -> None:
        self.value: str = value
        self.line: int = line
        self.whitespace: str = whitespace

    def func_pv6yt3bs(self, writer: '_CodeWriter') -> None:
        value: str = self.value
        if '<pre>' not in value:
            value = func_78jh8ctz(self.whitespace, value)
        if value:
            writer.write_line(f'_tt_append({repr(escape.utf8(value))})', self.line)


class ParseError(Exception):
    """Raised for template syntax errors.

    ``ParseError`` instances have ``filename`` and ``lineno`` attributes
    indicating the position of the error.

    .. versionchanged:: 4.3
       Added ``filename`` and ``lineno`` attributes.
    """

    def __init__(self, message: str, filename: Optional[str] = None, lineno: int = 0) -> None:
        self.message: str = message
        self.filename: Optional[str] = filename
        self.lineno: int = lineno

    def __str__(self) -> str:
        return f'{self.message} at {self.filename}:{self.lineno}'


class _CodeWriter:

    def __init__(
        self,
        file: StringIO,
        named_blocks: Dict[str, '_NamedBlock'],
        loader: 'BaseLoader',
        current_template: 'Template'
    ) -> None:
        self.file: StringIO = file
        self.named_blocks: Dict[str, '_NamedBlock'] = named_blocks
        self.loader: 'BaseLoader' = loader
        self.current_template: 'Template' = current_template
        self.apply_counter: int = 0
        self.include_stack: List[Tuple['Template', int]] = []
        self._indent: int = 0

    def indent_size(self) -> int:
        return self._indent

    def indent(self) -> 'ContextManager':
        class Indenter:
            def __enter__(_self) -> 'ContextManager':
                self._indent += 1
                return self

            def __exit__(_self, *args: Any) -> None:
                assert self._indent > 0
                self._indent -= 1
        return Indenter()

    def include(self, template: 'Template', line: int) -> 'ContextManager':
        self.include_stack.append((self.current_template, line))
        self.current_template = template

        class IncludeTemplate:
            def __enter__(_self) -> 'ContextManager':
                return self

            def __exit__(_self, *args: Any) -> None:
                self.current_template = self.include_stack.pop()[0]
        return IncludeTemplate()

    def write_line(self, line: str, line_number: int, indent: Optional[int] = None) -> None:
        if indent is None:
            indent_level: int = self._indent
        else:
            indent_level = indent
        line_comment: str = f'  # {self.current_template.name}:{line_number}'
        if self.include_stack:
            ancestors = [f'{tmpl.name}:{lineno}' for tmpl, lineno in
                        reversed(self.include_stack)]
            line_comment += f' (via {", ".join(ancestors)})'
        print('    ' * indent_level + line + line_comment, file=self.file)


class _TemplateReader:

    def __init__(self, name: str, text: str, whitespace: str) -> None:
        self.name: str = name
        self.text: str = text
        self.whitespace: str = whitespace
        self.line: int = 1
        self.pos: int = 0

    def find(self, needle: str, start: int = 0, end: Optional[int] = None) -> int:
        """Find the position of 'needle' in the text."""
        assert start >= 0, start
        pos: int = self.pos
        start += pos
        if end is None:
            index: int = self.text.find(needle, start)
        else:
            end += pos
            assert end >= start
            index = self.text.find(needle, start, end)
        if index != -1:
            index -= pos
        return index

    def consume(self, count: Optional[int] = None) -> str:
        """Consume 'count' characters from the current position."""
        if count is None:
            count = len(self) - self.pos
        newpos: int = self.pos + count
        self.line += self.text.count('\n', self.pos, newpos)
        s: str = self.text[self.pos:newpos]
        self.pos = newpos
        return s

    def remaining(self) -> int:
        return len(self.text) - self.pos

    def __len__(self) -> int:
        return self.remaining()

    def __getitem__(self, key: Union[int, slice]) -> str:
        if isinstance(key, slice):
            size = len(self)
            start, stop, step = key.indices(size)
            if start is None:
                start = self.pos
            else:
                start += self.pos
            if stop is not None:
                stop += self.pos
            return self.text[slice(start, stop, step)]
        elif key < 0:
            return self.text[key]
        else:
            return self.text[self.pos + key]

    def __str__(self) -> str:
        return self.text[self.pos:]

    def raise_parse_error(self, msg: str) -> None:
        raise ParseError(msg, self.name, self.line)


def func_0s9jywgi(code: str) -> str:
    """Format code with line numbers."""
    lines = code.splitlines()
    format_str = f'%%%dd  %%s\n' % len(repr(len(lines) + 1))
    return ''.join([(format_str % (i + 1, line)) for i, line in enumerate(lines)])


def func_ik11zyqu(
    reader: '_TemplateReader',
    template: 'Template',
    in_block: Optional[str] = None,
    in_loop: Optional[str] = None
) -> '_ChunkList':
    body: '_ChunkList' = _ChunkList([])
    while True:
        curly: int = 0
        while True:
            curly = reader.find('{', curly)
            if curly == -1 or curly + 1 == reader.remaining():
                if in_block:
                    reader.raise_parse_error(
                        f'Missing {{%% end %%}} block for {in_block}')
                body.chunks.append(_Text(reader.consume(), reader.line, reader.whitespace))
                return body
            if reader[curly + 1] not in ('{', '%', '#'):
                curly += 1
                continue
            if curly + 2 < reader.remaining() and reader[curly + 1] == '{' and reader[curly + 2] == '{':
                curly += 1
                continue
            break
        if curly > 0:
            cons: str = reader.consume(curly)
            body.chunks.append(_Text(cons, reader.line, reader.whitespace))
        start_brace: str = reader.consume(2)
        line: int = reader.line
        if reader.remaining() and reader[0] == '!':
            reader.consume(1)
            body.chunks.append(_Text(start_brace, line, reader.whitespace))
            continue
        if start_brace == '{#':
            end: int = reader.find('#}')
            if end == -1:
                reader.raise_parse_error('Missing end comment #}')
            contents: str = reader.consume(end).strip()
            reader.consume(2)
            continue
        if start_brace == '{{':
            end = reader.find('}}')
            if end == -1:
                reader.raise_parse_error('Missing end expression }}')
            contents: str = reader.consume(end).strip()
            reader.consume(2)
            if not contents:
                reader.raise_parse_error('Empty expression')
            body.chunks.append(_Expression(contents, line))
            continue
        assert start_brace == '{%', start_brace
        end = reader.find('%}')
        if end == -1:
            reader.raise_parse_error('Missing end block %}')
        contents = reader.consume(end).strip()
        reader.consume(2)
        if not contents:
            reader.raise_parse_error('Empty block tag ({% %})')
        operator, space, suffix = contents.partition(' ')
        suffix = suffix.strip()
        intermediate_blocks: Dict[str, set] = {
            'else': {'if', 'for', 'while', 'try'},
            'elif': {'if'},
            'except': {'try'},
            'finally': {'try'}
        }
        allowed_parents = intermediate_blocks.get(operator)
        if allowed_parents is not None:
            if not in_block:
                reader.raise_parse_error(
                    f'{operator} outside {allowed_parents} block')
            if in_block not in allowed_parents:
                reader.raise_parse_error(
                    f'{operator} block cannot be attached to {in_block} block')
            body.chunks.append(_IntermediateControlBlock(contents, line))
            continue
        elif operator == 'end':
            if not in_block:
                reader.raise_parse_error('Extra {% end %} block')
            return body
        elif operator in ('extends', 'include', 'set', 'import', 'from',
                          'comment', 'autoescape', 'whitespace', 'raw', 'module'):
            if operator == 'comment':
                continue
            if operator == 'extends':
                suffix = suffix.strip('"').strip("'")
                if not suffix:
                    reader.raise_parse_error('extends missing file path')
                block = _ExtendsBlock(suffix)
            elif operator in ('import', 'from'):
                if not suffix:
                    reader.raise_parse_error('import missing statement')
                block = _Statement(contents, line)
            elif operator == 'include':
                suffix = suffix.strip('"').strip("'")
                if not suffix:
                    reader.raise_parse_error('include missing file path')
                block = _IncludeBlock(suffix, reader, line)
            elif operator == 'set':
                if not suffix:
                    reader.raise_parse_error('set missing statement')
                block = _Statement(suffix, line)
            elif operator == 'autoescape':
                fn = suffix.strip()
                if fn == 'None':
                    fn = None
                template.autoescape = fn
                continue
            elif operator == 'whitespace':
                mode = suffix.strip()
                func_78jh8ctz(mode, '')
                reader.whitespace = mode
                continue
            elif operator == 'raw':
                block = _Expression(suffix, line, raw=True)
            elif operator == 'module':
                block = _Module(suffix, line)
            body.chunks.append(block)
            continue
        elif operator in ('apply', 'block', 'try', 'if', 'for', 'while'):
            if operator in ('for', 'while'):
                block_body = func_ik11zyqu(reader, template, operator, operator)
            elif operator == 'apply':
                block_body = func_ik11zyqu(reader, template, operator, None)
            else:
                block_body = func_ik11zyqu(reader, template, operator, in_loop)
            if operator == 'apply':
                if not suffix:
                    reader.raise_parse_error('apply missing method name')
                block = _ApplyBlock(suffix, line, block_body)
            elif operator == 'block':
                if not suffix:
                    reader.raise_parse_error('block missing name')
                block = _NamedBlock(suffix, block_body, template, line)
            else:
                block = _ControlBlock(contents, line, block_body)
            body.chunks.append(block)
            continue
        elif operator in ('break', 'continue'):
            if not in_loop:
                reader.raise_parse_error(f'{operator} outside {" , ".join({"for", "while"})} block')
            body.chunks.append(_Statement(contents, line))
            continue
        else:
            reader.raise_parse_error(f'unknown operator: {operator!r}')


def _format_code(code: str) -> str:
    """Format code with line numbers."""
    lines = code.splitlines()
    format_str = f'%{{:0{len(repr(len(lines) + 1))}d}}  %s\n'
    return ''.join([(format_str % (i + 1, line)) for i, line in enumerate(lines)])
