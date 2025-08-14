from typing import Any, Union, Callable, List, Dict, Iterable, Optional, TextIO, Tuple, ContextManager, TypeVar, cast

_DEFAULT_AUTOESCAPE = "xhtml_escape"


class _UnsetMarker:
    pass


_UNSET = _UnsetMarker()


def filter_whitespace(mode: str, text: str) -> str:
    """Transform whitespace in ``text`` according to ``mode``."""
    if mode == "all":
        return text
    elif mode == "single":
        text = re.sub(r"([\t ]+)", " ", text)
        text = re.sub(r"(\s*\n\s*)", "\n", text)
        return text
    elif mode == "oneline":
        return re.sub(r"(\s+)", " ", text)
    else:
        raise Exception("invalid whitespace mode %s" % mode)


class Template:
    """A compiled template."""

    def __init__(
        self,
        template_string: Union[str, bytes],
        name: str = "<string>",
        loader: Optional["BaseLoader"] = None,
        compress_whitespace: Union[bool, _UnsetMarker] = _UNSET,
        autoescape: Optional[Union[str, _UnsetMarker]] = _UNSET,
        whitespace: Optional[str] = None,
    ) -> None:
        self.name: str = escape.native_str(name)
        self.autoescape: Optional[str] = None
        self.namespace: Dict[str, Any] = {}
        self.file: _File = None  # type: ignore
        self.code: str = ""
        self.loader: Optional[BaseLoader] = None
        self.compiled: Any = None

        if compress_whitespace is not _UNSET:
            if whitespace is not None:
                raise Exception("cannot set both whitespace and compress_whitespace")
            whitespace = "single" if compress_whitespace else "all"
        if whitespace is None:
            if loader and loader.whitespace:
                whitespace = loader.whitespace
            else:
                if name.endswith(".html") or name.endswith(".js"):
                    whitespace = "single"
                else:
                    whitespace = "all"
        assert whitespace is not None
        filter_whitespace(whitespace, "")

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
            self.compiled = compile(
                escape.to_unicode(self.code),
                "%s.generated.py" % self.name.replace(".", "_"),
                "exec",
                dont_inherit=True,
            )
        except Exception:
            formatted_code = _format_code(self.code).rstrip()
            app_log.error("%s code:\n%s", self.name, formatted_code)
            raise

    def generate(self, **kwargs: Any) -> bytes:
        """Generate this template with the given arguments."""
        namespace: Dict[str, Any] = {
            "escape": escape.xhtml_escape,
            "xhtml_escape": escape.xhtml_escape,
            "url_escape": escape.url_escape,
            "json_encode": escape.json_encode,
            "squeeze": escape.squeeze,
            "linkify": escape.linkify,
            "datetime": datetime,
            "_tt_utf8": escape.utf8,
            "_tt_string_types": (unicode_type, bytes),
            "__name__": self.name.replace(".", "_"),
            "__loader__": ObjectDict(get_source=lambda name: self.code),
        }
        namespace.update(self.namespace)
        namespace.update(kwargs)
        exec_in(self.compiled, namespace)
        execute = cast(Callable[[], bytes], namespace["_tt_execute"])
        linecache.clearcache()
        return execute()

    def _generate_python(self, loader: Optional["BaseLoader"]) -> str:
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

    def _get_ancestors(self, loader: Optional["BaseLoader"]) -> List["_File"]:
        ancestors: List[_File] = [self.file]
        for chunk in self.file.body.chunks:
            if isinstance(chunk, _ExtendsBlock):
                if not loader:
                    raise ParseError(
                        "{% extends %} block found, but no " "template loader"
                    )
                template = loader.load(chunk.name, self.name)
                ancestors.extend(template._get_ancestors(loader))
        return ancestors


class BaseLoader:
    """Base class for template loaders."""

    def __init__(
        self,
        autoescape: Optional[str] = _DEFAULT_AUTOESCAPE,
        namespace: Optional[Dict[str, Any]] = None,
        whitespace: Optional[str] = None,
    ) -> None:
        self.autoescape: Optional[str] = autoescape
        self.namespace: Dict[str, Any] = namespace or {}
        self.whitespace: Optional[str] = whitespace
        self.templates: Dict[str, Template] = {}
        self.lock: threading.RLock = threading.RLock()

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
        self.root: str = os.path.abspath(root_directory)

    def resolve_path(self, name: str, parent_path: Optional[str] = None) -> str:
        if (
            parent_path
            and not parent_path.startswith("<")
            and not parent_path.startswith("/")
            and not name.startswith("/")
        ):
            current_path = os.path.join(self.root, parent_path)
            file_dir = os.path.dirname(os.path.abspath(current_path))
            relative_path = os.path.abspath(os.path.join(file_dir, name))
            if relative_path.startswith(self.root):
                name = relative_path[len(self.root) + 1 :]
        return name

    def _create_template(self, name: str) -> Template:
        path = os.path.join(self.root, name)
        with open(path, "rb") as f:
            template = Template(f.read(), name=name, loader=self)
            return template


class DictLoader(BaseLoader):
    """A template loader that loads from a dictionary."""

    def __init__(self, dict: Dict[str, str], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.dict: Dict[str, str] = dict

    def resolve_path(self, name: str, parent_path: Optional[str] = None) -> str:
        if (
            parent_path
            and not parent_path.startswith("<")
            and not parent_path.startswith("/")
            and not name.startswith("/")
        ):
            file_dir = posixpath.dirname(parent_path)
            name = posixpath.normpath(posixpath.join(file_dir, name))
        return name

    def _create_template(self, name: str) -> Template:
        return Template(self.dict[name], name=name, loader=self)


class _Node:
    def each_child(self) -> Iterable["_Node"]:
        return ()

    def generate(self, writer: "_CodeWriter") -> None:
        raise NotImplementedError()

    def find_named_blocks(
        self, loader: Optional[BaseLoader], named_blocks: Dict[str, "_NamedBlock"]
    ) -> None:
        for child in self.each_child():
            child.find_named_blocks(loader, named_blocks)


class _File(_Node):
    def __init__(self, template: Template, body: "_ChunkList") -> None:
        self.template: Template = template
        self.body: _ChunkList = body
        self.line: int = 0

    def generate(self, writer: "_CodeWriter") -> None:
        writer.write_line("def _tt_execute():", self.line)
        with writer.indent():
            writer.write_line("_tt_buffer = []", self.line)
            writer.write_line("_tt_append = _tt_buffer.append", self.line)
            self.body.generate(writer)
            writer.write_line("return _tt_utf8('').join(_tt_buffer)", self.line)

    def each_child(self) -> Iterable["_Node"]:
        return (self.body,)


class _ChunkList(_Node):
    def __init__(self, chunks: List[_Node]) -> None:
        self.chunks: List[_Node] = chunks

    def generate(self, writer: "_CodeWriter") -> None:
        for chunk in self.chunks:
            chunk.generate(writer)

    def each_child(self) -> Iterable["_Node"]:
        return self.chunks


class _NamedBlock(_Node):
    def __init__(self, name: str, body: _Node, template: Template, line: int) -> None:
        self.name: str = name
        self.body: _Node = body
        self.template: Template = template
        self.line: int = line

    def each_child(self) -> Iterable["_Node"]:
        return (self.body,)

    def generate(self, writer: "_CodeWriter") -> None:
        block = writer.named_blocks[self.name]
        with writer.include(block.template, self.line):
            block.body.generate(writer)

    def find_named_blocks(
        self, loader: Optional[BaseLoader], named_blocks: Dict[str, "_NamedBlock"]
    ) -> None:
        named_blocks[self.name] = self
        _Node.find_named_blocks(self, loader, named_blocks)


class _ExtendsBlock(_Node):
    def __init__(self, name: str) -> None:
        self.name: str = name


class _IncludeBlock(_Node):
    def __init__(self, name: str, reader: "_TemplateReader", line: int) -> None:
        self.name: str = name
        self.template_name: str = reader.name
        self.line: int = line

    def find_named_blocks(
        self, loader: Optional[BaseLoader], named_blocks: Dict[str, _NamedBlock]
    ) -> None:
        assert loader is not None
        included = loader.load(self.name, self.template_name)
        included.file.find_named_blocks(loader, named_blocks)

    def generate(self, writer: "_CodeWriter") -> None:
        assert writer.loader is not None
        included = writer.loader.load(self.name, self.template_name)
        with writer.include(included, self.line):
            included.file.body.generate(writer)


class _ApplyBlock(_Node):
    def __init__(self, method: str, line: int, body: _Node) -> None:
        self.method: str = method
        self.line: int = line
        self.body: _Node = body

    def each_child(self) -> Iterable["_Node"]:
        return (self.body,)

    def generate(self, writer: "_CodeWriter") -> None:
        method_name = "_tt_apply%d" % writer.apply_counter
        writer.apply_counter += 1
        writer.write_line("def %s():" % method_name, self.line)
        with writer.indent():
            writer.write_line("_tt_buffer = []", self.line)
            writer.write_line("_tt_append = _tt_buffer.append", self.line)
            self.body.generate(writer)
            writer.write_line("return _tt_utf8('').join(_tt_buffer)", self.line)
        writer.write_line(
            f"_tt_append(_tt_utf8({self.method}({method_name}())))", self.line
        )


class _ControlBlock(_Node):
    def __init__(self, statement: str, line: int, body: _Node) -> None:
        self.statement: str = statement
        self.line: int = line
        self.body: _Node = body

    def each_child(self) -> Iterable[_Node]:
        return (self.body,)

    def generate(self, writer: "_CodeWriter") -> None:
        writer.write_line("%s:" % self.statement, self.line)
        with writer.indent():
            self.body.generate(writer)
            # Just in case the body was empty
            writer.write_line("pass", self.line)


class _IntermediateControlBlock(_Node):
    def __init__(self, statement: str, line: int) -> None:
        self.statement: str = statement
        self.line: int = line

    def generate(self, writer: "_CodeWriter") -> None:
        # In case the previous block was empty
        writer.write_line("pass", self.line)
        writer.write_line("%s:" % self.statement, self.line, writer.indent_size() - 1)


class _Statement(_Node):
    def __init__(self, statement: str, line: int) -> None:
        self.statement: str = statement
        self.line: int = line

    def generate(self, writer: "_CodeWriter") -> None:
        writer.write_line(self.statement, self.line)


class _Expression(_Node):
    def __init__(self, expression: str, line: int, raw: bool = False) -> None:
        self.expression: str = expression
        self.line: int = line
        self.raw: bool = raw

    def generate(self, writer: "_CodeWriter") -> None:
        writer.write_line("_tt_tmp = %s" % self.expression, self.line)
        writer.write_line(
            "if isinstance(_tt_tmp, _tt_string_types):" " _tt_tmp = _tt_utf8(_tt_tmp)",
            self.line,
        )
        writer.write_line("else: _tt_tmp = _tt_utf8(str(_tt_tmp))", self.line)
        if not self.raw and writer.current_template.autoescape is not None:
            writer.write_line(
                "_tt_tmp = _tt_utf8(%s(_tt_tmp))" % writer.current_template.autoescape,
                self.line,
            )
        writer.write_line("_tt_append(_tt_tmp)", self.line)


class _Module(_Expression):
    def __init__(self, expression: str, line: int) -> None:
        super().__init__("_tt_modules." + expression, line, raw=True)


class _Text(_Node):
    def __init__(self, value: str, line: int, whitespace: str) -> None:
        self.value: str = value
        self.line: int = line
        self.whitespace: str = whitespace

    def generate(self, writer: "_CodeWriter") -> None:
        value = self.value

        if "<pre>" not in value:
            value = filter_whitespace(self.whitespace, value)

        if value:
            writer.write_line("_tt_append(%r)" % escape.utf8(value), self.line)


class ParseError(Exception):
    """Raised for template syntax errors."""

    def __init__(
        self, message: str, filename: Optional[str] = None, lineno: int = 0
    ) -> None:
        self.message: str = message
        self.filename: Optional[str] = filename
        self.lineno: int = lineno

    def __str__(self) -> str:
        return "%s at %s:%d" % (self.message, self.filename, self.lineno)


class _CodeWriter:
    def __init__(
        self,
        file: TextIO,
        named_blocks: Dict[str, _NamedBlock],
        loader: Optional[BaseLoader],
        current_template: Template,
    ) -> None:
        self.file: TextIO = file
        self.named_blocks: Dict[str, _NamedBlock] = named_blocks
        self.loader: Optional[BaseLoader] = loader
        self.current_template: Template = current_template
        self.apply_counter: int = 0
        self.include_stack: List[Tuple[Template, int]] = []
        self._indent: int = 0

    def indent_size(self) -> int:
        return self._indent

    def indent(self) -> ContextManager["_CodeWriter"]:
        class Indenter:
            def __enter__(_) -> "_CodeWriter":
                self._indent += 1
                return self

            def __exit__(_, *args: Any) -> None:
                assert self._indent > 0
                self._indent -= 1

        return Indenter()

    def include(self, template: Template, line: int) -> ContextManager["_CodeWriter"]:
        self.include_stack.append((self.current_template, line))
        self.current_template = template

        class IncludeTemplate:
            def __enter__(_) -> "_CodeWriter":
                return self

            def __exit__(_, *args: Any) -> None:
                self.current_template = self.include_stack.pop()[0]

        return IncludeTemplate()

    def write_line(
        self, line: str, line_number: int, indent: Optional[int] = None
    ) -> None:
        if indent is None:
            indent = self._indent
        line_comment = "  # %s:%d" % (self.current_template.name, line_number)
        if self.include_stack:
            ancestors = [
                "%s:%d" % (tmpl.name, lineno) for (tmpl, lineno) in self.include_stack
            ]
            line_comment += " (via %s)" % ", ".join(reversed(ancestors))
        print("    " * indent + line + line_comment, file=self.file)


class _TemplateReader:
    def __init__(self, name: str, text: str, whitespace: str) -> None:
        self.name: str = name
        self.text: str = text
        self.whitespace: str = whitespace
        self.line: int = 1
        self.pos: int = 0

    def find(self, needle: str, start: int = 0, end: Optional[int] = None) -> int:
        assert start >= 0, start
        pos = self.pos
        start += pos
        if end is None:
            index = self.text.find(needle, start)
        else:
            end += pos
            assert end >= start
            index = self.text.find(needle, start, end)
        if index != -1:
            index -= pos
        return index

    def consume(self, count: Optional[int] = None) -> str:
        if count is None:
            count = len(self.text) - self.pos
        newpos = self.pos + count
        self.line += self.text.count("\n", self.pos, newpos)
        s = self.text[self.pos : newpos]
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
        return self.text[self.pos :]

    def raise_parse_error(self, msg: str) -> None:
        raise ParseError(msg, self.name, self.line)


def _format_code(code: str) -> str:
    lines = code.splitlines()
    format = "%%%dd  %%s\n" % len(repr(len(lines) + 1))
    return "".join([format % (i + 1, line) for (i, line) in enumerate(lines)])


def _parse(
    reader: _TemplateReader,
    template: Template,
    in_block: Optional[str] = None,
    in_loop: Optional[str] = None,
) -> _ChunkList:
    body = _ChunkList([])
    while True:
        curly = 0
        while True:
            curly = reader.find("{", curly)
            if curly == -1 or curly + 1 == reader.remaining():
                if in_block:
                    reader.raise_parse_error(
                        "Missing {%% end %%} block for %s" % in_block
                    )
                body.chunks.append(
                    _Text(reader.consume(), reader.line, reader.whitespace)
                )
                return body
            if reader[curly + 1] not in ("{", "%", "#"):
                curly += 1
                continue
            if (
                curly + 2 < reader.remaining()
                and reader[curly + 1] == "{"
                and reader[curly + 2] == "{"
            ):
                curly += 1
                continue
            break

        if curly > 0:
            cons = reader.consume(curly)
            body.chunks.append(_Text(cons, reader.line, reader.whitespace))

        start_brace = reader.consume(2)
        line = reader.line

        if reader.remaining() and reader[0] == "!":
            reader.consume(1)
            body.chunks.append(_Text(start_brace, line, reader.whitespace))
            continue

        if start_brace == "{#":
            end = reader.find("#}")
            if end == -1:
                reader.raise_parse_error("Missing end comment #}")
            contents = reader.consume(end).strip()
            reader.consume(2)
            continue

        if start_brace == "{{":
            end = reader.find("}}")
            if end == -1:
                reader.raise_parse_error("Missing end expression }}")
            contents = reader.consume(end).strip()
            reader.consume(2)
            if not contents:
                reader.raise_parse_error("Empty expression")
            body.chunks.append(_Expression(contents, line))
            continue

        assert start_brace == "{%", start_brace
        end = reader.find("%}")
        if end == -1:
            reader.raise_parse_error("Missing end block %}")
        contents = reader.consume(end).strip()
        reader.consume(2)
        if not contents:
            reader.raise_parse_error("Empty block tag ({% %})")

        operator, space, suffix = contents.partition(" ")
        suffix = suffix.strip()

        intermediate_blocks: Dict[str, set] = {
            "else": {"if", "for", "while", "try"},
            "elif": {"if"},
            "except": {"try"},
            "finally": {"try"},
        }
        allowed_parents = intermediate_blocks.get(operator)
        if allowed_parents is not None:
            if not in_block:
                reader.raise_parse_error(f"{operator} outside {allowed_parents} block")
            if in_block not in allowed_parents:
                reader.raise_parse_error(
                    f"{operator} block cannot be attached to {in_block} block"
                )
            body.chunks.append(_IntermediateControlBlock(contents, line))
            continue

        elif operator == "end":
            if not in_block:
                reader.raise_parse_error("Extra {% end %} block")
            return body

        elif operator in (
            "extends",
            "include",
            "set",
            "import",
            "from",
            "comment",
            "autoescape",
            "whitespace",
            "raw",
            "module",
        ):
            if operator == "comment":
                continue
            if operator == "extends":
                suffix = suffix.strip('"').strip("'")
                if not suffix:
                    reader.raise_parse_error("extends missing file path")
                block: _Node = _ExtendsBlock(suffix)
            elif operator in ("import", "from"):
                if not suffix:
                    reader.raise_parse_error("import missing statement")
                block = _Statement(contents, line)
            elif operator == "include":
                suffix = suffix.strip('"').strip("'")
                if not suffix:
                    reader.raise_parse_error("include missing file path")
                block = _IncludeBlock(suffix, reader, line)
            elif operator == "set":
                if not suffix:
                    reader.raise_parse_error("set missing statement")
                block = _Statement(suffix, line)
            elif operator == "autoescape":
                fn: Optional[str] = suffix.strip()
                if fn == "None":
                    fn = None
                template.autoescape = fn
                continue
            elif operator == "whitespace":
                mode = suffix.strip()
                filter_whitespace(mode, "")
                reader.whitespace = mode
                continue
            elif operator == "raw":
                block = _Expression(suffix, line, raw=True)
            elif operator == "module":
                block = _Module(suffix, line)
            body.chunks.append(block)
            continue

        elif operator in ("apply", "block", "try", "if", "for", "while"):
            if operator in ("for", "while"):
                block_body = _parse(reader, template, operator, operator)
            elif operator == "apply":
                block_body = _parse(reader, template, operator, None)
            else:
                block_body = _parse(reader, template, operator, in_loop)

            if operator == "apply":
                if not suffix:
                    reader.raise_parse_error("apply missing method name")
                block = _ApplyBlock(suffix, line, block_body)
            elif operator == "block":
                if not suffix:
                    reader.raise_parse_error("block missing name")
                block = _NamedBlock(suffix, block_body, template, line)
            else:
                block = _ControlBlock(contents, line, block_body)
            body.chunks.append(block)
            continue

        elif operator in ("break", "continue"):
            if not in_loop:
                reader.raise_parse_error(
                    "{} outside {} block".format(operator, {"for", "while"})
                )
            body.chunks.append(_Statement(contents, line))
            continue

        else:
            reader.raise_parse_error("unknown operator: %r" % operator)
