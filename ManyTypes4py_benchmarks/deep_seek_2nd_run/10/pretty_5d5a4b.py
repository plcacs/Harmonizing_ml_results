"""
Python advanced pretty printer.  This pretty printer is intended to
replace the old `pprint` python module which does not allow developers
to provide their own pretty print callbacks.
This module is based on ruby's `prettyprint.rb` library by `Tanaka Akira`.
"""
import ast
import datetime
import re
import struct
import sys
import types
import warnings
from collections import Counter, OrderedDict, defaultdict, deque
from collections.abc import Generator, Iterable, Sequence
from contextlib import contextmanager, suppress
from enum import Enum, Flag
from functools import partial
from io import StringIO, TextIOBase
from math import copysign, isnan
from typing import (
    TYPE_CHECKING, Any, Callable, DefaultDict, Deque, Dict, List, Optional, 
    Set, Tuple, Type, TypeVar, Union, cast
)

if TYPE_CHECKING:
    from typing import TypeAlias
    from hypothesis.control import BuildContext

T = TypeVar('T')
PrettyPrintFunction = Callable[[Any, 'RepresentationPrinter', bool], None]
__all__ = ['IDKey', 'RepresentationPrinter', 'pretty']

def _safe_getattr(obj: Any, attr: str, default: Any = None) -> Any:
    """Safe version of getattr."""
    try:
        return getattr(obj, attr, default)
    except Exception:
        return default

def pretty(obj: Any) -> str:
    """Pretty print the object's representation."""
    printer = RepresentationPrinter()
    printer.pretty(obj)
    return printer.getvalue()

class IDKey:
    def __init__(self, value: Any) -> None:
        self.value = value

    def __hash__(self) -> int:
        return hash((type(self), id(self.value)))

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, type(self)) and id(self.value) == id(cast(IDKey, __o).value

class RepresentationPrinter:
    def __init__(self, output: Optional[TextIOBase] = None, *, context: Optional['BuildContext'] = None) -> None:
        self.broken: bool = False
        self.output: TextIOBase = StringIO() if output is None else output
        self.max_width: int = 79
        self.max_seq_length: int = 1000
        self.output_width: int = 0
        self.buffer_width: int = 0
        self.buffer: deque[Printable] = deque()
        root_group = Group(0)
        self.group_stack: List[Group] = [root_group]
        self.group_queue: GroupQueue = GroupQueue(root_group)
        self.indentation: int = 0
        self.stack: List[int] = []
        self.singleton_pprinters: Dict[int, PrettyPrintFunction] = {}
        self.type_pprinters: Dict[Type[Any], PrettyPrintFunction] = {}
        self.deferred_pprinters: Dict[Tuple[Optional[str], Optional[str]], PrettyPrintFunction] = {}
        if 'IPython.lib.pretty' in sys.modules:
            ipp = sys.modules['IPython.lib.pretty']
            self.singleton_pprinters.update(ipp._singleton_pprinters)
            self.type_pprinters.update(ipp._type_pprinters)
            self.deferred_pprinters.update(ipp._deferred_type_pprinters)
        self.singleton_pprinters.update(_singleton_pprinters)
        self.type_pprinters.update(_type_pprinters)
        self.deferred_pprinters.update(_deferred_type_pprinters)
        if context is None:
            self.known_object_printers: DefaultDict[IDKey, List[PrettyPrintFunction]] = defaultdict(list)
            self.slice_comments: Dict[Any, str] = {}
        else:
            self.known_object_printers = context.known_object_printers
            self.slice_comments = context.data.slice_comments
        assert all((isinstance(k, IDKey) for k in self.known_object_printers))

    def pretty(self, obj: Any) -> None:
        """Pretty print the given object."""
        obj_id = id(obj)
        cycle = obj_id in self.stack
        self.stack.append(obj_id)
        try:
            with self.group():
                obj_class = _safe_getattr(obj, '__class__', None) or type(obj)
                try:
                    printer = self.singleton_pprinters[obj_id]
                except (TypeError, KeyError):
                    pass
                else:
                    return printer(obj, self, cycle)
                pretty_method = _safe_getattr(obj, '_repr_pretty_', None)
                if callable(pretty_method):
                    return pretty_method(self, cycle)
                for cls in obj_class.__mro__:
                    if cls in self.type_pprinters:
                        return self.type_pprinters[cls](obj, self, cycle)
                    else:
                        key = (_safe_getattr(cls, '__module__', None), _safe_getattr(cls, '__name__', None)
                        if key in self.deferred_pprinters:
                            printer = self.deferred_pprinters.pop(key)
                            self.type_pprinters[cls] = printer
                            return printer(obj, self, cycle)
                        else:
                            if hasattr(cls, '__attrs_attrs__'):
                                return pprint_fields(obj, self, cycle, [at.name for at in cls.__attrs_attrs__ if at.init])
                            if hasattr(cls, '__dataclass_fields__'):
                                return pprint_fields(obj, self, cycle, [k for k, v in cls.__dataclass_fields__.items() if v.init])
                printers = self.known_object_printers[IDKey(obj)]
                if len(printers) == 1:
                    return printers[0](obj, self, cycle)
                elif printers:
                    strs = set()
                    for f in printers:
                        p = RepresentationPrinter()
                        f(obj, p, cycle)
                        strs.add(p.getvalue())
                    if len(strs) == 1:
                        return printers[0](obj, self, cycle)
                return _repr_pprint(obj, self, cycle)
        finally:
            self.stack.pop()

    def _break_outer_groups(self) -> None:
        while self.max_width < self.output_width + self.buffer_width:
            group = self.group_queue.deq()
            if not group:
                return
            while group.breakables:
                x = self.buffer.popleft()
                self.output_width = x.output(self.output, self.output_width)
                self.buffer_width -= x.width
            while self.buffer and isinstance(self.buffer[0], Text):
                x = self.buffer.popleft()
                self.output_width = x.output(self.output, self.output_width)
                self.buffer_width -= x.width

    def text(self, obj: str) -> None:
        """Add literal text to the output."""
        width = len(obj)
        if self.buffer:
            text = self.buffer[-1]
            if not isinstance(text, Text):
                text = Text()
                self.buffer.append(text)
            text.add(obj, width)
            self.buffer_width += width
            self._break_outer_groups()
        else:
            self.output.write(obj)
            self.output_width += width

    def breakable(self, sep: str = ' ') -> None:
        """Add a breakable separator to the output."""
        width = len(sep)
        group = self.group_stack[-1]
        if group.want_break:
            self.flush()
            self.output.write('\n' + ' ' * self.indentation)
            self.output_width = self.indentation
            self.buffer_width = 0
        else:
            self.buffer.append(Breakable(sep, width, self))
            self.buffer_width += width
            self._break_outer_groups()

    def break_(self) -> None:
        """Explicitly insert a newline into the output."""
        self.flush()
        self.output.write('\n' + ' ' * self.indentation)
        self.output_width = self.indentation
        self.buffer_width = 0

    @contextmanager
    def indent(self, indent: int) -> Generator[None, None, None]:
        """`with`-statement support for indenting/dedenting."""
        self.indentation += indent
        try:
            yield
        finally:
            self.indentation -= indent

    @contextmanager
    def group(self, indent: int = 0, open: str = '', close: str = '') -> Generator[None, None, None]:
        """Context manager for an indented group."""
        self.begin_group(indent=indent, open=open)
        try:
            yield
        finally:
            self.end_group(dedent=indent, close=close)

    def begin_group(self, indent: int = 0, open: str = '') -> None:
        """Begin a new group."""
        if open:
            self.text(open)
        group = Group(self.group_stack[-1].depth + 1)
        self.group_stack.append(group)
        self.group_queue.enq(group)
        self.indentation += indent

    def end_group(self, dedent: int = 0, close: str = '') -> None:
        """End a group."""
        self.indentation -= dedent
        group = self.group_stack.pop()
        if not group.breakables:
            self.group_queue.remove(group)
        if close:
            self.text(close)

    def _enumerate(self, seq: Sequence[Any]) -> Generator[Tuple[int, Any], None, None]:
        """Like enumerate, but with an upper limit on the number of items."""
        for idx, x in enumerate(seq):
            if self.max_seq_length and idx >= self.max_seq_length:
                self.text(',')
                self.breakable()
                self.text('...')
                return
            yield (idx, x)

    def flush(self) -> None:
        """Flush data that is left in the buffer."""
        for data in self.buffer:
            self.output_width += data.output(self.output, self.output_width)
        self.buffer.clear()
        self.buffer_width = 0

    def getvalue(self) -> str:
        """Get the string value of the output."""
        assert isinstance(self.output, StringIO)
        self.flush()
        return self.output.getvalue()

    def maybe_repr_known_object_as_call(self, obj: Any, cycle: bool, name: str, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
        if cycle:
            return self.text('<...>')
        with suppress(Exception):
            ast.parse(repr(obj))
            p = RepresentationPrinter()
            p.stack = self.stack.copy()
            p.known_object_printers = self.known_object_printers
            p.repr_call(name, args, kwargs)
            try:
                ast.parse(p.getvalue())
            except Exception:
                return _repr_pprint(obj, self, cycle)
        return self.repr_call(name, args, kwargs)

    def repr_call(self, func_name: str, args: Tuple[Any, ...], kwargs: Dict[str, Any], *, force_split: Optional[bool] = None, arg_slices: Optional[Dict[Union[int, str], Any]] = None, leading_comment: Optional[str] = None) -> None:
        """Helper function to represent a function call."""
        assert isinstance(func_name, str)
        if func_name.startswith(('lambda:', 'lambda ')):
            func_name = f'({func_name})'
        self.text(func_name)
        all_args = [(None, v) for v in args] + list(kwargs.items())
        comments = {k: self.slice_comments[v] for k, v in (arg_slices or {}).items() if v in self.slice_comments}
        if leading_comment or any((k in comments for k, _ in all_args)):
            force_split = True
        if force_split is None:
            p = RepresentationPrinter()
            p.stack = self.stack.copy()
            p.known_object_printers = self.known_object_printers
            p.repr_call('_' * self.output_width, args, kwargs, force_split=False)
            s = p.getvalue()
            force_split = '\n' in s
        with self.group(indent=4, open='(', close=''):
            for i, (k, v) in enumerate(all_args):
                if force_split:
                    if i == 0 and leading_comment:
                        self.break_()
                        self.text(leading_comment)
                    self.break_()
                else:
                    assert leading_comment is None
                    self.breakable(' ' if i else '')
                if k:
                    self.text(f'{k}=')
                self.pretty(v)
                if force_split or i + 1 < len(all_args):
                    self.text(',')
                comment = None
                if k is not None:
                    comment = comments.get(i) or comments.get(k)
                if comment:
                    self.text(f'  # {comment}')
        if all_args and force_split:
            self.break_()
        self.text(')')

class Printable:
    def output(self, stream: TextIOBase, output_width: int) -> int:
        raise NotImplementedError

class Text(Printable):
    def __init__(self) -> None:
        self.objs: List[str] = []
        self.width: int = 0

    def output(self, stream: TextIOBase, output_width: int) -> int:
        for obj in self.objs:
            stream.write(obj)
        return output_width + self.width

    def add(self, obj: str, width: int) -> None:
        self.objs.append(obj)
        self.width += width

class Breakable(Printable):
    def __init__(self, seq: str, width: int, pretty: RepresentationPrinter) -> None:
        self.obj = seq
        self.width = width
        self.pretty = pretty
        self.indentation = pretty.indentation
        self.group = pretty.group_stack[-1]
        self.group.breakables.append(self)

    def output(self, stream: TextIOBase, output_width: int) -> int:
        self.group.breakables.popleft()
        if self.group.want_break:
            stream.write('\n' + ' ' * self.indentation)
            return self.indentation
        if not self.group.breakables:
            self.pretty.group_queue.remove(self.group)
        stream.write(self.obj)
        return output_width + self.width

class Group(Printable):
    def __init__(self, depth: int) -> None:
        self.depth = depth
        self.breakables: deque[Breakable] = deque()
        self.want_break = False

class GroupQueue:
    def __init__(self, *groups: Group) -> None:
        self.queue: List[List[Group]] = []
        for group in groups:
            self.enq(group)

    def enq(self, group: Group) -> None:
        depth = group.depth
        while depth > len(self.queue) - 1:
            self.queue.append([])
        self.queue[depth].append(group)

    def deq(self) -> Optional[Group]:
        for stack in self.queue:
            for idx, group in enumerate(reversed(stack)):
                if group.breakables:
                    del stack[idx]
                    group.want_break = True
                    return group
            for group in stack:
                group.want_break = True
            del stack[:]
        return None

    def remove(self, group: Group) -> None:
        try:
            self.queue[group.depth].remove(group)
        except ValueError:
            pass

def _seq_pprinter_factory(start: str, end: str, basetype: Optional[Type[Any]] = None) -> PrettyPrintFunction:
    def inner(obj: Any, p: RepresentationPrinter, cycle: bool) -> None:
        typ = type(obj)
        if basetype is not None and typ is not basetype and (typ.__repr__ != basetype.__repr__):
            return p.text(typ.__repr__(obj))
        if cycle:
            return p.text(start + '...' + end)
        step = len(start)
        with p.group(step, start, end):
            for idx, x in p._enumerate(obj):
                if idx:
                    p.text(',')
                    p.breakable()
                p.pretty(x)
            if len(obj) == 1 and type(obj) is tuple:
                p.text(',')
    return inner

def get_class_name(cls: Type[Any]) -> str:
    class_name = _safe_getattr(cls, '__qualname__', cls.__name__)
    assert isinstance(class_name, str)
    return class_name

def _set_pprinter_factory(start: str, end: str, basetype: Optional[Type[Any]] = None) -> PrettyPrintFunction:
    def inner(obj: Any, p: RepresentationPrinter, cycle: bool) -> None:
        typ = type(obj)
        if basetype is not None and typ is not basetype and (typ.__repr__ != basetype.__repr__):
            return p.text(typ.__repr__(obj))
        if cycle:
            return p.text(start + '...' + end)
        if not obj:
            p.text(get_class_name(basetype) + '()')
        else:
            step = len(start)
            with p.group(step, start, end):
                items = obj
                if not (p.max_seq_length and len(obj) >= p.max_seq_length):
                    try:
                        items = sorted(obj)
                    except Exception:
                        pass
                for idx, x in p._enumerate(items):
                    if idx:
                        p.text(',')
                        p.breakable()
                    p.pretty(x)
    return inner

def _dict_pprinter_factory(start: str, end: str, basetype: Optional[Type[Any]] = None) -> PrettyPrintFunction:
    def inner(obj: Any, p: RepresentationPrinter, cycle: bool) -> None:
        typ = type(obj)
        if basetype is not None and typ is not basetype and (typ.__repr__ != basetype.__repr__):
            return p.text(typ.__repr__(obj))
        if cycle:
            return p.text('{...}')
        with p.group(1, start, end):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', BytesWarning)
                for idx, key in p._enumerate(obj):
                    if idx:
                        p.text(',')
                        p.breakable()
                    p.pretty(key)
                    p.text(': ')
                    p.pretty(obj[key])
    inner.__name__ = f'_dict_pp