"""
Python advanced pretty printer.  This pretty printer is intended to
replace the old `pprint` python module which does not allow developers
to provide their own pretty print callbacks.
This module is based on ruby's `prettyprint.rb` library by `Tanaka Akira`.
Example Usage
-------------
To get a string of the output use `pretty`::
    from pretty import pretty
    string = pretty(complex_object)
Extending
---------
The pretty library allows developers to add pretty printing rules for their
own objects.  This process is straightforward.  All you have to do is to
add a `_repr_pretty_` method to your object and call the methods on the
pretty printer passed::
    class MyObject(object):
        def _repr_pretty_(self, p, cycle):
            ...
Here is an example implementation of a `_repr_pretty_` method for a list
subclass::
    class MyList(list):
        def _repr_pretty_(self, p, cycle):
            if cycle:
                p.text('MyList(...)')
            else:
                with p.group(8, 'MyList([', '])'):
                    for idx, item in enumerate(self):
                        if idx:
                            p.text(',')
                            p.breakable()
                        p.pretty(item)
The `cycle` parameter is `True` if pretty detected a cycle.  You *have* to
react to that or the result is an infinite loop.  `p.text()` just adds
non breaking text to the output, `p.breakable()` either adds a whitespace
or breaks here.  If you pass it an argument it's used instead of the
default space.  `p.pretty` prettyprints another object using the pretty print
method.
The first parameter to the `group` function specifies the extra indentation
of the next line.  In this example the next item will either be on the same
line (if the items are short enough) or aligned with the right edge of the
opening bracket of `MyList`.
If you just want to indent something you can use the group function
without open / close parameters.  You can also use this code::
    with p.indent(2):
        ...
Inheritance diagram:
.. inheritance-diagram:: IPython.lib.pretty
   :parts: 3
:copyright: 2007 by Armin Ronacher.
            Portions (c) 2009 by Robert Kern.
:license: BSD License.
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
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar, Union
if TYPE_CHECKING:
    from typing import TypeAlias
    from hypothesis.control import BuildContext
T = TypeVar('T')
PrettyPrintFunction = Callable[[Any, 'RepresentationPrinter', bool], None]
__all__ = ['IDKey', 'RepresentationPrinter', 'pretty']


def func_9jxa5c2m(obj, attr, default=None):
    """Safe version of getattr.

    Same as getattr, but will return ``default`` on any Exception,
    rather than raising.

    """
    try:
        return getattr(obj, attr, default)
    except Exception:
        return default


def func_sdcwjvrw(obj):
    """Pretty print the object's representation."""
    printer = RepresentationPrinter()
    printer.pretty(obj)
    return printer.getvalue()


class IDKey:

    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return hash((type(self), id(self.value)))

    def __eq__(self, __o):
        return isinstance(__o, type(self)) and id(self.value) == id(__o.value)


class RepresentationPrinter:
    """Special pretty printer that has a `pretty` method that calls the pretty
    printer for a python object.

    This class stores processing data on `self` so you must *never* use
    this class in a threaded environment.  Always lock it or
    reinstantiate it.

    """

    def __init__(self, output=None, *, context=None):
        """Optionally pass the output stream and the current build context.

        We use the context to represent objects constructed by strategies by showing
        *how* they were constructed, and add annotations showing which parts of the
        minimal failing example can vary without changing the test result.
        """
        self.broken = False
        self.output = StringIO() if output is None else output
        self.max_width = 79
        self.max_seq_length = 1000
        self.output_width = 0
        self.buffer_width = 0
        self.buffer = deque()
        root_group = Group(0)
        self.group_stack = [root_group]
        self.group_queue = GroupQueue(root_group)
        self.indentation = 0
        self.stack = []
        self.singleton_pprinters = {}
        self.type_pprinters = {}
        self.deferred_pprinters = {}
        if 'IPython.lib.pretty' in sys.modules:
            ipp = sys.modules['IPython.lib.pretty']
            self.singleton_pprinters.update(ipp._singleton_pprinters)
            self.type_pprinters.update(ipp._type_pprinters)
            self.deferred_pprinters.update(ipp._deferred_type_pprinters)
        self.singleton_pprinters.update(_singleton_pprinters)
        self.type_pprinters.update(_type_pprinters)
        self.deferred_pprinters.update(_deferred_type_pprinters)
        if context is None:
            self.known_object_printers = defaultdict(list)
            self.slice_comments = {}
        else:
            self.known_object_printers = context.known_object_printers
            self.slice_comments = context.data.slice_comments
        assert all(isinstance(k, IDKey) for k in self.known_object_printers)

    def func_sdcwjvrw(self, obj):
        """Pretty print the given object."""
        obj_id = id(obj)
        cycle = obj_id in self.stack
        self.stack.append(obj_id)
        try:
            with self.group():
                obj_class = func_9jxa5c2m(obj, '__class__', None) or type(obj)
                try:
                    printer = self.singleton_pprinters[obj_id]
                except (TypeError, KeyError):
                    pass
                else:
                    return printer(obj, self, cycle)
                pretty_method = func_9jxa5c2m(obj, '_repr_pretty_', None)
                if callable(pretty_method):
                    return pretty_method(self, cycle)
                for cls in obj_class.__mro__:
                    if cls in self.type_pprinters:
                        return self.type_pprinters[cls](obj, self, cycle)
                    else:
                        key = func_9jxa5c2m(cls, '__module__', None
                            ), func_9jxa5c2m(cls, '__name__', None)
                        if key in self.deferred_pprinters:
                            printer = self.deferred_pprinters.pop(key)
                            self.type_pprinters[cls] = printer
                            return printer(obj, self, cycle)
                        else:
                            if hasattr(cls, '__attrs_attrs__'):
                                return pprint_fields(obj, self, cycle, [at.
                                    name for at in cls.__attrs_attrs__ if
                                    at.init])
                            if hasattr(cls, '__dataclass_fields__'):
                                return pprint_fields(obj, self, cycle, [k for
                                    k, v in cls.__dataclass_fields__.items(
                                    ) if v.init])
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

    def func_hspnlc31(self):
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

    def func_c7vh15ll(self, obj):
        """Add literal text to the output."""
        width = len(obj)
        if self.buffer:
            text = self.buffer[-1]
            if not isinstance(text, Text):
                text = Text()
                self.buffer.append(text)
            func_c7vh15ll.add(obj, width)
            self.buffer_width += width
            self._break_outer_groups()
        else:
            self.output.write(obj)
            self.output_width += width

    def func_273o4ddn(self, sep=' '):
        """Add a breakable separator to the output.

        This does not mean that it will automatically break here.  If no
        breaking on this position takes place the `sep` is inserted
        which default to one space.

        """
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

    def func_zz1xxd1b(self):
        """Explicitly insert a newline into the output, maintaining correct
        indentation."""
        self.flush()
        self.output.write('\n' + ' ' * self.indentation)
        self.output_width = self.indentation
        self.buffer_width = 0

    @contextmanager
    def func_t5280hoc(self, indent):
        """`with`-statement support for indenting/dedenting."""
        self.indentation += indent
        try:
            yield
        finally:
            self.indentation -= indent

    @contextmanager
    def func_2z801uvl(self, indent=0, open='', close=''):
        """Context manager for an indented group.

            with p.group(1, '{', '}'):

        The first parameter specifies the indentation for the next line
        (usually the width of the opening text), the second and third the
        opening and closing delimiters.
        """
        self.begin_group(indent=indent, open=open)
        try:
            yield
        finally:
            self.end_group(dedent=indent, close=close)

    def func_cj246ves(self, indent=0, open=''):
        """Use the `with group(...) context manager instead.

        The begin_group() and end_group() methods are for IPython compatibility only;
        see https://github.com/HypothesisWorks/hypothesis/issues/3721 for details.
        """
        if open:
            self.text(open)
        group = Group(self.group_stack[-1].depth + 1)
        self.group_stack.append(group)
        self.group_queue.enq(group)
        self.indentation += indent

    def func_nfg2pm49(self, dedent=0, close=''):
        """See begin_group()."""
        self.indentation -= dedent
        group = self.group_stack.pop()
        if not group.breakables:
            self.group_queue.remove(group)
        if close:
            self.text(close)

    def func_jz3bl8pj(self, seq):
        """Like enumerate, but with an upper limit on the number of items."""
        for idx, x in enumerate(seq):
            if self.max_seq_length and idx >= self.max_seq_length:
                self.text(',')
                self.breakable()
                self.text('...')
                return
            yield idx, x

    def func_1ratsq3e(self):
        """Flush data that is left in the buffer."""
        for data in self.buffer:
            self.output_width += data.output(self.output, self.output_width)
        self.buffer.clear()
        self.buffer_width = 0

    def func_om7caa6h(self):
        assert isinstance(self.output, StringIO)
        self.flush()
        return self.output.getvalue()

    def func_zqo9hqke(self, obj, cycle, name, args, kwargs):
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

    def func_vglouwau(self, func_name, args, kwargs, *, force_split=None,
        arg_slices=None, leading_comment=None):
        """Helper function to represent a function call.

        - func_name, args, and kwargs should all be pretty obvious.
        - If split_lines, we'll force one-argument-per-line; otherwise we'll place
          calls that fit on a single line (and split otherwise).
        - arg_slices is a mapping from pos-idx or keyword to (start_idx, end_idx)
          of the Conjecture buffer, by which we can look up comments to add.
        """
        assert isinstance(func_name, str)
        if func_name.startswith(('lambda:', 'lambda ')):
            func_name = f'({func_name})'
        self.text(func_name)
        all_args = [(None, v) for v in args] + list(kwargs.items())
        comments = {k: self.slice_comments[v] for k, v in (arg_slices or {}
            ).items() if v in self.slice_comments}
        if leading_comment or any(k in comments for k, _ in all_args):
            force_split = True
        if force_split is None:
            p = RepresentationPrinter()
            p.stack = self.stack.copy()
            p.known_object_printers = self.known_object_printers
            p.repr_call('_' * self.output_width, args, kwargs, force_split=
                False)
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

    def func_np0nllho(self, stream, output_width):
        raise NotImplementedError


class Text(Printable):

    def __init__(self):
        self.objs = []
        self.width = 0

    def func_np0nllho(self, stream, output_width):
        for obj in self.objs:
            stream.write(obj)
        return output_width + self.width

    def func_mnuwebbe(self, obj, width):
        self.objs.append(obj)
        self.width += width


class Breakable(Printable):

    def __init__(self, seq, width, pretty):
        self.obj = seq
        self.width = width
        self.pretty = pretty
        self.indentation = pretty.indentation
        self.group = pretty.group_stack[-1]
        self.group.breakables.append(self)

    def func_np0nllho(self, stream, output_width):
        self.group.breakables.popleft()
        if self.group.want_break:
            stream.write('\n' + ' ' * self.indentation)
            return self.indentation
        if not self.group.breakables:
            self.pretty.group_queue.remove(self.group)
        stream.write(self.obj)
        return output_width + self.width


class Group(Printable):

    def __init__(self, depth):
        self.depth = depth
        self.breakables = deque()
        self.want_break = False


class GroupQueue:

    def __init__(self, *groups):
        self.queue = []
        for group in groups:
            self.enq(group)

    def func_13lfmk8r(self, group):
        depth = group.depth
        while depth > len(self.queue) - 1:
            self.queue.append([])
        self.queue[depth].append(group)

    def func_9wo12h92(self):
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

    def func_yazumnbf(self, group):
        try:
            self.queue[group.depth].remove(group)
        except ValueError:
            pass


def func_hphtq3t2(start, end, basetype):
    """Factory that returns a pprint function useful for sequences.

    Used by the default pprint for tuples, dicts, and lists.
    """

    def func_5iw1irmp(obj, p, cycle):
        typ = type(obj)
        if (basetype is not None and typ is not basetype and typ.__repr__ !=
            basetype.__repr__):
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


def func_aoyidz10(cls):
    class_name = func_9jxa5c2m(cls, '__qualname__', cls.__name__)
    assert isinstance(class_name, str)
    return class_name


def func_fajrlavv(start, end, basetype):
    """Factory that returns a pprint function useful for sets and
    frozensets."""

    def func_5iw1irmp(obj, p, cycle):
        typ = type(obj)
        if (basetype is not None and typ is not basetype and typ.__repr__ !=
            basetype.__repr__):
            return p.text(typ.__repr__(obj))
        if cycle:
            return p.text(start + '...' + end)
        if not obj:
            p.text(func_aoyidz10(basetype) + '()')
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


def func_3dgvolh0(start, end, basetype=None):
    """Factory that returns a pprint function used by the default pprint of
    dicts and dict proxies."""

    def func_5iw1irmp(obj, p, cycle):
        typ = type(obj)
        if (basetype is not None and typ is not basetype and typ.__repr__ !=
            basetype.__repr__):
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
    inner.__name__ = (
        f'_dict_pprinter_factory({start!r}, {end!r}, {basetype!r})')
    return inner


def func_1iitzu1s(obj, p, cycle):
    """The pprint for the super type."""
    with p.group(8, '<super: ', '>'):
        p.pretty(obj.__thisclass__)
        p.text(',')
        p.breakable()
        p.pretty(obj.__self__)


def func_vuaya6qx(obj, p, cycle):
    """The pprint function for regular expression patterns."""
    p.text('re.compile(')
    pattern = repr(obj.pattern)
    if pattern[:1] in 'uU':
        pattern = pattern[1:]
        prefix = 'ur'
    else:
        prefix = 'r'
    pattern = prefix + pattern.replace('\\\\', '\\')
    p.text(pattern)
    if obj.flags:
        p.text(',')
        p.breakable()
        done_one = False
        for flag in ('TEMPLATE', 'IGNORECASE', 'LOCALE', 'MULTILINE',
            'DOTALL', 'UNICODE', 'VERBOSE', 'DEBUG'):
            if obj.flags & getattr(re, flag, 0):
                if done_one:
                    p.text('|')
                p.text('re.' + flag)
                done_one = True
    p.text(')')


def func_rtmqsbkn(obj, p, cycle):
    """The pprint for classes and types."""
    if type(obj).__repr__ != type.__repr__:
        _repr_pprint(obj, p, cycle)
        return
    mod = func_9jxa5c2m(obj, '__module__', None)
    try:
        name = obj.__qualname__
    except Exception:
        name = obj.__name__
        if not isinstance(name, str):
            name = '<unknown type>'
    if mod in (None, '__builtin__', 'builtins', 'exceptions'):
        p.text(name)
    else:
        p.text(mod + '.' + name)


def func_tnq83km3(obj, p, cycle):
    """A pprint that just redirects to the normal repr function."""
    output = repr(obj)
    for idx, output_line in enumerate(func_np0nllho.splitlines()):
        if idx:
            p.break_()
        p.text(output_line)


def func_1x35rjua(obj, p, cycle, fields):
    name = func_aoyidz10(obj.__class__)
    if cycle:
        return p.text(f'{name}(...)')
    with p.group(1, name + '(', ')'):
        for idx, field in enumerate(fields):
            if idx:
                p.text(',')
                p.breakable()
            p.text(field)
            p.text('=')
            p.pretty(getattr(obj, field))


def func_s1o6rieo(obj, p, cycle):
    """Base pprint for all functions and builtin functions."""
    from hypothesis.internal.reflection import get_pretty_function_description
    p.text(get_pretty_function_description(obj))


def func_samxwgiz(obj, p, cycle):
    """Base pprint for all exceptions."""
    name = getattr(obj.__class__, '__qualname__', obj.__class__.__name__)
    if obj.__class__.__module__ not in ('exceptions', 'builtins'):
        name = f'{obj.__class__.__module__}.{name}'
    step = len(name) + 1
    with p.group(step, name + '(', ')'):
        for idx, arg in enumerate(getattr(obj, 'args', ())):
            if idx:
                p.text(',')
                p.breakable()
            p.pretty(arg)


def func_zzz2t6bd(obj, p, cycle):
    if abs(obj) < 1000000000:
        p.text(repr(obj))
    elif abs(obj) < 10 ** 640:
        p.text(f'{obj:#_d}')
    else:
        p.text(f'{obj:#_x}')


def func_rnj56r30(obj, p, cycle):
    if isnan(obj):
        if struct.pack('!d', abs(obj)) != struct.pack('!d', float('nan')):
            show = hex(*struct.unpack('Q', struct.pack('d', obj)))
            return p.text(f"struct.unpack('d', struct.pack('Q', {show}))[0]")
        elif copysign(1.0, obj) == -1.0:
            return p.text('-nan')
    p.text(repr(obj))


_type_pprinters = {int: _repr_integer, float: _repr_float_counting_nans,
    str: _repr_pprint, tuple: func_hphtq3t2('(', ')', tuple), list:
    func_hphtq3t2('[', ']', list), dict: func_3dgvolh0('{', '}', dict), set:
    func_fajrlavv('{', '}', set), frozenset: func_fajrlavv('frozenset({',
    '})', frozenset), super: _super_pprint, re.Pattern: _re_pattern_pprint,
    type: _type_pprint, types.FunctionType: _function_pprint, types.
    BuiltinFunctionType: _function_pprint, types.MethodType:
    _function_pprint, datetime.datetime: _repr_pprint, datetime.timedelta:
    _repr_pprint, BaseException: _exception_pprint, slice: _repr_pprint,
    range: _repr_pprint, bytes: _repr_pprint}
_deferred_type_pprinters = {}


def func_xts7ncx9(type_module, type_name, func):
    """Add a pretty printer for a type specified by the module and name of a
    type rather than the type object itself."""
    key = type_module, type_name
    oldfunc = _deferred_type_pprinters.get(key)
    _deferred_type_pprinters[key] = func
    return oldfunc


_singleton_pprinters = dict.fromkeys(map(id, [None, True, False, Ellipsis,
    NotImplemented]), _repr_pprint)


def func_gq2na86o(obj, p, cycle):
    name = obj.__class__.__name__
    with p.group(len(name) + 1, name + '(', ')'):
        if cycle:
            p.text('...')
        else:
            p.pretty(obj.default_factory)
            p.text(',')
            p.breakable()
            p.pretty(dict(obj))


def func_awhrfj9u(obj, p, cycle):
    name = obj.__class__.__name__
    with p.group(len(name) + 1, name + '(', ')'):
        if cycle:
            p.text('...')
        elif obj:
            p.pretty(list(obj.items()))


def func_rqganb4k(obj, p, cycle):
    name = obj.__class__.__name__
    with p.group(len(name) + 1, name + '(', ')'):
        if cycle:
            p.text('...')
        else:
            p.pretty(list(obj))


def func_2gtbzysk(obj, p, cycle):
    name = obj.__class__.__name__
    with p.group(len(name) + 1, name + '(', ')'):
        if cycle:
            p.text('...')
        elif obj:
            p.pretty(dict(obj))


def func_r5vusd48(obj, p, cycle):
    with p.indent(4):
        p.break_()
        func_tnq83km3(obj, p, cycle)
    p.break_()


def func_35r121ix(obj, p, cycle):
    tname = func_aoyidz10(type(obj))
    if isinstance(obj, Flag):
        p.text(' | '.join(f'{tname}.{x.name}' for x in type(obj) if x & obj ==
            x) or f'{tname}({obj.value!r})')
    else:
        p.text(f'{tname}.{obj.name}')


class _ReprDots:

    def __repr__(self):
        return '...'


def func_p8ch303g(obj, p, cycle):
    args, kw = obj.args, obj.keywords
    if cycle:
        args, kw = (_ReprDots(),), {}
    p.repr_call(func_sdcwjvrw(type(obj)), (obj.func, *args), kw)


func_xts7ncx9('collections', 'defaultdict', _defaultdict_pprint)
func_xts7ncx9('collections', 'OrderedDict', _ordereddict_pprint)
func_xts7ncx9('ordereddict', 'OrderedDict', _ordereddict_pprint)
func_xts7ncx9('collections', 'deque', _deque_pprint)
func_xts7ncx9('collections', 'Counter', _counter_pprint)
func_xts7ncx9('pandas.core.frame', 'DataFrame', _repr_dataframe)
func_xts7ncx9('enum', 'Enum', _repr_enum)
func_xts7ncx9('functools', 'partial', _repr_partial)
