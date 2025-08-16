# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

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
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    Deque,
    Set,
    FrozenSet,
)

if TYPE_CHECKING:
    from typing import TypeAlias

    from hypothesis.control import BuildContext

# ruff: noqa: FBT001

T = TypeVar("T")
PrettyPrintFunction: "TypeAlias" = Callable[[Any, "RepresentationPrinter", bool], None]

__all__ = [
    "IDKey",
    "RepresentationPrinter",
    "pretty",
]


def _safe_getattr(obj: object, attr: str, default: Optional[Any] = None) -> Any:
    """Safe version of getattr.

    Same as getattr, but will return ``default`` on any Exception,
    rather than raising.

    """
    try:
        return getattr(obj, attr, default)
    except Exception:
        return default


def pretty(obj: object) -> str:
    """Pretty print the object's representation."""
    printer = RepresentationPrinter()
    printer.pretty(obj)
    return printer.getvalue()


class IDKey:
    def __init__(self, value: object) -> None:
        self.value = value

    def __hash__(self) -> int:
        return hash((type(self), id(self.value)))

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, type(self)) and id(self.value) == id(__o.value)


class RepresentationPrinter:
    """Special pretty printer that has a `pretty` method that calls the pretty
    printer for a python object.

    This class stores processing data on `self` so you must *never* use
    this class in a threaded environment.  Always lock it or
    reinstantiate it.

    """

    def __init__(
        self,
        output: Optional[TextIOBase] = None,
        *,
        context: Optional["BuildContext"] = None,
    ) -> None:
        """Optionally pass the output stream and the current build context.

        We use the context to represent objects constructed by strategies by showing
        *how* they were constructed, and add annotations showing which parts of the
        minimal failing example can vary without changing the test result.
        """
        self.broken: bool = False
        self.output: TextIOBase = StringIO() if output is None else output
        self.max_width: int = 79
        self.max_seq_length: int = 1000
        self.output_width: int = 0
        self.buffer_width: int = 0
        self.buffer: Deque[Union["Breakable", "Text"]] = deque()

        root_group = Group(0)
        self.group_stack: List[Group] = [root_group]
        self.group_queue = GroupQueue(root_group)
        self.indentation: int = 0

        self.stack: List[int] = []
        self.singleton_pprinters: Dict[int, PrettyPrintFunction] = {}
        self.type_pprinters: Dict[type, PrettyPrintFunction] = {}
        self.deferred_pprinters: Dict[Tuple[str, str], PrettyPrintFunction] = {}
        # If IPython has been imported, load up their pretty-printer registry
        if "IPython.lib.pretty" in sys.modules:
            ipp = sys.modules["IPython.lib.pretty"]
            self.singleton_pprinters.update(ipp._singleton_pprinters)
            self.type_pprinters.update(ipp._type_pprinters)
            self.deferred_pprinters.update(ipp._deferred_type_pprinters)
        # If there's overlap between our pprinters and IPython's, we'll use ours.
        self.singleton_pprinters.update(_singleton_pprinters)
        self.type_pprinters.update(_type_pprinters)
        self.deferred_pprinters.update(_deferred_type_pprinters)

        # for which-parts-matter, we track a mapping from the (start_idx, end_idx)
        # of slices into the minimal failing example; this is per-interesting_origin
        # but we report each separately so that's someone else's problem here.
        # Invocations of self.repr_call() can report the slice for each argument,
        # which will then be used to look up the relevant comment if any.
        self.known_object_printers: Dict[IDKey, List[PrettyPrintFunction]]
        self.slice_comments: Dict[Tuple[int, int], str]
        if context is None:
            self.known_object_printers = defaultdict(list)
            self.slice_comments = {}
        else:
            self.known_object_printers = context.known_object_printers
            self.slice_comments = context.data.slice_comments
        assert all(isinstance(k, IDKey) for k in self.known_object_printers)

    def pretty(self, obj: object) -> None:
        """Pretty print the given object."""
        obj_id = id(obj)
        cycle = obj_id in self.stack
        self.stack.append(obj_id)
        try:
            with self.group():
                obj_class = _safe_getattr(obj, "__class__", None) or type(obj)
                # First try to find registered singleton printers for the type.
                try:
                    printer = self.singleton_pprinters[obj_id]
                except (TypeError, KeyError):
                    pass
                else:
                    return printer(obj, self, cycle)

                # Look for the _repr_pretty_ method which allows users
                # to define custom pretty printing.
                # Some objects automatically create any requested
                # attribute. Try to ignore most of them by checking for
                # callability.
                pretty_method = _safe_getattr(obj, "_repr_pretty_", None)
                if callable(pretty_method):
                    return pretty_method(self, cycle)

                # Next walk the mro and check for either:
                #   1) a registered printer
                #   2) a _repr_pretty_ method
                for cls in obj_class.__mro__:
                    if cls in self.type_pprinters:
                        # printer registered in self.type_pprinters
                        return self.type_pprinters[cls](obj, self, cycle)
                    else:
                        # Check if the given class is specified in the deferred type
                        # registry; move it to the regular type registry if so.
                        key = (
                            _safe_getattr(cls, "__module__", None),
                            _safe_getattr(cls, "__name__", None),
                        )
                        if key in self.deferred_pprinters:
                            # Move the printer over to the regular registry.
                            printer = self.deferred_pprinters.pop(key)
                            self.type_pprinters[cls] = printer
                            return printer(obj, self, cycle)
                        else:
                            if hasattr(cls, "__attrs_attrs__"):
                                return pprint_fields(
                                    obj,
                                    self,
                                    cycle,
                                    [at.name for at in cls.__attrs_attrs__ if at.init],
                                )
                            if hasattr(cls, "__dataclass_fields__"):
                                return pprint_fields(
                                    obj,
                                    self,
                                    cycle,
                                    [
                                        k
                                        for k, v in cls.__dataclass_fields__.items()
                                        if v.init
                                    ],
                                )
                # Now check for object-specific printers which show how this
                # object was constructed (a Hypothesis special feature).
                printers = self.known_object_printers[IDKey(obj)]
                if len(printers) == 1:
                    return printers[0](obj, self, cycle)
                elif printers:
                    # We've ended up with multiple registered functions for the same
                    # object, which must have been returned from multiple calls due to
                    # e.g. memoization.  If they all return the same string, we'll use
                    # the first; otherwise we'll pretend that *none* were registered.
                    #
                    # It's annoying, but still seems to be the best option for which-
                    # parts-matter too, as unreportable results aren't very useful.
                    strs = set()
                    for f in printers:
                        p = RepresentationPrinter()
                        f(obj, p, cycle)
                        strs.add(p.getvalue())
                    if len(strs) == 1:
                        return printers[0](obj, self, cycle)

                # A user-provided repr. Find newlines and replace them with p.break_()
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

    def breakable(self, sep: str = " ") -> None:
        """Add a breakable separator to the output.

        This does not mean that it will automatically break here.  If no
        breaking on this position takes place the `sep` is inserted
        which default to one space.

        """
        width = len(sep)
        group = self.group_stack[-1]
        if group.want_break:
            self.flush()
            self.output.write("\n" + " " * self.indentation)
            self.output_width = self.indentation
            self.buffer_width = 0
        else:
            self.buffer.append(Breakable(sep, width, self))
            self.buffer_width += width
            self._break_outer_groups()

    def break_(self) -> None:
        """Explicitly insert a newline into the output, maintaining correct
        indentation."""
        self.flush()
        self.output.write("\n" + " " * self.indentation)
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
    def group(
        self, indent: int = 0, open: str = "", close: str = ""
    ) -> Generator[None, None, None]:
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

    def begin_group(self, indent: int = 0, open: str = "") -> None:
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

    def end_group(self, dedent: int = 0, close: str = "") -> None:
        """See begin_group()."""
        self.indentation -= dedent
        group = self.group_stack.pop()
        if not group.breakables:
            self.group_queue.remove(group)
        if close:
            self.text(close)

    def _enumerate(self, seq: Iterable[T]) -> Generator[Tuple[int, T], None, None]:
        """Like enumerate, but with an upper limit on the number of items."""
        for idx, x in enumerate(seq):
            if self.max_seq_length and idx >= self.max_seq_length:
                self.text(",")
                self.breakable()
                self.text("...")
                return
            yield idx, x

    def flush(self) -> None:
        """Flush data that is left in the buffer."""
        for data in self.buffer:
            self.output_width += data.output(self.output, self.output_width)
        self.buffer.clear()
        self.buffer_width = 0

    def getvalue(self) -> str:
        assert isinstance(self.output, StringIO)
        self.flush()
        return self.output.getvalue()

    def maybe_repr_known_object_as_call(
        self,
        obj: object,
        cycle: bool,
        name: str,
        args: Sequence[object],
        kwargs: Dict[str, object],
    ) -> None:
        # pprint this object as a call, _unless_ the call would be invalid syntax
        # and the repr would be valid and there are not comments on arguments.
        if cycle:
            return self.text("<...>")
        # Since we don't yet track comments for sub-argument parts, we omit the
        # "if no comments" condition here for now.  Add it when we revive
        # https://github.com/HypothesisWorks/hypothesis/pull/3624/
        with suppress(Exception):
            # Check whether the repr is valid syntax:
            ast.parse(repr(obj))
            # Given that the repr is valid syntax, check the call:
            p = RepresentationPrinter()
            p.stack = self.stack.copy()
            p.known_object_printers = self.known_object_printers
            p.repr_call(name, args, kwargs)
            # If the call is not valid syntax, use the repr
            try:
                ast.parse(p.getvalue())
            except Exception:
                return _repr_pprint(obj, self, cycle)
        return self.repr_call(name, args, kwargs)

    def repr_call(
        self,
        func_name: str,
        args: Sequence[object],
        kwargs: Dict[str, object],
        *,
        force_split: Optional[bool] = None,
        arg_slices: Optional[Dict[str, Tuple[int, int]]] = None,
        leading_comment: Optional[str] = None,
    ) -> None:
        """Helper function to represent a function call.

        - func_name, args, and kwargs should all be pretty obvious.
        - If split_lines, we'll force one-argument-per-line; otherwise we'll place
          calls that fit on a single line (and split otherwise).
        - arg_slices is a mapping from pos-idx or keyword to (start_idx, end_idx)
          of the Conjecture buffer, by which we can look up comments