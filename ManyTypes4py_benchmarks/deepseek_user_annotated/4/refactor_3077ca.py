# Copyright 2006 Google, Inc. All Rights Reserved.
# Licensed to PSF under a Contributor Agreement.

"""Refactoring framework.

Used as a main program, this can refactor any number of files and/or
recursively descend down directories.  Imported as a module, this
provides infrastructure to write your own refactoring tool.
"""

from __future__ import with_statement
from typing import (
    Any, Dict, List, Set, Tuple, Optional, Union, Iterator, Callable,
    DefaultDict, Sequence, Iterable, TextIO, TypeVar, Generic, FrozenSet
)

__author__ = "Guido van Rossum <guido@python.org>"


# Python imports
import os
import sys
import logging
import operator
import collections
import io
from itertools import chain

# Local imports
from .pgen2 import driver, tokenize, token
from .fixer_util import find_root
from . import pytree, pygram
from . import btm_utils as bu
from . import btm_matcher as bm

T = TypeVar('T')
Pattern = Union[pytree.LeafPattern, pytree.NodePattern, pytree.NegatedPattern, pytree.WildcardPattern]

def get_all_fix_names(fixer_pkg: str, remove_prefix: bool = True) -> List[str]:
    """Return a sorted list of all available fix names in the given package."""
    pkg = __import__(fixer_pkg, [], [], ["*"])
    fixer_dir = os.path.dirname(pkg.__file__)
    fix_names = []
    for name in sorted(os.listdir(fixer_dir)):
        if name.startswith("fix_") and name.endswith(".py"):
            if remove_prefix:
                name = name[4:]
            fix_names.append(name[:-3])
    return fix_names


class _EveryNode(Exception):
    pass


def _get_head_types(pat: Pattern) -> Set[int]:
    """ Accepts a pytree Pattern Node and returns a set
        of the pattern types which will match first. """

    if isinstance(pat, (pytree.NodePattern, pytree.LeafPattern)):
        if pat.type is None:
            raise _EveryNode
        return set([pat.type])

    if isinstance(pat, pytree.NegatedPattern):
        if pat.content:
            return _get_head_types(pat.content)
        raise _EveryNode

    if isinstance(pat, pytree.WildcardPattern):
        r = set()
        for p in pat.content:
            for x in p:
                r.update(_get_head_types(x))
        return r

    raise Exception("Oh no! I don't understand pattern %s" %(pat))


def _get_headnode_dict(fixer_list: List[Any]) -> Dict[int, List[Any]]:
    """ Accepts a list of fixers and returns a dictionary
        of head node type --> fixer list.  """
    head_nodes: DefaultDict[int, List[Any]] = collections.defaultdict(list)
    every: List[Any] = []
    for fixer in fixer_list:
        if fixer.pattern:
            try:
                heads = _get_head_types(fixer.pattern)
            except _EveryNode:
                every.append(fixer)
            else:
                for node_type in heads:
                    head_nodes[node_type].append(fixer)
        else:
            if fixer._accept_type is not None:
                head_nodes[fixer._accept_type].append(fixer)
            else:
                every.append(fixer)
    for node_type in chain(pygram.python_grammar.symbol2number.values(),
                           pygram.python_grammar.tokens):
        head_nodes[node_type].extend(every)
    return dict(head_nodes)


def get_fixers_from_package(pkg_name: str) -> List[str]:
    """
    Return the fully qualified names for fixers in the package pkg_name.
    """
    return [pkg_name + "." + fix_name
            for fix_name in get_all_fix_names(pkg_name, False)]

def _identity(obj: T) -> T:
    return obj

if sys.version_info < (3, 0):
    import codecs
    _open_with_encoding = codecs.open
    def _from_system_newlines(input: str) -> str:
        return input.replace("\r\n", "\n")
    def _to_system_newlines(input: str) -> str:
        if os.linesep != "\n":
            return input.replace("\n", os.linesep)
        else:
            return input
else:
    _open_with_encoding = open
    _from_system_newlines = _identity
    _to_system_newlines = _identity


def _detect_future_features(source: str) -> FrozenSet[str]:
    have_docstring = False
    gen = tokenize.generate_tokens(io.StringIO(source).readline)
    def advance() -> Tuple[int, str]:
        tok = next(gen)
        return tok[0], tok[1]
    ignore = frozenset((token.NEWLINE, tokenize.NL, token.COMMENT))
    features = set()
    try:
        while True:
            tp, value = advance()
            if tp in ignore:
                continue
            elif tp == token.STRING:
                if have_docstring:
                    break
                have_docstring = True
            elif tp == token.NAME and value == "from":
                tp, value = advance()
                if tp != token.NAME or value != "__future__":
                    break
                tp, value = advance()
                if tp != token.NAME or value != "import":
                    break
                tp, value = advance()
                if tp == token.OP and value == "(":
                    tp, value = advance()
                while tp == token.NAME:
                    features.add(value)
                    tp, value = advance()
                    if tp != token.OP or value != ",":
                        break
                    tp, value = advance()
            else:
                break
    except StopIteration:
        pass
    return frozenset(features)


class FixerError(Exception):
    """A fixer could not be loaded."""


class RefactoringTool(object):

    _default_options = {"print_function" : False,
                        "write_unchanged_files" : False}

    CLASS_PREFIX = "Fix" # The prefix for fixer classes
    FILE_PREFIX = "fix_" # The prefix for modules with a fixer within

    def __init__(self, fixer_names: List[str], options: Optional[Dict[str, Any]] = None, explicit: Optional[List[str]] = None) -> None:
        self.fixers = fixer_names
        self.explicit = explicit or []
        self.options = self._default_options.copy()
        if options is not None:
            self.options.update(options)
        if self.options["print_function"]:
            self.grammar = pygram.python_grammar_no_print_statement
        else:
            self.grammar = pygram.python_grammar
        self.write_unchanged_files = self.options.get("write_unchanged_files")
        self.errors: List[Tuple[str, Tuple[Any, ...], Dict[str, Any]]] = []
        self.logger = logging.getLogger("RefactoringTool")
        self.fixer_log: List[str] = []
        self.wrote = False
        self.driver = driver.Driver(self.grammar,
                                    convert=pytree.convert,
                                    logger=self.logger)
        self.pre_order, self.post_order = self.get_fixers()

        self.files: List[str] = []

        self.BM = bm.BottomMatcher()
        self.bmi_pre_order: List[Any] = []
        self.bmi_post_order: List[Any] = []

        for fixer in chain(self.post_order, self.pre_order):
            if fixer.BM_compatible:
                self.BM.add_fixer(fixer)
            elif fixer in self.pre_order:
                self.bmi_pre_order.append(fixer)
            elif fixer in self.post_order:
                self.bmi_post_order.append(fixer)

        self.bmi_pre_order_heads = _get_headnode_dict(self.bmi_pre_order)
        self.bmi_post_order_heads = _get_headnode_dict(self.bmi_post_order)

    def get_fixers(self) -> Tuple[List[Any], List[Any]]:
        pre_order_fixers: List[Any] = []
        post_order_fixers: List[Any] = []
        for fix_mod_path in self.fixers:
            mod = __import__(fix_mod_path, {}, {}, ["*"])
            fix_name = fix_mod_path.rsplit(".", 1)[-1]
            if fix_name.startswith(self.FILE_PREFIX):
                fix_name = fix_name[len(self.FILE_PREFIX):]
            parts = fix_name.split("_")
            class_name = self.CLASS_PREFIX + "".join([p.title() for p in parts])
            try:
                fix_class = getattr(mod, class_name)
            except AttributeError:
                raise FixerError("Can't find %s.%s" % (fix_name, class_name))
            fixer = fix_class(self.options, self.fixer_log)
            if fixer.explicit and self.explicit is not True and \
                    fix_mod_path not in self.explicit:
                self.log_message("Skipping implicit fixer: %s", fix_name)
                continue

            self.log_debug("Adding transformation: %s", fix_name)
            if fixer.order == "pre":
                pre_order_fixers.append(fixer)
            elif fixer.order == "post":
                post_order_fixers.append(fixer)
            else:
                raise FixerError("Illegal fixer order: %r" % fixer.order)

        key_func = operator.attrgetter("run_order")
        pre_order_fixers.sort(key=key_func)
        post_order_fixers.sort(key=key_func)
        return (pre_order_fixers, post_order_fixers)

    def log_error(self, msg: str, *args: Any, **kwds: Any) -> None:
        raise

    def log_message(self, msg: str, *args: Any) -> None:
        if args:
            msg = msg % args
        self.logger.info(msg)

    def log_debug(self, msg: str, *args: Any) -> None:
        if args:
            msg = msg % args
        self.logger.debug(msg)

    def print_output(self, old_text: str, new_text: str, filename: str, equal: bool) -> None:
        pass

    def refactor(self, items: List[str], write: bool = False, doctests_only: bool = False) -> None:
        for dir_or_file in items:
            if os.path.isdir(dir_or_file):
                self.refactor_dir(dir_or_file, write, doctests_only)
            else:
                self.refactor_file(dir_or_file, write, doctests_only)

    def refactor_dir(self, dir_name: str, write: bool = False, doctests_only: bool = False) -> None:
        py_ext = os.extsep + "py"
        for dirpath, dirnames, filenames in os.walk(dir_name):
            self.log_debug("Descending into %s", dirpath)
            dirnames.sort()
            filenames.sort()
            for name in filenames:
                if (not name.startswith(".") and
                    os.path.splitext(name)[1] == py_ext):
                    fullname = os.path.join(dirpath, name)
                    self.refactor_file(fullname, write, doctests_only)
            dirnames[:] = [dn for dn in dirnames if not dn.startswith(".")]

    def _read_python_source(self, filename: str) -> Tuple[Optional[str], Optional[str]]:
        try:
            f = open(filename, "rb")
        except IOError as err:
            self.log_error("Can't open %s: %s", filename, err)
            return None, None
        try:
            encoding = tokenize.detect_encoding(f.readline)[0]
        finally:
            f.close()
        with _open_with_encoding(filename, "r", encoding=encoding) as f:
            return _from_system_newlines(f.read()), encoding

    def refactor_file(self, filename: str, write: bool = False, doctests_only: bool = False) -> None:
        input, encoding = self._read_python_source(filename)
        if input is None:
            return
        input += "\n"
        if doctests_only:
            self.log_debug("Refactoring doctests in %s", filename)
            output = self.refactor_docstring(input, filename)
            if self.write_unchanged_files or output != input:
                self.processed_file(output, filename, input, write, encoding)
            else:
                self.log_debug("No doctest changes in %s", filename)
        else:
            tree = self.refactor_string(input, filename)
            if self.write_unchanged_files or (tree and tree.was_changed):
                self.processed_file(str(tree)[:-1], filename,
                                    write=write, encoding=encoding)
            else:
                self.log_debug("No changes in %s", filename)

    def refactor_string(self, data: str, name: str) -> Optional[pytree.Base]:
        features = _detect_future_features(data)
        if "print_function" in features:
            self.driver.grammar = pygram.python_grammar_no_print_statement
        try:
            tree = self.driver.parse_string(data)
        except Exception as err:
            self.log_error("Can't parse %s: %s: %s",
                           name, err.__class__.__name__, err)
            return None
        finally:
            self.driver.grammar = self.grammar
        tree.future_features = features
        self.log_debug("Refactoring %s", name)
        self.refactor_tree(tree, name)
        return tree

    def refactor_stdin(self, doctests_only: bool = False) -> None:
        input = sys.stdin.read()
        if doctests_only:
            self.log_debug("Refactoring doctests in stdin")
            output = self.refactor_docstring(input, "<stdin>")
            if self.write_unchanged_files or output != input:
                self.processed_file(output, "<stdin>", input)
            else:
                self.log_debug("No doctest changes in stdin")
        else:
            tree = self.refactor_string(input, "<stdin>")
            if self.write_unchanged_files or (tree and tree.was_changed):
                self.processed_file(str(tree), "<stdin>", input)
            else:
                self.log_debug("No changes in stdin")

    def refactor_tree(self, tree: pytree.Base, name: str) -> bool:
        for fixer in chain(self.pre_order, self.post_order):
            fixer.start_tree(tree, name)

        self.traverse_by(self.bmi_pre_order_heads, tree.pre_order())
        self.traverse_by(self.bmi_post_order_heads, tree.post_order())

        match_set = self.BM.run(tree.leaves())

        while any(match_set.values()):
            for fixer in self.BM.fixers:
                if fixer in match_set and match_set[fixer]:
                    match_set[fixer].sort(key=pytree.Base.depth, reverse=True)

                    if fixer.keep_line_order:
                        match_set[fixer].sort(key=pytree.Base.get_lineno)

                    for node in list(match_set[fixer]):
                        if node in match_set[fixer]:
                            match_set[fixer].remove(node)

                        try:
                            find_root(node)
                        except ValueError:
                            continue

                        if node.fixers_applied and fixer in node.fixers_applied:
                            continue

                        results = fixer.match(node)

                        if results:
                            new = fixer.transform(node, results)
                            if new is not None:
                                node.replace(new)
                                for node in new.post_order():
                                    if not node.fixers_applied:
                                        node.fixers_applied = []
                                    node.fixers_applied.append(fixer)

                                new_matches = self.BM.run(new.leaves())
                                for fxr in new_matches:
                                    if not fxr in match_set:
                                        match_set[fxr]=[]
                                    match_set[fxr].extend(new_matches[fxr])

        for fixer in chain(self.pre_order, self.post_order):
            fixer.finish_tree(tree, name)
        return tree.was_changed

    def traverse_by(self, fixers: Dict[int, List[Any]], traversal: Iterator[pytree.Base]) -> None:
        if not fixers:
            return
        for node in traversal:
            for fixer in fixers[node.type]:
                results = fixer.match(node)
                if results:
                    new = fixer.transform(node, results)
                    if new is not None:
                        node.replace(new)
                        node = new

    def processed_file(self, new_text: str, filename: str, old_text: Optional[str] = None, write: bool = False,
                       encoding: Optional[str] = None) -> None:
        self.files.append(filename)
        if old_text is None:
            old_text = self._read_python_source(filename)[0]
            if old_text is None:
                return
        equal = old_text == new_text
        self.print_output(old_text, new_text, filename, equal)
        if equal:
            self.log_debug("No changes to %s", filename)
            if not self.write_unchanged_files:
                return
        if write:
            self.write_file(new_text, filename, old_text, encoding)
        else:
            self.log_debug("Not writing changes to %s", filename)

    def write_file(self, new_text: str, filename: str, old_text: str, encoding: Optional[str] = None) -> None:
        try:
            f = _open_with_encoding(filename, "w", encoding=encoding)
        except os.error as err:
            self.log_error("Can't create %s: %s", filename, err)
            return
        try:
            f.write(_to_system_newlines(new_text))
        except os.error as err:
            self.log_error("Can't write %s: %s", filename, err)
        finally:
            f.close()
        self.log_debug("Wrote changes to %s", filename)
        self.wrote = True

    PS1 = ">>> "
    PS2 = "... "

    def refactor_docstring(self, input: str, filename: str) -> str:
        result = []
        block = None
        block_lineno = None
        indent = None
        lineno = 