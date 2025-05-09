from typing import (
    Any, Dict, List, Set, Tuple, Optional, Union, Callable, Iterator, 
    Generator, DefaultDict, FrozenSet, TypeVar, Generic, Sequence, Iterable
)
import io
import os
import pkgutil
import sys
import logging
import operator
import collections
from itertools import chain
from .pgen2 import driver, tokenize, token
from .fixer_util import find_root
from . import pytree, pygram
from . import btm_matcher as bm

T = TypeVar('T')
Pattern = Union[pytree.NodePattern, pytree.LeafPattern, pytree.NegatedPattern, pytree.WildcardPattern]

def get_all_fix_names(fixer_pkg: str, remove_prefix: bool = True) -> List[str]:
    pkg = __import__(fixer_pkg, [], [], ['*'])
    fix_names: List[str] = []
    for (finder, name, ispkg) in pkgutil.iter_modules(pkg.__path__):
        if name.startswith('fix_'):
            if remove_prefix:
                name = name[4:]
            fix_names.append(name)
    return fix_names

class _EveryNode(Exception):
    pass

def _get_head_types(pat: Pattern) -> Set[int]:
    if isinstance(pat, (pytree.NodePattern, pytree.LeafPattern)):
        if pat.type is None:
            raise _EveryNode
        return {pat.type}
    if isinstance(pat, pytree.NegatedPattern):
        if pat.content:
            return _get_head_types(pat.content)
        raise _EveryNode
    if isinstance(pat, pytree.WildcardPattern):
        r: Set[int] = set()
        for p in pat.content:
            for x in p:
                r.update(_get_head_types(x))
        return r
    raise Exception(f"Oh no! I don't understand pattern {pat}")

def _get_headnode_dict(fixer_list: List[Any]) -> Dict[int, List[Any]]:
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
        elif fixer._accept_type is not None:
            head_nodes[fixer._accept_type].append(fixer)
        else:
            every.append(fixer)
    for node_type in chain(pygram.python_grammar.symbol2number.values(), pygram.python_grammar.tokens):
        head_nodes[node_type].extend(every)
    return dict(head_nodes)

def get_fixers_from_package(pkg_name: str) -> List[str]:
    return [(pkg_name + '.' + fix_name) for fix_name in get_all_fix_names(pkg_name, False)]

def _identity(obj: T) -> T:
    return obj

def _detect_future_features(source: str) -> FrozenSet[str]:
    have_docstring = False
    gen = tokenize.generate_tokens(io.StringIO(source).readline)

    def advance() -> Tuple[int, str]:
        tok = next(gen)
        return (tok[0], tok[1])
    
    ignore = frozenset({token.NEWLINE, tokenize.NL, token.COMMENT})
    features: Set[str] = set()
    try:
        while True:
            tp, value = advance()
            if tp in ignore:
                continue
            elif tp == token.STRING:
                if have_docstring:
                    break
                have_docstring = True
            elif (tp == token.NAME) and (value == 'from'):
                tp, value = advance()
                if (tp != token.NAME) or (value != '__future__'):
                    break
                tp, value = advance()
                if (tp != token.NAME) or (value != 'import'):
                    break
                tp, value = advance()
                if (tp == token.OP) and (value == '('):
                    tp, value = advance()
                while tp == token.NAME:
                    features.add(value)
                    tp, value = advance()
                    if (tp != token.OP) or (value != ','):
                        break
                    tp, value = advance()
            else:
                break
    except StopIteration:
        pass
    return frozenset(features)

class FixerError(Exception):
    pass

class RefactoringTool:
    _default_options: Dict[str, bool] = {
        'print_function': False,
        'exec_function': False,
        'write_unchanged_files': False
    }
    CLASS_PREFIX: str = 'Fix'
    FILE_PREFIX: str = 'fix_'

    def __init__(
        self,
        fixer_names: List[str],
        options: Optional[Dict[str, bool]] = None,
        explicit: Optional[List[str]] = None
    ) -> None:
        self.fixers: List[str] = fixer_names
        self.explicit: List[str] = explicit or []
        self.options: Dict[str, bool] = self._default_options.copy()
        if options is not None:
            self.options.update(options)
        self.grammar = pygram.python_grammar.copy()
        if self.options['print_function']:
            del self.grammar.keywords['print']
        elif self.options['exec_function']:
            del self.grammar.keywords['exec']
        self.write_unchanged_files: bool = self.options.get('write_unchanged_files', False)
        self.errors: List[Tuple[str, Tuple[Any, ...], Dict[str, Any]]] = []
        self.logger: logging.Logger = logging.getLogger('RefactoringTool')
        self.fixer_log: List[str] = []
        self.wrote: bool = False
        self.driver: driver.Driver = driver.Driver(
            self.grammar,
            convert=pytree.convert,
            logger=self.logger
        )
        self.pre_order: List[Any]
        self.post_order: List[Any]
        self.pre_order, self.post_order = self.get_fixers()
        self.files: List[str] = []
        self.BM: bm.BottomMatcher = bm.BottomMatcher()
        self.bmi_pre_order: List[Any] = []
        self.bmi_post_order: List[Any] = []
        for fixer in chain(self.post_order, self.pre_order):
            if fixer.BM_compatible:
                self.BM.add_fixer(fixer)
            elif fixer in self.pre_order:
                self.bmi_pre_order.append(fixer)
            elif fixer in self.post_order:
                self.bmi_post_order.append(fixer)
        self.bmi_pre_order_heads: Dict[int, List[Any]] = _get_headnode_dict(self.bmi_pre_order)
        self.bmi_post_order_heads: Dict[int, List[Any]] = _get_headnode_dict(self.bmi_post_order)

    def get_fixers(self) -> Tuple[List[Any], List[Any]]:
        pre_order_fixers: List[Any] = []
        post_order_fixers: List[Any] = []
        for fix_mod_path in self.fixers:
            mod = __import__(fix_mod_path, {}, {}, ['*'])
            fix_name = fix_mod_path.rsplit('.', 1)[-1]
            if fix_name.startswith(self.FILE_PREFIX):
                fix_name = fix_name[len(self.FILE_PREFIX):]
            parts = fix_name.split('_')
            class_name = self.CLASS_PREFIX + ''.join([p.title() for p in parts])
            try:
                fix_class = getattr(mod, class_name)
            except AttributeError:
                raise FixerError(f"Can't find {fix_name}.{class_name}") from None
            fixer = fix_class(self.options, self.fixer_log)
            if fixer.explicit and (self.explicit is not True) and (fix_mod_path not in self.explicit):
                self.log_message('Skipping optional fixer: %s', fix_name)
                continue
            self.log_debug('Adding transformation: %s', fix_name)
            if fixer.order == 'pre':
                pre_order_fixers.append(fixer)
            elif fixer.order == 'post':
                post_order_fixers.append(fixer)
            else:
                raise FixerError(f'Illegal fixer order: {fixer.order!r}')
        key_func = operator.attrgetter('run_order')
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
        py_ext = os.extsep + 'py'
        for dirpath, dirnames, filenames in os.walk(dir_name):
            self.log_debug('Descending into %s', dirpath)
            dirnames.sort()
            filenames.sort()
            for name in filenames:
                if (not name.startswith('.')) and (os.path.splitext(name)[1] == py_ext):
                    fullname = os.path.join(dirpath, name)
                    self.refactor_file(fullname, write, doctests_only)
            dirnames[:] = [dn for dn in dirnames if not dn.startswith('.')]

    def _read_python_source(self, filename: str) -> Tuple[Optional[str], Optional[str]]:
        try:
            with open(filename, 'rb') as f:
                encoding = tokenize.detect_encoding(f.readline)[0]
        except OSError as err:
            self.log_error("Can't open %s: %s", filename, err)
            return (None, None)
        try:
            with io.open(filename, 'r', encoding=encoding, newline='') as f:
                return (f.read(), encoding)
        except OSError:
            return (None, None)

    def refactor_file(self, filename: str, write: bool = False, doctests_only: bool = False) -> None:
        input, encoding = self._read_python_source(filename)
        if input is None:
            return
        input += '\n'
        if doctests_only:
            self.log_debug('Refactoring doctests in %s', filename)
            output = self.refactor_docstring(input, filename)
            if self.write_unchanged_files or (output != input):
                self.processed_file(output, filename, input, write, encoding)
            else:
                self.log_debug('No doctest changes in %s', filename)
        else:
            tree = self.refactor_string(input, filename)
            if self.write_unchanged_files or (tree and tree.was_changed):
                self.processed_file(str(tree)[:-1], filename, write=write, encoding=encoding)
            else:
                self.log_debug('No changes in %s', filename)

    def refactor_string(self, data: str, name: str) -> Optional[pytree.Base]:
        features = _detect_future_features(data)
        if 'print_function' in features:
            self.driver.grammar = pygram.python_grammar_no_print_statement
        try:
            tree = self.driver.parse_string(data)
        except Exception as err:
            self.log_error("Can't parse %s: %s: %s", name, err.__class__.__name__, err)
            return None
        finally:
            self.driver.grammar = self.grammar
        tree.future_features = features
        self.log_debug('Refactoring %s', name)
        self.refactor_tree(tree, name)
        return tree

    def refactor_stdin(self, doctests_only: bool = False) -> None:
        input = sys.stdin.read()
        if doctests_only:
            self.log_debug('Refactoring doctests in stdin')
            output = self.refactor_docstring(input, '<stdin>')
            if self.write_unchanged_files or (output != input):
                self.processed_file(output, '<stdin>', input)
            else:
                self.log_debug('No doctest changes in stdin')
        else:
            tree = self.refactor_string(input, '<stdin>')
            if self.write_unchanged_files or (tree and tree.was_changed):
                self.processed_file(str(tree), '<stdin>', input)
            else:
                self.log_debug('No changes in stdin')

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
                        if node.fixers_applied and (fixer in node.fixers_applied):
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
                                    if fxr not in match_set:
                                        match_set[fxr] = []
                                    match_set[fxr].extend(new_matches[fxr])
        for fixer in chain(self.pre_order, self.post_order):
            fixer.finish_tree(tree, name)
        return tree.was_changed

    def traverse_by(self, fixers: Dict[int, List[Any]], traversal: Iterator[pytree.Base]) -> None:
        if not fixers:
            return
        for node in traversal:
            for fixer in fixers.get(node.type, []):
                results = fixer.match(node)
                if results:
                    new = fixer.transform(node, results)
                    if new is not None:
                        node.replace(new)
                        node = new

    def processed_file(
        self,
        new_text: str,
        filename: str,
        old_text: Optional[str] = None,
        write: bool = False,
        encoding: Optional[str] = None
    ) -> None:
        self.files.append(filename)
        if old_text is None:
            old_text_result = self._read_python_source(filename)
            if old_text_result[0] is None:
                return
            old_text = old_text_result[0]
        equal = old_text == new_text
        self.print_output(old_text, new_text, filename, equal)
        if equal:
            self.log_debug('No changes to %s', filename)
            if not self.write_unchanged_files:
                return
        if write:
            self.write_file(new_text, filename, old_text, encoding)
        else:
            self.log_debug('Not writing changes to %s', filename)

    def write_file(
        self,
        new_text: str,
        filename: str,
        old_text: str,
        encoding: Optional[str] = None
    ) -> None:
        try:
            with io.open(filename, 'w', encoding=encoding, newline='') as fp:
                fp.write(new_text)
        except OSError as err:
            self.log_error("Can't write %s: %s", filename, err)
            return
        self.log_debug('Wrote changes to %s', filename)
        self.wrote = True

    PS1: str = '>>> '
    PS2: str = '... '

    def refactor_docstring(self, input: str, filename: str) -> str:
        result: List[str] = []
        block: Optional[List[str]] = None
        block_lineno: Optional[int] = None
        indent: Optional[str] = None
        lineno = 0
        for line in input.splitlines(keepends=True):
            lineno += 1
            if line.lstrip().startswith(self.PS1):
                if block is not None:
                    result.extend(self.refactor_doctest(block, block_lineno, indent, filename))
                block_lineno = lineno
                block = [line]
                i = line.find(self.PS1)
                indent = line[:i]
            elif (indent is not None) and (line.startswith(indent + self.PS2) or line == (indent + self.PS2.rstrip() + '\n')):
                block.append(line)
            else:
                if block is not None:
                    result.extend(self.refactor_doctest(block, block_lineno, indent, filename))
                block = None
                indent = None
                result.append(line)
        if block is not None:
            result.extend(self.refactor_doctest(block, block_lineno, indent, filename))
        return ''.