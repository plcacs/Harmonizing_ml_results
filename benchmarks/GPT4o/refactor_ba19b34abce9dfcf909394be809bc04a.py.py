'Refactoring framework.\n\nUsed as a main program, this can refactor any number of files and/or\nrecursively descend down directories.  Imported as a module, this\nprovides infrastructure to write your own refactoring tool.\n'
__author__ = 'Guido van Rossum <guido@python.org>'
import io
import os
import pkgutil
import sys
import logging
import operator
import collections
from itertools import chain
from typing import List, Dict, Set, Tuple, Optional, Generator, Any
from .pgen2 import driver, tokenize, token
from .fixer_util import find_root
from . import pytree, pygram
from . import btm_matcher as bm

def get_all_fix_names(fixer_pkg: str, remove_prefix: bool = True) -> List[str]:
    'Return a sorted list of all available fix names in the given package.'
    pkg = __import__(fixer_pkg, [], [], ['*'])
    fix_names = []
    for (finder, name, ispkg) in pkgutil.iter_modules(pkg.__path__):
        if name.startswith('fix_'):
            if remove_prefix:
                name = name[4:]
            fix_names.append(name)
    return fix_names

class _EveryNode(Exception):
    pass

def _get_head_types(pat: pytree.BasePattern) -> Set[int]:
    ' Accepts a pytree Pattern Node and returns a set\n        of the pattern types which will match first. '
    if isinstance(pat, (pytree.NodePattern, pytree.LeafPattern)):
        if (pat.type is None):
            raise _EveryNode
        return {pat.type}
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
    raise Exception(("Oh no! I don't understand pattern %s" % pat))

def _get_headnode_dict(fixer_list: List[Any]) -> Dict[int, List[Any]]:
    ' Accepts a list of fixers and returns a dictionary\n        of head node type --> fixer list.  '
    head_nodes = collections.defaultdict(list)
    every = []
    for fixer in fixer_list:
        if fixer.pattern:
            try:
                heads = _get_head_types(fixer.pattern)
            except _EveryNode:
                every.append(fixer)
            else:
                for node_type in heads:
                    head_nodes[node_type].append(fixer)
        elif (fixer._accept_type is not None):
            head_nodes[fixer._accept_type].append(fixer)
        else:
            every.append(fixer)
    for node_type in chain(pygram.python_grammar.symbol2number.values(), pygram.python_grammar.tokens):
        head_nodes[node_type].extend(every)
    return dict(head_nodes)

def get_fixers_from_package(pkg_name: str) -> List[str]:
    '\n    Return the fully qualified names for fixers in the package pkg_name.\n    '
    return [((pkg_name + '.') + fix_name) for fix_name in get_all_fix_names(pkg_name, False)]

def _identity(obj: Any) -> Any:
    return obj

def _detect_future_features(source: str) -> Set[str]:
    have_docstring = False
    gen = tokenize.generate_tokens(io.StringIO(source).readline)

    def advance() -> Tuple[int, str]:
        tok = next(gen)
        return (tok[0], tok[1])
    ignore = frozenset({token.NEWLINE, tokenize.NL, token.COMMENT})
    features = set()
    try:
        while True:
            (tp, value) = advance()
            if (tp in ignore):
                continue
            elif (tp == token.STRING):
                if have_docstring:
                    break
                have_docstring = True
            elif ((tp == token.NAME) and (value == 'from')):
                (tp, value) = advance()
                if ((tp != token.NAME) or (value != '__future__')):
                    break
                (tp, value) = advance()
                if ((tp != token.NAME) or (value != 'import')):
                    break
                (tp, value) = advance()
                if ((tp == token.OP) and (value == '(')):
                    (tp, value) = advance()
                while (tp == token.NAME):
                    features.add(value)
                    (tp, value) = advance()
                    if ((tp != token.OP) or (value != ',')):
                        break
                    (tp, value) = advance()
            else:
                break
    except StopIteration:
        pass
    return frozenset(features)

class FixerError(Exception):
    'A fixer could not be loaded.'

class RefactoringTool(object):
    _default_options = {'print_function': False, 'exec_function': False, 'write_unchanged_files': False}
    CLASS_PREFIX = 'Fix'
    FILE_PREFIX = 'fix_'

    def __init__(self, fixer_names: List[str], options: Optional[Dict[str, bool]] = None, explicit: Optional[List[str]] = None):
        'Initializer.\n\n        Args:\n            fixer_names: a list of fixers to import\n            options: a dict with configuration.\n            explicit: a list of fixers to run even if they are explicit.\n        '
        self.fixers = fixer_names
        self.explicit = (explicit or [])
        self.options = self._default_options.copy()
        if (options is not None):
            self.options.update(options)
        self.grammar = pygram.python_grammar.copy()
        if self.options['print_function']:
            del self.grammar.keywords['print']
        elif self.options['exec_function']:
            del self.grammar.keywords['exec']
        self.write_unchanged_files = self.options.get('write_unchanged_files')
        self.errors = []
        self.logger = logging.getLogger('RefactoringTool')
        self.fixer_log = []
        self.wrote = False
        self.driver = driver.Driver(self.grammar, convert=pytree.convert, logger=self.logger)
        (self.pre_order, self.post_order) = self.get_fixers()
        self.files = []
        self.BM = bm.BottomMatcher()
        self.bmi_pre_order = []
        self.bmi_post_order = []
        for fixer in chain(self.post_order, self.pre_order):
            if fixer.BM_compatible:
                self.BM.add_fixer(fixer)
            elif (fixer in self.pre_order):
                self.bmi_pre_order.append(fixer)
            elif (fixer in self.post_order):
                self.bmi_post_order.append(fixer)
        self.bmi_pre_order_heads = _get_headnode_dict(self.bmi_pre_order)
        self.bmi_post_order_heads = _get_headnode_dict(self.bmi_post_order)

    def get_fixers(self) -> Tuple[List[Any], List[Any]]:
        'Inspects the options to load the requested patterns and handlers.\n\n        Returns:\n          (pre_order, post_order), where pre_order is the list of fixers that\n          want a pre-order AST traversal, and post_order is the list that want\n          post-order traversal.\n        '
        pre_order_fixers = []
        post_order_fixers = []
        for fix_mod_path in self.fixers:
            mod = __import__(fix_mod_path, {}, {}, ['*'])
            fix_name = fix_mod_path.rsplit('.', 1)[(- 1)]
            if fix_name.startswith(self.FILE_PREFIX):
                fix_name = fix_name[len(self.FILE_PREFIX):]
            parts = fix_name.split('_')
            class_name = (self.CLASS_PREFIX + ''.join([p.title() for p in parts]))
            try:
                fix_class = getattr(mod, class_name)
            except AttributeError:
                raise FixerError(("Can't find %s.%s" % (fix_name, class_name))) from None
            fixer = fix_class(self.options, self.fixer_log)
            if (fixer.explicit and (self.explicit is not True) and (fix_mod_path not in self.explicit)):
                self.log_message('Skipping optional fixer: %s', fix_name)
                continue
            self.log_debug('Adding transformation: %s', fix_name)
            if (fixer.order == 'pre'):
                pre_order_fixers.append(fixer)
            elif (fixer.order == 'post'):
                post_order_fixers.append(fixer)
            else:
                raise FixerError(('Illegal fixer order: %r' % fixer.order))
        key_func = operator.attrgetter('run_order')
        pre_order_fixers.sort(key=key_func)
        post_order_fixers.sort(key=key_func)
        return (pre_order_fixers, post_order_fixers)

    def log_error(self, msg: str, *args: Any, **kwds: Any) -> None:
        'Called when an error occurs.'
        raise

    def log_message(self, msg: str, *args: Any) -> None:
        'Hook to log a message.'
        if args:
            msg = (msg % args)
        self.logger.info(msg)

    def log_debug(self, msg: str, *args: Any) -> None:
        if args:
            msg = (msg % args)
        self.logger.debug(msg)

    def print_output(self, old_text: str, new_text: str, filename: str, equal: bool) -> None:
        'Called with the old version, new version, and filename of a\n        refactored file.'
        pass

    def refactor(self, items: List[str], write: bool = False, doctests_only: bool = False) -> None:
        'Refactor a list of files and directories.'
        for dir_or_file in items:
            if os.path.isdir(dir_or_file):
                self.refactor_dir(dir_or_file, write, doctests_only)
            else:
                self.refactor_file(dir_or_file, write, doctests_only)

    def refactor_dir(self, dir_name: str, write: bool = False, doctests_only: bool = False) -> None:
        "Descends down a directory and refactor every Python file found.\n\n        Python files are assumed to have a .py extension.\n\n        Files and subdirectories starting with '.' are skipped.\n        "
        py_ext = (os.extsep + 'py')
        for (dirpath, dirnames, filenames) in os.walk(dir_name):
            self.log_debug('Descending into %s', dirpath)
            dirnames.sort()
            filenames.sort()
            for name in filenames:
                if ((not name.startswith('.')) and (os.path.splitext(name)[1] == py_ext)):
                    fullname = os.path.join(dirpath, name)
                    self.refactor_file(fullname, write, doctests_only)
            dirnames[:] = [dn for dn in dirnames if (not dn.startswith('.'))]

    def _read_python_source(self, filename: str) -> Tuple[Optional[str], Optional[str]]:
        '\n        Do our best to decode a Python source file correctly.\n        '
        try:
            f = open(filename, 'rb')
        except OSError as err:
            self.log_error("Can't open %s: %s", filename, err)
            return (None, None)
        try:
            encoding = tokenize.detect_encoding(f.readline)[0]
        finally:
            f.close()
        with io.open(filename, 'r', encoding=encoding, newline='') as f:
            return (f.read(), encoding)

    def refactor_file(self, filename: str, write: bool = False, doctests_only: bool = False) -> None:
        'Refactors a file.'
        (input, encoding) = self._read_python_source(filename)
        if (input is None):
            return
        input += '\n'
        if doctests_only:
            self.log_debug('Refactoring doctests in %s', filename)
            output = self.refactor_docstring(input, filename)
            if (self.write_unchanged_files or (output != input)):
                self.processed_file(output, filename, input, write, encoding)
            else:
                self.log_debug('No doctest changes in %s', filename)
        else:
            tree = self.refactor_string(input, filename)
            if (self.write_unchanged_files or (tree and tree.was_changed)):
                self.processed_file(str(tree)[:(- 1)], filename, write=write, encoding=encoding)
            else:
                self.log_debug('No changes in %s', filename)

    def refactor_string(self, data: str, name: str) -> Optional[pytree.Node]:
        'Refactor a given input string.\n\n        Args:\n            data: a string holding the code to be refactored.\n            name: a human-readable name for use in error/log messages.\n\n        Returns:\n            An AST corresponding to the refactored input stream; None if\n            there were errors during the parse.\n        '
        features = _detect_future_features(data)
        if ('print_function' in features):
            self.driver.grammar = pygram.python_grammar_no_print_statement
        try:
            tree = self.driver.parse_string(data)
        except Exception as err:
            self.log_error("Can't parse %s: %s: %s", name, err.__class__.__name__, err)
            return
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
            if (self.write_unchanged_files or (output != input)):
                self.processed_file(output, '<stdin>', input)
            else:
                self.log_debug('No doctest changes in stdin')
        else:
            tree = self.refactor_string(input, '<stdin>')
            if (self.write_unchanged_files or (tree and tree.was_changed)):
                self.processed_file(str(tree), '<stdin>', input)
            else:
                self.log_debug('No changes in stdin')

    def refactor_tree(self, tree: pytree.Node, name: str) -> bool:
        'Refactors a parse tree (modifying the tree in place).\n\n        For compatible patterns the bottom matcher module is\n        used. Otherwise the tree is traversed node-to-node for\n        matches.\n\n        Args:\n            tree: a pytree.Node instance representing the root of the tree\n                  to be refactored.\n            name: a human-readable name for this tree.\n\n        Returns:\n            True if the tree was modified, False otherwise.\n        '
        for fixer in chain(self.pre_order, self.post_order):
            fixer.start_tree(tree, name)
        self.traverse_by(self.bmi_pre_order_heads, tree.pre_order())
        self.traverse_by(self.bmi_post_order_heads, tree.post_order())
        match_set = self.BM.run(tree.leaves())
        while any(match_set.values()):
            for fixer in self.BM.fixers:
                if ((fixer in match_set) and match_set[fixer]):
                    match_set[fixer].sort(key=pytree.Base.depth, reverse=True)
                    if fixer.keep_line_order:
                        match_set[fixer].sort(key=pytree.Base.get_lineno)
                    for node in list(match_set[fixer]):
                        if (node in match_set[fixer]):
                            match_set[fixer].remove(node)
                        try:
                            find_root(node)
                        except ValueError:
                            continue
                        if (node.fixers_applied and (fixer in node.fixers_applied)):
                            continue
                        results = fixer.match(node)
                        if results:
                            new = fixer.transform(node, results)
                            if (new is not None):
                                node.replace(new)
                                for node in new.post_order():
                                    if (not node.fixers_applied):
                                        node.fixers_applied = []
                                    node.fixers_applied.append(fixer)
                                new_matches = self.BM.run(new.leaves())
                                for fxr in new_matches:
                                    if (not (fxr in match_set)):
                                        match_set[fxr] = []
                                    match_set[fxr].extend(new_matches[fxr])
        for fixer in chain(self.pre_order, self.post_order):
            fixer.finish_tree(tree, name)
        return tree.was_changed

    def traverse_by(self, fixers: Dict[int, List[Any]], traversal: Generator[pytree.Node, None, None]) -> None:
        'Traverse an AST, applying a set of fixers to each node.\n\n        This is a helper method for refactor_tree().\n\n        Args:\n            fixers: a list of fixer instances.\n            traversal: a generator that yields AST nodes.\n\n        Returns:\n            None\n        '
        if (not fixers):
            return
        for node in traversal:
            for fixer in fixers[node.type]:
                results = fixer.match(node)
                if results:
                    new = fixer.transform(node, results)
                    if (new is not None):
                        node.replace(new)
                        node = new

    def processed_file(self, new_text: str, filename: str, old_text: Optional[str] = None, write: bool = False, encoding: Optional[str] = None) -> None:
        '\n        Called when a file has been refactored and there may be changes.\n        '
        self.files.append(filename)
        if (old_text is None):
            old_text = self._read_python_source(filename)[0]
            if (old_text is None):
                return
        equal = (old_text == new_text)
        self.print_output(old_text, new_text, filename, equal)
        if equal:
            self.log_debug('No changes to %s', filename)
            if (not self.write_unchanged_files):
                return
        if write:
            self.write_file(new_text, filename, old_text, encoding)
        else:
            self.log_debug('Not writing changes to %s', filename)

    def write_file(self, new_text: str, filename: str, old_text: str, encoding: Optional[str] = None) -> None:
        'Writes a string to a file.\n\n        It first shows a unified diff between the old text and the new text, and\n        then rewrites the file; the latter is only done if the write option is\n        set.\n        '
        try:
            fp = io.open(filename, 'w', encoding=encoding, newline='')
        except OSError as err:
            self.log_error("Can't create %s: %s", filename, err)
            return
        with fp:
            try:
                fp.write(new_text)
            except OSError as err:
                self.log_error("Can't write %s: %s", filename, err)
        self.log_debug('Wrote changes to %s', filename)
        self.wrote = True
    PS1 = '>>> '
    PS2 = '... '

    def refactor_docstring(self, input: str, filename: str) -> str:
        'Refactors a docstring, looking for doctests.\n\n        This returns a modified version of the input string.  It looks\n        for doctests, which start with a ">>>" prompt, and may be\n        continued with "..." prompts, as long as the "..." is indented\n        the same as the ">>>".\n\n        (Unfortunately we can\'t use the doctest module\'s parser,\n        since, like most parsers, it is not geared towards preserving\n        the original source.)\n        '
        result = []
        block = None
        block_lineno = None
        indent = None
        lineno = 0
        for line in input.splitlines(keepends=True):
            lineno += 1
            if line.lstrip().startswith(self.PS1):
                if (block is not None):
                    result.extend(self.refactor_doctest(block, block_lineno, indent, filename))
                block_lineno = lineno
                block = [line]
                i = line.find(self.PS1)
                indent = line[:i]
            elif ((indent is not None) and (line.startswith((indent + self.PS2)) or (line == ((indent + self.PS2.rstrip()) + '\n')))):
                block.append(line)
            else:
                if (block is not None):
                    result.extend(self.refactor_doctest(block, block_lineno, indent, filename))
                block = None
                indent = None
                result.append(line)
        if (block is not None):
            result.extend(self.refactor_doctest(block, block_lineno, indent, filename))
        return ''.join(result)

    def refactor_doctest(self, block: List[str], lineno: int, indent: str, filename: str) -> List[str]:
        'Refactors one doctest.\n\n        A doctest is given as a block of lines, the first of which starts\n        with ">>>" (possibly indented), while the remaining lines start\n        with "..." (identically indented).\n\n        '
        try:
            tree = self.parse_block(block, lineno, indent)
        except Exception as err:
            if self.logger.isEnabledFor(logging.DEBUG):
                for line in block:
                    self.log_debug('Source: %s', line.rstrip('\n'))
            self.log_error("Can't parse docstring in %s line %s: %s: %s", filename, lineno, err.__class__.__name__, err)
            return block
        if self.refactor_tree(tree, filename):
            new = str(tree).splitlines(keepends=True)
            (clipped, new) = (new[:(lineno - 1)], new[(lineno - 1):])
            assert (clipped == (['\n'] * (lineno - 1))), clipped
            if (not new[(- 1)].endswith('\n')):
                new[(- 1)] += '\n'
            block = [((indent + self.PS1) + new.pop(0))]
            if new:
                block += [((indent + self.PS2) + line) for line in new]
        return block

    def summarize(self) -> None:
        if self.wrote:
            were = 'were'
        else:
            were = 'need to be'
        if (not self.files):
            self.log_message('No files %s modified.', were)
        else:
            self.log_message('Files that %s modified:', were)
            for file in self.files:
                self.log_message(file)
        if self.fixer_log:
            self.log_message('Warnings/messages while refactoring:')
            for message in self.fixer_log:
                self.log_message(message)
        if self.errors:
            if (len(self.errors) == 1):
                self.log_message('There was 1 error:')
            else:
                self.log_message('There were %d errors:', len(self.errors))
            for (msg, args, kwds) in self.errors:
                self.log_message(msg, *args, **kwds)

    def parse_block(self, block: List[str], lineno: int, indent: str) -> pytree.Node:
        'Parses a block into a tree.\n\n        This is necessary to get correct line number / offset information\n        in the parser diagnostics and embedded into the parse tree.\n        '
        tree = self.driver.parse_tokens(self.wrap_toks(block, lineno, indent))
        tree.future_features = frozenset()
        return tree

    def wrap_toks(self, block: List[str], lineno: int, indent: str) -> Generator[Tuple[int, str, Tuple[int, int], Tuple[int, int], str], None, None]:
        'Wraps a tokenize stream to systematically modify start/end.'
        tokens = tokenize.generate_tokens(self.gen_lines(block, indent).__next__)
        for (type, value, (line0, col0), (line1, col1), line_text) in tokens:
            line0 += (lineno - 1)
            line1 += (lineno - 1)
            (yield (type, value, (line0, col0), (line1, col1), line_text))

    def gen_lines(self, block: List[str], indent: str) -> Generator[str, None, None]:
        'Generates lines as expected by tokenize from a list of lines.\n\n        This strips the first len(indent + self.PS1) characters off each line.\n        '
        prefix1 = (indent + self.PS1)
        prefix2 = (indent + self.PS2)
        prefix = prefix1
        for line in block:
            if line.startswith(prefix):
                (yield line[len(prefix):])
            elif (line == (prefix.rstrip() + '\n')):
                (yield '\n')
            else:
                raise AssertionError(('line=%r, prefix=%r' % (line, prefix)))
            prefix = prefix2
        while True:
            (yield '')

class MultiprocessingUnsupported(Exception):
    pass

class MultiprocessRefactoringTool(RefactoringTool):

    def __init__(self, *args: Any, **kwargs: Any):
        super(MultiprocessRefactoringTool, self).__init__(*args, **kwargs)
        self.queue = None
        self.output_lock = None

    def refactor(self, items: List[str], write: bool = False, doctests_only: bool = False, num_processes: int = 1) -> None:
        if (num_processes == 1):
            return super(MultiprocessRefactoringTool, self).refactor(items, write, doctests_only)
        try:
            import multiprocessing
        except ImportError:
            raise MultiprocessingUnsupported
        if (self.queue is not None):
            raise RuntimeError('already doing multiple processes')
        self.queue = multiprocessing.JoinableQueue()
        self.output_lock = multiprocessing.Lock()
        processes = [multiprocessing.Process(target=self._child) for i in range(num_processes)]
        try:
            for p in processes:
                p.start()
            super(MultiprocessRefactoringTool, self).refactor(items, write, doctests_only)
        finally:
            self.queue.join()
            for i in range(num_processes):
                self.queue.put(None)
            for p in processes:
                if p.is_alive():
                    p.join()
            self.queue = None

    def _child(self) -> None:
        task = self.queue.get()
        while (task is not None):
            (args, kwargs) = task
            try:
                super(MultiprocessRefactoringTool, self).refactor_file(*args, **kwargs)
            finally:
                self.queue.task_done()
            task = self.queue.get()

    def refactor_file(self, *args: Any, **kwargs: Any) -> None:
        if (self.queue is not None):
            self.queue.put((args, kwargs))
        else:
            return super(MultiprocessRefactoringTool, self).refactor_file(*args, **kwargs)
