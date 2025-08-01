import hashlib
import os
from typing import Generic, TypeVar, Union, Dict, Optional, Any, List, Iterator, Tuple, Set, Type, Callable, Iterable
from pathlib import Path
from parso._compatibility import is_pypy
from parso.pgen2 import generate_grammar
from parso.utils import split_lines, python_bytes_to_unicode, PythonVersionInfo, parse_version_string
from parso.python.diff import DiffParser
from parso.python.tokenize import tokenize_lines, tokenize
from parso.python.token import PythonTokenTypes
from parso.cache import parser_cache, load_module, try_to_save_module
from parso.parser import BaseParser
from parso.python.parser import Parser as PythonParser
from parso.python.errors import ErrorFinderConfig
from parso.python import pep8
from parso.file_io import FileIO, KnownContentFileIO
from parso.normalizer import RefactoringNormalizer, NormalizerConfig
_loaded_grammars: Dict[str, Grammar] = {}
_NodeT = TypeVar('_NodeT')

class Grammar(Generic[_NodeT]):
    """
    :py:func:`parso.load_grammar` returns instances of this class.

    Creating custom none-python grammars by calling this is not supported, yet.

    :param text: A BNF representation of your grammar.
    """
    _error_normalizer_config: Optional[NormalizerConfig] = None
    _token_namespace: Optional[Any] = None
    _default_normalizer_config: Optional[NormalizerConfig] = pep8.PEP8NormalizerConfig()
    _start_nonterminal: str

    def __init__(self, text: str, *, tokenizer: Callable, parser: Type = BaseParser, diff_parser: Optional[Type] = None) -> None:
        self._pgen_grammar = generate_grammar(text, token_namespace=self._get_token_namespace())
        self._parser = parser
        self._tokenizer = tokenizer
        self._diff_parser = diff_parser
        self._hashed = hashlib.sha256(text.encode('utf-8')).hexdigest()

    def parse(self, code: Optional[Union[str, bytes]] = None, *, error_recovery: bool = True, path: Optional[Union[str, Path]] = None, 
              start_symbol: Optional[str] = None, cache: bool = False, diff_cache: bool = False, 
              cache_path: Optional[Union[str, Path]] = None, file_io: Optional[Union[FileIO, KnownContentFileIO]] = None) -> _NodeT:
        """
        If you want to parse a Python file you want to start here, most likely.

        If you need finer grained control over the parsed instance, there will be
        other ways to access it.

        :param str code: A unicode or bytes string. When it's not possible to
            decode bytes to a string, returns a
            :py:class:`UnicodeDecodeError`.
        :param bool error_recovery: If enabled, any code will be returned. If
            it is invalid, it will be returned as an error node. If disabled,
            you will get a ParseError when encountering syntax errors in your
            code.
        :param str start_symbol: The grammar rule (nonterminal) that you want
            to parse. Only allowed to be used when error_recovery is False.
        :param str path: The path to the file you want to open. Only needed for caching.
        :param bool cache: Keeps a copy of the parser tree in RAM and on disk
            if a path is given. Returns the cached trees if the corresponding
            files on disk have not changed. Note that this stores pickle files
            on your file system (e.g. for Linux in ``~/.cache/parso/``).
        :param bool diff_cache: Diffs the cached python module against the new
            code and tries to parse only the parts that have changed. Returns
            the same (changed) module that is found in cache. Using this option
            requires you to not do anything anymore with the cached modules
            under that path, because the contents of it might change. This
            option is still somewhat experimental. If you want stability,
            please don't use it.
        :param bool cache_path: If given saves the parso cache in this
            directory. If not given, defaults to the default cache places on
            each platform.

        :return: A subclass of :py:class:`parso.tree.NodeOrLeaf`. Typically a
            :py:class:`parso.python.tree.Module`.
        """
        if code is None and path is None and (file_io is None):
            raise TypeError('Please provide either code or a path.')
        if isinstance(path, str):
            path = Path(path)
        if isinstance(cache_path, str):
            cache_path = Path(cache_path)
        if start_symbol is None:
            start_symbol = self._start_nonterminal
        if error_recovery and start_symbol != 'file_input':
            raise NotImplementedError('This is currently not implemented.')
        if file_io is None:
            if code is None:
                file_io = FileIO(path)
            else:
                file_io = KnownContentFileIO(path, code)
        if cache and file_io.path is not None:
            module_node = load_module(self._hashed, file_io, cache_path=cache_path)
            if module_node is not None:
                return module_node
        if code is None:
            code = file_io.read()
        code = python_bytes_to_unicode(code)
        lines = split_lines(code, keepends=True)
        if diff_cache:
            if self._diff_parser is None:
                raise TypeError('You have to define a diff parser to be able to use this option.')
            try:
                module_cache_item = parser_cache[self._hashed][file_io.path]
            except KeyError:
                pass
            else:
                module_node = module_cache_item.node
                old_lines = module_cache_item.lines
                if old_lines == lines:
                    return module_node
                new_node = self._diff_parser(self._pgen_grammar, self._tokenizer, module_node).update(old_lines=old_lines, new_lines=lines)
                try_to_save_module(self._hashed, file_io, new_node, lines, pickling=cache and (not is_pypy), cache_path=cache_path)
                return new_node
        tokens = self._tokenizer(lines)
        p = self._parser(self._pgen_grammar, error_recovery=error_recovery, start_nonterminal=start_symbol)
        root_node = p.parse(tokens=tokens)
        if cache or diff_cache:
            try_to_save_module(self._hashed, file_io, root_node, lines, pickling=cache and (not is_pypy), cache_path=cache_path)
        return root_node

    def _get_token_namespace(self) -> Any:
        ns = self._token_namespace
        if ns is None:
            raise ValueError('The token namespace should be set.')
        return ns

    def iter_errors(self, node: _NodeT) -> Iterator[Any]:
        """
        Given a :py:class:`parso.tree.NodeOrLeaf` returns a generator of
        :py:class:`parso.normalizer.Issue` objects. For Python this is
        a list of syntax/indentation errors.
        """
        if self._error_normalizer_config is None:
            raise ValueError('No error normalizer specified for this grammar.')
        return self._get_normalizer_issues(node, self._error_normalizer_config)

    def refactor(self, base_node: _NodeT, node_to_str_map: Dict[Any, str]) -> _NodeT:
        return RefactoringNormalizer(node_to_str_map).walk(base_node)

    def _get_normalizer(self, normalizer_config: Optional[NormalizerConfig]) -> Any:
        if normalizer_config is None:
            normalizer_config = self._default_normalizer_config
            if normalizer_config is None:
                raise ValueError("You need to specify a normalizer, because there's no default normalizer for this tree.")
        return normalizer_config.create_normalizer(self)

    def _normalize(self, node: _NodeT, normalizer_config: Optional[NormalizerConfig] = None) -> str:
        """
        TODO this is not public, yet.
        The returned code will be normalized, e.g. PEP8 for Python.
        """
        normalizer = self._get_normalizer(normalizer_config)
        return normalizer.walk(node)

    def _get_normalizer_issues(self, node: _NodeT, normalizer_config: Optional[NormalizerConfig] = None) -> List[Any]:
        normalizer = self._get_normalizer(normalizer_config)
        normalizer.walk(node)
        return normalizer.issues

    def __repr__(self) -> str:
        nonterminals = self._pgen_grammar.nonterminal_to_dfas.keys()
        txt = ' '.join(list(nonterminals)[:3]) + ' ...'
        return '<%s:%s>' % (self.__class__.__name__, txt)

class PythonGrammar(Grammar[_NodeT]):
    _error_normalizer_config: ErrorFinderConfig = ErrorFinderConfig()
    _token_namespace: Any = PythonTokenTypes
    _start_nonterminal: str = 'file_input'

    def __init__(self, version_info: PythonVersionInfo, bnf_text: str) -> None:
        super().__init__(bnf_text, tokenizer=self._tokenize_lines, parser=PythonParser, diff_parser=DiffParser)
        self.version_info: PythonVersionInfo = version_info

    def _tokenize_lines(self, lines: List[str], **kwargs: Any) -> Iterable[Any]:
        return tokenize_lines(lines, version_info=self.version_info, **kwargs)

    def _tokenize(self, code: str) -> Iterable[Any]:
        return tokenize(code, version_info=self.version_info)

def load_grammar(*, version: Optional[str] = None, path: Optional[str] = None) -> Grammar:
    """
    Loads a :py:class:`parso.Grammar`. The default version is the current Python
    version.

    :param str version: A python version string, e.g. ``version='3.8'``.
    :param str path: A path to a grammar file
    """
    version_info = parse_version_string(version)
    file = path or os.path.join('python', 'grammar%s%s.txt' % (version_info.major, version_info.minor))
    global _loaded_grammars
    path = os.path.join(os.path.dirname(__file__), file)
    try:
        return _loaded_grammars[path]
    except KeyError:
        try:
            with open(path) as f:
                bnf_text = f.read()
            grammar = PythonGrammar(version_info, bnf_text)
            return _loaded_grammars.setdefault(path, grammar)
        except FileNotFoundError:
            message = 'Python version %s.%s is currently not supported.' % (version_info.major, version_info.minor)
            raise NotImplementedError(message)
