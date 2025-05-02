import hashlib
import os
from typing import Generic, TypeVar, Union, Dict, Optional, Any, Generator
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
from parso.tree import NodeOrLeaf
from parso.normalizer import Issue

_loaded_grammars: Dict[str, 'Grammar'] = {}
_NodeT = TypeVar('_NodeT')

class Grammar(Generic[_NodeT]):
    _error_normalizer_config: Optional[NormalizerConfig] = None
    _token_namespace: Optional[Any] = None
    _default_normalizer_config: NormalizerConfig = pep8.PEP8NormalizerConfig()

    def __init__(self, text: str, *, tokenizer: Any, parser: BaseParser = BaseParser, diff_parser: Optional[DiffParser] = None) -> None:
        self._pgen_grammar = generate_grammar(text, token_namespace=self._get_token_namespace())
        self._parser = parser
        self._tokenizer = tokenizer
        self._diff_parser = diff_parser
        self._hashed = hashlib.sha256(text.encode('utf-8')).hexdigest()

    def parse(self, code: Optional[Union[str, bytes]] = None, *, error_recovery: bool = True, path: Optional[Union[str, Path]] = None, start_symbol: Optional[str] = None, cache: bool = False, diff_cache: bool = False, cache_path: Optional[Union[str, Path]] = None, file_io: Optional[FileIO] = None) -> NodeOrLeaf:
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

    def iter_errors(self, node: NodeOrLeaf) -> Generator[Issue, None, None]:
        if self._error_normalizer_config is None:
            raise ValueError('No error normalizer specified for this grammar.')
        return self._get_normalizer_issues(node, self._error_normalizer_config)

    def refactor(self, base_node: NodeOrLeaf, node_to_str_map: Dict[NodeOrLeaf, str]) -> NodeOrLeaf:
        return RefactoringNormalizer(node_to_str_map).walk(base_node)

    def _get_normalizer(self, normalizer_config: Optional[NormalizerConfig]) -> NormalizerConfig:
        if normalizer_config is None:
            normalizer_config = self._default_normalizer_config
            if normalizer_config is None:
                raise ValueError("You need to specify a normalizer, because there's no default normalizer for this tree.")
        return normalizer_config.create_normalizer(self)

    def _normalize(self, node: NodeOrLeaf, normalizer_config: Optional[NormalizerConfig] = None) -> NodeOrLeaf:
        normalizer = self._get_normalizer(normalizer_config)
        return normalizer.walk(node)

    def _get_normalizer_issues(self, node: NodeOrLeaf, normalizer_config: Optional[NormalizerConfig] = None) -> Generator[Issue, None, None]:
        normalizer = self._get_normalizer(normalizer_config)
        normalizer.walk(node)
        return normalizer.issues

    def __repr__(self) -> str:
        nonterminals = self._pgen_grammar.nonterminal_to_dfas.keys()
        txt = ' '.join(list(nonterminals)[:3]) + ' ...'
        return '<%s:%s>' % (self.__class__.__name__, txt)

class PythonGrammar(Grammar):
    _error_normalizer_config: ErrorFinderConfig = ErrorFinderConfig()
    _token_namespace: Any = PythonTokenTypes
    _start_nonterminal: str = 'file_input'

    def __init__(self, version_info: PythonVersionInfo, bnf_text: str) -> None:
        super().__init__(bnf_text, tokenizer=self._tokenize_lines, parser=PythonParser, diff_parser=DiffParser)
        self.version_info = version_info

    def _tokenize_lines(self, lines: Any, **kwargs: Any) -> Any:
        return tokenize_lines(lines, version_info=self.version_info, **kwargs)

    def _tokenize(self, code: Any) -> Any:
        return tokenize(code, version_info=self.version_info)

def load_grammar(*, version: Optional[str] = None, path: Optional[str] = None) -> Grammar:
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
