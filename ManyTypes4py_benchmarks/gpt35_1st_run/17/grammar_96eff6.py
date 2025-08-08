import hashlib
import os
from typing import Generic, TypeVar, Union, Dict, Optional, Any
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
    _error_normalizer_config: Optional[ErrorFinderConfig] = None
    _token_namespace: Optional[PythonTokenTypes] = None
    _default_normalizer_config: pep8.PEP8NormalizerConfig = pep8.PEP8NormalizerConfig()

    def __init__(self, text: str, *, tokenizer, parser=BaseParser, diff_parser=None) -> None:
        self._pgen_grammar = generate_grammar(text, token_namespace=self._get_token_namespace())
        self._parser = parser
        self._tokenizer = tokenizer
        self._diff_parser = diff_parser
        self._hashed = hashlib.sha256(text.encode('utf-8')).hexdigest()

    def parse(self, code: Optional[str] = None, *, error_recovery: bool = True, path: Optional[str] = None, start_symbol: Optional[str] = None, cache: bool = False, diff_cache: bool = False, cache_path: Optional[str] = None, file_io: Optional[Union[FileIO, KnownContentFileIO]] = None) -> Any:
        ...

    def _get_token_namespace(self) -> PythonTokenTypes:
        ...

    def iter_errors(self, node: Any) -> Any:
        ...

    def refactor(self, base_node: Any, node_to_str_map: Dict[Any, str]) -> Any:
        ...

    def _get_normalizer(self, normalizer_config: Optional[NormalizerConfig]) -> Any:
        ...

    def _normalize(self, node: Any, normalizer_config: Optional[NormalizerConfig] = None) -> Any:
        ...

    def _get_normalizer_issues(self, node: Any, normalizer_config: Optional[NormalizerConfig] = None) -> Any:
        ...

    def __repr__(self) -> str:
        ...

class PythonGrammar(Grammar):
    _error_normalizer_config: ErrorFinderConfig = ErrorFinderConfig()
    _token_namespace: PythonTokenTypes = PythonTokenTypes
    _start_nonterminal: str = 'file_input'

    def __init__(self, version_info: PythonVersionInfo, bnf_text: str) -> None:
        super().__init__(bnf_text, tokenizer=self._tokenize_lines, parser=PythonParser, diff_parser=DiffParser)
        self.version_info = version_info

    def _tokenize_lines(self, lines: List[str], **kwargs) -> Any:
        ...

    def _tokenize(self, code: str) -> Any:
        ...

def load_grammar(*, version: Optional[str] = None, path: Optional[str] = None) -> Grammar:
    ...
