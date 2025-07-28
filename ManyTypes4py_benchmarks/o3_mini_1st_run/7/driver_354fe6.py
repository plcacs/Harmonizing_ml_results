#!/usr/bin/env python
"""Parser driver.

This provides a high-level interface to parse a file into a syntax tree.

"""
__author__ = 'Guido van Rossum <guido@python.org>'
__all__ = ['Driver', 'load_grammar']

import io
import logging
import os
import pkgutil
import sys
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from logging import Logger
from typing import IO, Any, Optional, Union, Iterator as TypingIterator, List, Tuple
from blib2to3.pgen2.grammar import Grammar
from blib2to3.pgen2.tokenize import GoodTokenInfo
from blib2to3.pytree import NL
from . import grammar, parse, pgen, token, tokenize

Path = Union[str, os.PathLike[str]]

@dataclass
class ReleaseRange:
    start: int
    end: Optional[int] = None
    tokens: List[GoodTokenInfo] = field(default_factory=list)

    def lock(self) -> None:
        total_eaten: int = len(self.tokens)
        self.end = self.start + total_eaten

class TokenProxy:
    def __init__(self, generator: TypingIterator[GoodTokenInfo]) -> None:
        self._tokens: TypingIterator[GoodTokenInfo] = generator
        self._counter: int = 0
        self._release_ranges: List[ReleaseRange] = []

    @contextmanager
    def release(self) -> Iterator["TokenProxy"]:
        release_range: ReleaseRange = ReleaseRange(self._counter)
        self._release_ranges.append(release_range)
        try:
            yield self
        finally:
            release_range.lock()

    def eat(self, point: int) -> GoodTokenInfo:
        eaten_tokens: List[GoodTokenInfo] = self._release_ranges[-1].tokens
        if point < len(eaten_tokens):
            return eaten_tokens[point]
        else:
            while point >= len(eaten_tokens):
                token_item: GoodTokenInfo = next(self._tokens)
                eaten_tokens.append(token_item)
            return token_item

    def __iter__(self) -> TypingIterator[GoodTokenInfo]:
        return self

    def __next__(self) -> GoodTokenInfo:
        token_item: GoodTokenInfo
        for release_range in self._release_ranges:
            assert release_range.end is not None
            start, end = (release_range.start, release_range.end)
            if start <= self._counter < end:
                token_item = release_range.tokens[self._counter - start]
                break
        else:
            token_item = next(self._tokens)
        self._counter += 1
        return token_item

    def can_advance(self, to: int) -> bool:
        try:
            self.eat(to)
        except StopIteration:
            return False
        else:
            return True

class Driver:
    def __init__(self, grammar: Grammar, logger: Optional[Logger] = None) -> None:
        self.grammar: Grammar = grammar
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger: Logger = logger

    def parse_tokens(self, tokens: TypingIterator[GoodTokenInfo], debug: bool = False) -> Any:
        """Parse a series of tokens and return the syntax tree."""
        proxy: TokenProxy = TokenProxy(tokens)
        p: parse.Parser = parse.Parser(self.grammar)
        p.setup(proxy=proxy)
        lineno: int = 1
        column: int = 0
        indent_columns: List[int] = []
        type_: Optional[int] = None
        value: Optional[str] = None
        start: Optional[Tuple[int, int]] = None
        end: Optional[Tuple[int, int]] = None
        line_text: Optional[str] = None
        prefix: str = ''
        for quintuple in proxy:
            type_, value, start, end, line_text = quintuple
            if start != (lineno, column):
                assert (lineno, column) <= start, ((lineno, column), start)
                s_lineno, s_column = start
                if lineno < s_lineno:
                    prefix += '\n' * (s_lineno - lineno)
                    lineno = s_lineno
                    column = 0
                if column < s_column:
                    prefix += line_text[column:s_column]
                    column = s_column
            if type_ in (tokenize.COMMENT, tokenize.NL):
                prefix += value
                lineno, column = end
                if value.endswith('\n'):
                    lineno += 1
                    column = 0
                continue
            if type_ == token.OP:
                type_ = grammar.opmap[value]
            if debug:
                assert type_ is not None
                self.logger.debug('%s %r (prefix=%r)', token.tok_name[type_], value, prefix)
            if type_ == token.INDENT:
                indent_columns.append(len(value))
                _prefix: str = prefix + value
                prefix = ''
                value = ''
            elif type_ == token.DEDENT:
                _indent_col: int = indent_columns.pop()
                prefix, _prefix = self._partially_consume_prefix(prefix, _indent_col)
            if p.addtoken(int(type_), value, (prefix, start)):
                if debug:
                    self.logger.debug('Stop.')
                break
            prefix = ''
            if type_ in {token.INDENT, token.DEDENT}:
                prefix = _prefix
            lineno, column = end  # type: ignore
            if value.endswith('\n') and type_ != token.FSTRING_MIDDLE:
                lineno += 1
                column = 0
        else:
            assert start is not None
            raise parse.ParseError('incomplete input', type_, value, (prefix, start))
        assert p.rootnode is not None
        return p.rootnode

    def parse_stream_raw(self, stream: IO[str], debug: bool = False) -> Any:
        """Parse a stream and return the syntax tree."""
        tokens: TypingIterator[GoodTokenInfo] = tokenize.generate_tokens(stream.readline, grammar=self.grammar)
        return self.parse_tokens(tokens, debug)

    def parse_stream(self, stream: IO[str], debug: bool = False) -> Any:
        """Parse a stream and return the syntax tree."""
        return self.parse_stream_raw(stream, debug)

    def parse_file(self, filename: Path, encoding: Optional[str] = None, debug: bool = False) -> Any:
        """Parse a file and return the syntax tree."""
        with open(filename, encoding=encoding) as stream:
            return self.parse_stream(stream, debug)

    def parse_string(self, text: str, debug: bool = False) -> Any:
        """Parse a string and return the syntax tree."""
        tokens: TypingIterator[GoodTokenInfo] = tokenize.generate_tokens(io.StringIO(text).readline, grammar=self.grammar)
        return self.parse_tokens(tokens, debug)

    def _partially_consume_prefix(self, prefix: str, column: int) -> Tuple[str, str]:
        lines: List[str] = []
        current_line: str = ''
        current_column: int = 0
        wait_for_nl: bool = False
        for char in prefix:
            current_line += char
            if wait_for_nl:
                if char == '\n':
                    if current_line.strip() and current_column < column:
                        res: str = ''.join(lines)
                        return (res, prefix[len(res):])
                    lines.append(current_line)
                    current_line = ''
                    current_column = 0
                    wait_for_nl = False
            elif char in ' \t':
                current_column += 1
            elif char == '\n':
                current_column = 0
            elif char == '\x0c':
                current_column = 0
            else:
                wait_for_nl = True
        return (''.join(lines), current_line)

def _generate_pickle_name(gt: str, cache_dir: Optional[Path] = None) -> str:
    head, tail = os.path.splitext(gt)
    if tail == '.txt':
        tail = ''
    name: str = head + tail + '.'.join(map(str, sys.version_info)) + '.pickle'
    if cache_dir:
        return os.path.join(cache_dir, os.path.basename(name))
    else:
        return name

def load_grammar(gt: str = 'Grammar.txt', gp: Optional[str] = None, save: bool = True, force: bool = False, logger: Optional[Logger] = None) -> Grammar:
    """Load the grammar (maybe from a pickle)."""
    if logger is None:
        logger = logging.getLogger(__name__)
    gp = _generate_pickle_name(gt) if gp is None else gp
    if force or not _newer(gp, gt):
        g: Grammar = pgen.generate_grammar(gt)
        if save:
            try:
                g.dump(gp)
            except OSError:
                pass
    else:
        g = grammar.Grammar()
        g.load(gp)
    return g

def _newer(a: str, b: str) -> bool:
    """Inquire whether file a was written since file b."""
    if not os.path.exists(a):
        return False
    if not os.path.exists(b):
        return True
    return os.path.getmtime(a) >= os.path.getmtime(b)

def load_packaged_grammar(package: str, grammar_source: str, cache_dir: Optional[Path] = None) -> Grammar:
    """Normally, loads a pickled grammar by doing
        pkgutil.get_data(package, pickled_grammar)
    where *pickled_grammar* is computed from *grammar_source* by adding the
    Python version and using a ``.pickle`` extension.

    However, if *grammar_source* is an extant file, load_grammar(grammar_source)
    is called instead. This facilitates using a packaged grammar file when needed
    but preserves load_grammar's automatic regeneration behavior when possible.
    """
    if os.path.isfile(grammar_source):
        gp: Optional[str] = _generate_pickle_name(grammar_source, cache_dir) if cache_dir else None
        return load_grammar(grammar_source, gp=gp)
    pickled_name: str = _generate_pickle_name(os.path.basename(grammar_source), cache_dir)
    data: Optional[bytes] = pkgutil.get_data(package, pickled_name)
    assert data is not None
    g: Grammar = grammar.Grammar()
    g.loads(data)
    return g

def main(*args: str) -> bool:
    """Main program, when run as a script: produce grammar pickle files.

    Calls load_grammar for each argument, a path to a grammar text file.
    """
    if not args:
        args = tuple(sys.argv[1:])
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(message)s')
    for gt in args:
        load_grammar(gt, save=True, force=True)
    return True

if __name__ == '__main__':
    sys.exit(int(not main()))