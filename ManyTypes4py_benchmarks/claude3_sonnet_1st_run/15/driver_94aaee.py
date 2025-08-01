"""Parser driver.

This provides a high-level interface to parse a file into a syntax tree.

"""
from __future__ import annotations

__author__ = 'Guido van Rossum <guido@python.org>'
__all__ = ['Driver', 'load_grammar']
import codecs
import io
import os
import logging
import sys
from typing import Any, Callable, Iterator, List, Optional, Sequence, TextIO, Tuple, Union
from . import grammar, parse, token, tokenize, pgen

class Driver:

    def __init__(self, grammar: grammar.Grammar, convert: Optional[Callable] = None, logger: Optional[logging.Logger] = None) -> None:
        self.grammar = grammar
        if logger is None:
            logger = logging.getLogger()
        self.logger = logger
        self.convert = convert

    def parse_tokens(self, tokens: Iterator[Tuple[int, str, Tuple[int, int], Tuple[int, int], str]], debug: bool = False) -> Any:
        """Parse a series of tokens and return the syntax tree."""
        p = parse.Parser(self.grammar, self.convert)
        p.setup()
        lineno = 1
        column = 0
        type = value = start = end = line_text = None
        prefix = ''
        for quintuple in tokens:
            type, value, start, end, line_text = quintuple
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
            if type in (tokenize.COMMENT, tokenize.NL):
                prefix += value
                lineno, column = end
                if value.endswith('\n'):
                    lineno += 1
                    column = 0
                continue
            if type == token.OP:
                type = grammar.opmap[value]
            if debug:
                self.logger.debug('%s %r (prefix=%r)', token.tok_name[type], value, prefix)
            if p.addtoken(type, value, (prefix, start)):
                if debug:
                    self.logger.debug('Stop.')
                break
            prefix = ''
            lineno, column = end
            if value.endswith('\n'):
                lineno += 1
                column = 0
        else:
            raise parse.ParseError('incomplete input', type, value, (prefix, start))
        return p.rootnode

    def parse_stream_raw(self, stream: TextIO, debug: bool = False) -> Any:
        """Parse a stream and return the syntax tree."""
        tokens = tokenize.generate_tokens(stream.readline)
        return self.parse_tokens(tokens, debug)

    def parse_stream(self, stream: TextIO, debug: bool = False) -> Any:
        """Parse a stream and return the syntax tree."""
        return self.parse_stream_raw(stream, debug)

    def parse_file(self, filename: str, encoding: Optional[str] = None, debug: bool = False) -> Any:
        """Parse a file and return the syntax tree."""
        stream = codecs.open(filename, 'r', encoding)
        try:
            return self.parse_stream(stream, debug)
        finally:
            stream.close()

    def parse_string(self, text: str, debug: bool = False) -> Any:
        """Parse a string and return the syntax tree."""
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        return self.parse_tokens(tokens, debug)

def load_grammar(gt: str = 'Grammar.txt', gp: Optional[str] = None, save: bool = True, force: bool = False, logger: Optional[logging.Logger] = None) -> grammar.Grammar:
    """Load the grammar (maybe from a pickle)."""
    if logger is None:
        logger = logging.getLogger()
    if gp is None:
        head, tail = os.path.splitext(gt)
        if tail == '.txt':
            tail = ''
        gp = head + tail + '.'.join(map(str, sys.version_info)) + '.pickle'
    if force or not _newer(gp, gt):
        logger.info('Generating grammar tables from %s', gt)
        g = pgen.generate_grammar(gt)
        if save:
            logger.info('Writing grammar tables to %s', gp)
            try:
                g.dump(gp)
            except IOError as e:
                logger.info('Writing failed:' + str(e))
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

def main(*args: str) -> bool:
    """Main program, when run as a script: produce grammar pickle files.

    Calls load_grammar for each argument, a path to a grammar text file.
    """
    if not args:
        args = sys.argv[1:]
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(message)s')
    for gt in args:
        load_grammar(gt, save=True, force=True)
    return True
if __name__ == '__main__':
    sys.exit(int(not main()))
