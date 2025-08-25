from __future__ import annotations
import io
import os
import logging
import pkgutil
import sys
from typing import Optional, Iterable, Tuple, Any, IO, Callable, Union
from . import grammar, parse, token, tokenize, pgen

class Driver(object):
    def __init__(self, grammar: grammar.Grammar, convert: Optional[Callable] = None, logger: Optional[logging.Logger] = None) -> None:
        self.grammar = grammar
        if logger is None:
            logger = logging.getLogger()
        self.logger = logger
        self.convert = convert

    def parse_tokens(self, tokens: Iterable[Tuple[int, str, Tuple[int, int], Tuple[int, int], str]], debug: bool = False) -> Any:
        "Parse a series of tokens and return the syntax tree."
        p = parse.Parser(self.grammar, self.convert)
        p.setup()
        lineno: int = 1
        column: int = 0
        type_: Optional[int] = None
        value: Optional[str] = None
        start: Optional[Tuple[int, int]] = None
        end: Optional[Tuple[int, int]] = None
        line_text: Optional[str] = None
        prefix: str = ''
        for quintuple in tokens:
            (type_, value, start, end, line_text) = quintuple
            if start != (lineno, column):
                assert ( (lineno, column) <= start ), ((lineno, column), start)
                (s_lineno, s_column) = start  # type: Tuple[int, int]
                if lineno < s_lineno:
                    prefix += '\n' * (s_lineno - lineno)
                    lineno = s_lineno
                    column = 0
                if column < s_column:
                    prefix += line_text[column:s_column]
                    column = s_column
            if type_ in (tokenize.COMMENT, tokenize.NL):
                prefix += value
                (lineno, column) = end
                if value.endswith('\n'):
                    lineno += 1
                    column = 0
                continue
            if type_ == token.OP:
                type_ = grammar.opmap[value]
            if debug:
                self.logger.debug('%s %r (prefix=%r)', token.tok_name[type_], value, prefix)
            if p.addtoken(type_, value, (prefix, start)):
                if debug:
                    self.logger.debug('Stop.')
                break
            prefix = ''
            (lineno, column) = end
            if value.endswith('\n'):
                lineno += 1
                column = 0
        else:
            raise parse.ParseError('incomplete input', type_, value, (prefix, start))
        return p.rootnode

    def parse_stream_raw(self, stream: IO[str], debug: bool = False) -> Any:
        "Parse a stream and return the syntax tree."
        tokens = tokenize.generate_tokens(stream.readline)
        return self.parse_tokens(tokens, debug)

    def parse_stream(self, stream: IO[str], debug: bool = False) -> Any:
        "Parse a stream and return the syntax tree."
        return self.parse_stream_raw(stream, debug)

    def parse_file(self, filename: str, encoding: Optional[str] = None, debug: bool = False) -> Any:
        "Parse a file and return the syntax tree."
        with io.open(filename, 'r', encoding=encoding) as stream:
            return self.parse_stream(stream, debug)

    def parse_string(self, text: str, debug: bool = False) -> Any:
        "Parse a string and return the syntax tree."
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        return self.parse_tokens(tokens, debug)

def _generate_pickle_name(gt: str) -> str:
    (head, tail) = os.path.splitext(gt)
    if tail == '.txt':
        tail = ''
    return (head + tail + '.'.join(map(str, sys.version_info)) + '.pickle')

def load_grammar(gt: str = 'Grammar.txt', gp: Optional[str] = None, save: bool = True, force: bool = False, logger: Optional[logging.Logger] = None) -> Any:
    "Load the grammar (maybe from a pickle)."
    if logger is None:
        logger = logging.getLogger()
    gp = _generate_pickle_name(gt) if (gp is None) else gp
    if force or (not _newer(gp, gt)):
        logger.info('Generating grammar tables from %s', gt)
        g = pgen.generate_grammar(gt)
        if save:
            logger.info('Writing grammar tables to %s', gp)
            try:
                g.dump(gp)
            except OSError as e:
                logger.info('Writing failed: %s', e)
    else:
        g = grammar.Grammar()
        g.load(gp)
    return g

def _newer(a: str, b: str) -> bool:
    "Inquire whether file a was written since file b."
    if not os.path.exists(a):
        return False
    if not os.path.exists(b):
        return True
    return os.path.getmtime(a) >= os.path.getmtime(b)

def load_packaged_grammar(package: str, grammar_source: str) -> Any:
    """Normally, loads a pickled grammar by doing
    pkgutil.get_data(package, pickled_grammar)
    where *pickled_grammar* is computed from *grammar_source* by adding the
    Python version and using a ``.pickle`` extension.

    However, if *grammar_source* is an extant file, load_grammar(grammar_source)
    is called instead. This facilitates using a packaged grammar file when needed
    but preserves load_grammar's automatic regeneration behavior when possible.
    """
    if os.path.isfile(grammar_source):
        return load_grammar(grammar_source)
    pickled_name = _generate_pickle_name(os.path.basename(grammar_source))
    data: Optional[bytes] = pkgutil.get_data(package, pickled_name)
    g = grammar.Grammar()
    if data is not None:
        g.loads(data)
    return g

def main(*args: str) -> bool:
    "Main program, when run as a script: produce grammar pickle files.\n\n    Calls load_grammar for each argument, a path to a grammar text file.\n    "
    if not args:
        args = tuple(sys.argv[1:])
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(message)s')
    for gt in args:
        load_grammar(gt, save=True, force=True)
    return True

if __name__ == '__main__':
    sys.exit(int(not main()))