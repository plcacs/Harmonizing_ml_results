from typing import IO, Any, Optional, Union

Path = Union[str, 'os.PathLike[str]']

class ReleaseRange:
    end: Optional[int] = None
    tokens: list = field(default_factory=list)

    def lock(self):
        total_eaten = len(self.tokens)
        self.end = self.start + total_eaten

class TokenProxy:

    def __init__(self, generator):
        self._tokens = generator
        self._counter = 0
        self._release_ranges = []

    def eat(self, point):
        eaten_tokens = self._release_ranges[-1].tokens
        if point < len(eaten_tokens):
            return eaten_tokens[point]
        else:
            while point >= len(eaten_tokens):
                token = next(self._tokens)
                eaten_tokens.append(token)
            return token

    def __iter__(self):
        return self

    def __next__(self):
        for release_range in self._release_ranges:
            assert release_range.end is not None
            start, end = (release_range.start, release_range.end)
            if start <= self._counter < end:
                token = release_range.tokens[self._counter - start]
                break
        else:
            token = next(self._tokens)
        self._counter += 1
        return token

    def can_advance(self, to):
        try:
            self.eat(to)
        except StopIteration:
            return False
        else:
            return True

class Driver:

    def __init__(self, grammar: Grammar, logger: Optional[Logger] = None):
        self.grammar = grammar
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    def parse_tokens(self, tokens: Iterable, debug: bool = False):
        ...

    def parse_stream_raw(self, stream: IO, debug: bool = False):
        ...

    def parse_stream(self, stream: IO, debug: bool = False):
        ...

    def parse_file(self, filename: str, encoding: Optional[str] = None, debug: bool = False):
        ...

    def parse_string(self, text: str, debug: bool = False):
        ...

    def _partially_consume_prefix(self, prefix: str, column: int):
        ...

def load_grammar(gt: str = 'Grammar.txt', gp: Optional[str] = None, save: bool = True, force: bool = False, logger: Optional[Logger] = None):
    ...

def load_packaged_grammar(package, grammar_source, cache_dir: Optional[str] = None):
    ...

def main(*args):
    ...
