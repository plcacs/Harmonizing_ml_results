from typing import IO, Any, Optional, Union, List, Tuple, cast

class ReleaseRange:
    end: Optional[int] = None
    tokens: List[Any] = field(default_factory=list)

    def lock(self) -> None:
        total_eaten = len(self.tokens)
        self.end = self.start + total_eaten

class TokenProxy:

    def __init__(self, generator: Iterator) -> None:
        self._tokens = generator
        self._counter = 0
        self._release_ranges: List[ReleaseRange] = []

    def eat(self, point: int) -> Any:
        eaten_tokens = self._release_ranges[-1].tokens
        if point < len(eaten_tokens):
            return eaten_tokens[point]
        else:
            while point >= len(eaten_tokens):
                token = next(self._tokens)
                eaten_tokens.append(token)
            return token

    def __iter__(self) -> 'TokenProxy':
        return self

    def __next__(self) -> Any:
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

    def can_advance(self, to: int) -> bool:
        try:
            self.eat(to)
        except StopIteration:
            return False
        else:
            return True

class Driver:

    def __init__(self, grammar: Grammar, logger: Optional[Logger] = None) -> None:
        self.grammar = grammar
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    def parse_tokens(self, tokens: Iterator, debug: bool = False) -> Any:
        ...

    def parse_stream_raw(self, stream: IO, debug: bool = False) -> Any:
        ...

    def parse_stream(self, stream: IO, debug: bool = False) -> Any:
        ...

    def parse_file(self, filename: str, encoding: Optional[str] = None, debug: bool = False) -> Any:
        ...

    def parse_string(self, text: str, debug: bool = False) -> Any:
        ...

    def _partially_consume_prefix(self, prefix: str, column: int) -> Tuple[str, str]:
        ...

def load_grammar(gt: str = 'Grammar.txt', gp: Optional[str] = None, save: bool = True, force: bool = False, logger: Optional[Logger] = None) -> Grammar:
    ...

def _newer(a: str, b: str) -> bool:
    ...

def load_packaged_grammar(package: str, grammar_source: str, cache_dir: Optional[str] = None) -> Grammar:
    ...

def main(*args: str) -> bool:
    ...
