import re
from pgen2 import grammar, token

class Converter(grammar.Grammar):
    symbol2number: dict[str, int]
    number2symbol: dict[int, str]
    states: list[list[list[tuple[int, int]]]]
    dfas: dict[int, tuple[list[list[tuple[int, int]]], dict[int, int]]]
    labels: list[tuple[int, str]]
    start: int
    keywords: dict[str, int]
    tokens: dict[int, int]

    def run(self, graminit_h: str, graminit_c: str) -> None:
        ...

    def parse_graminit_h(self, filename: str) -> bool:
        ...

    def parse_graminit_c(self, filename: str) -> bool:
        ...

    def finish_off(self) -> None:
        ...
