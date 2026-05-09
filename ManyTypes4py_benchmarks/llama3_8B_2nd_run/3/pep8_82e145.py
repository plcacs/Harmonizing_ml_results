import re
from contextlib import contextmanager
from typing import Tuple
from parso.python.errors import ErrorFinder, ErrorFinderConfig
from parso.normalizer import Rule
from parso.python.tree import Flow, Scope

class IndentationTypes:
    VERTICAL_BRACKET = object()
    HANGING_BRACKET = object()
    BACKSLASH = object()
    SUITE = object()
    IMPLICIT = object()

class IndentationNode:
    type: IndentationTypes
    def __init__(self, config, indentation, parent=None):
        self.bracket_indentation: str
        self.parent: IndentationNode | None
        self.indentation: str

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"


class BracketNode(IndentationNode):
    def __init__(self, config, leaf, parent, in_suite_introducer: bool):
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__>"


class ImplicitNode(BracketNode):
    type: IndentationTypes
    def __init__(self, config, leaf, parent):
        ...


class BackslashNode(IndentationNode):
    type: IndentationTypes
    def __init__(self, config, parent_indentation, containing_leaf, spacing, parent=None):
        ...


class PEP8Normalizer(ErrorFinder):
    ...

    def _visit_part(self, part: object, spacing: object, leaf: object) -> None:
        ...

    def _check_line_length(self, part: object, spacing: object) -> None:
        ...

    def _check_spacing(self, part: object, spacing: object) -> None:
        ...

    def _analyse_non_prefix(self, leaf: object) -> None:
        ...

    def add_issue(self, node: object, code: int, message: str) -> None:
        ...


class PEP8NormalizerConfig(ErrorFinderConfig):
    normalizer_class: type
    indentation: str
    hanging_indentation: str
    max_characters: int
    spaces_before_comment: int

    def __init__(self, indentation: str = ' ' * 4, hanging_indentation: str | None = None, max_characters: int = 79, spaces_before_comment: int = 2):
        ...


class BlankLineAtEnd(Rule):
    code: int
    message: str

    def is_issue(self, leaf: object) -> bool:
        return self._newline_count >= 2
