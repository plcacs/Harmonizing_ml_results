from typing import Generator

def has_metaclass(parent: Node) -> bool:
    ...

def fixup_parse_tree(cls_node: Node) -> None:
    ...

def fixup_simple_stmt(parent: Node, i: int, stmt_node: Node) -> None:
    ...

def remove_trailing_newline(node: Node) -> None:
    ...

def find_metas(cls_node: Node) -> Generator:
    ...

def fixup_indent(suite: Node) -> None:
    ...

class FixMetaclass(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = '\n    classdef<any*>\n    '

    def transform(self, node: Node, results: dict) -> None:
        ...
