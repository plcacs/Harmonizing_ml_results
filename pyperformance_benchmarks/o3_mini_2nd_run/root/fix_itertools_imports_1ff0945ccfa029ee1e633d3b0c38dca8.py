from typing import Any, Dict, List
from lib2to3 import fixer_base
from lib2to3.fixer_util import BlankLine, syms, token
from lib2to3.pytree import Node, Leaf

class FixItertoolsImports(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = ("\n              import_from< 'from' 'itertools' 'import' imports=any >\n              " % locals())

    def transform(self, node: Node, results: Dict[str, Any]) -> Node:
        imports: Any = results['imports']
        if (imports.type == syms.import_as_name) or (not getattr(imports, 'children', None)):
            children: List[Any] = [imports]
        else:
            children = imports.children
        for child in children[::2]:
            if child.type == token.NAME:
                member: str = child.value
                name_node: Any = child
            elif child.type == token.STAR:
                return node
            else:
                assert child.type == syms.import_as_name
                name_node = child.children[0]
            member_name: str = name_node.value
            if member_name in ('imap', 'izip', 'ifilter'):
                child.value = None
                child.remove()
            elif member_name in ('ifilterfalse', 'izip_longest'):
                node.changed()
                name_node.value = 'filterfalse' if member_name[1] == 'f' else 'zip_longest'
        children = (imports.children[:] if getattr(imports, 'children', None) else [imports])
        remove_comma: bool = True
        for child in children:
            if remove_comma and child.type == token.COMMA:
                child.remove()
            else:
                remove_comma ^= True
        while children and children[-1].type == token.COMMA:
            children.pop().remove()
        if (not (getattr(imports, 'children', None) or getattr(imports, 'value', None))) or (imports.parent is None):
            p: str = node.prefix
            node = BlankLine()
            node.prefix = p
            return node
        return node