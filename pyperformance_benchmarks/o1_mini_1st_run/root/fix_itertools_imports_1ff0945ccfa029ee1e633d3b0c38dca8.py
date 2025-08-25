from typing import Any, Dict, List, Optional

from lib2to3 import fixer_base
from lib2to3.fixer_util import BlankLine, syms, token
from lib2to3.pytree import Node, Leaf


class FixItertoolsImports(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = (
        "import_from< 'from' 'itertools' 'import' imports=any >"
    )

    def transform(self, node: Node, results: Dict[str, Any]) -> Optional[Node]:
        imports = results['imports']
        if imports.type == syms.import_as_name or not imports.children:
            children: List[Node] = [imports]
        else:
            children = imports.children

        for child in children[::2]:
            if child.type == token.NAME:
                member: str = child.value
                name_node: Leaf = child
            elif child.type == token.STAR:
                return
            else:
                assert child.type == syms.import_as_name
                name_node = child.children[0]  # type: Leaf

            member_name: str = name_node.value
            if member_name in ('imap', 'izip', 'ifilter'):
                child.value = None
                child.remove()
            elif member_name in ('ifilterfalse', 'izip_longest'):
                node.changed()
                name_node.value = 'filterfalse' if member_name.startswith('if') else 'zip_longest'

        children = imports.children[:] or [imports]
        remove_comma: bool = True
        for child in children:
            if remove_comma and child.type == token.COMMA:
                child.remove()
            else:
                remove_comma = not remove_comma

        while children and children[-1].type == token.COMMA:
            children.pop().remove()

        if not (imports.children or getattr(imports, 'value', None)) or imports.parent is None:
            p: str = node.prefix
            new_node = BlankLine()
            new_node.prefix = p
            return new_node

        return None
