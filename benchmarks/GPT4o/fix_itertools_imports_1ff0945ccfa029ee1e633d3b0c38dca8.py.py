' Fixer for imports of itertools.(imap|ifilter|izip|ifilterfalse) '
from lib2to3 import fixer_base
from lib2to3.fixer_util import BlankLine, syms, token
from lib2to3.pytree import Node
from typing import Optional, Union

class FixItertoolsImports(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = ("\n              import_from< 'from' 'itertools' 'import' imports=any >\n              " % locals())

    def transform(self, node: Node, results: dict) -> Optional[Union[Node, None]]:
        imports = results['imports']
        if ((imports.type == syms.import_as_name) or (not imports.children)):
            children = [imports]
        else:
            children = imports.children
        for child in children[::2]:
            if (child.type == token.NAME):
                member = child.value
                name_node = child
            elif (child.type == token.STAR):
                return None
            else:
                assert (child.type == syms.import_as_name)
                name_node = child.children[0]
            member_name = name_node.value
            if (member_name in ('imap', 'izip', 'ifilter')):
                child.value = None
                child.remove()
            elif (member_name in ('ifilterfalse', 'izip_longest')):
                node.changed()
                name_node.value = ('filterfalse' if (member_name[1] == 'f') else 'zip_longest')
        children = (imports.children[:] or [imports])
        remove_comma = True
        for child in children:
            if (remove_comma and (child.type == token.COMMA)):
                child.remove()
            else:
                remove_comma ^= True
        while (children and (children[(- 1)].type == token.COMMA)):
            children.pop().remove()
        if ((not (imports.children or getattr(imports, 'value', None))) or (imports.parent is None)):
            p = node.prefix
            node = BlankLine()
            node.prefix = p
            return node
        return None
