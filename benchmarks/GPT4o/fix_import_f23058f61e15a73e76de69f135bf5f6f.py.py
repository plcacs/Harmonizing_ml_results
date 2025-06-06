'Fixer for import statements.\nIf spam is being imported from the local directory, this import:\n    from spam import eggs\nBecomes:\n    from .spam import eggs\n\nAnd this import:\n    import spam\nBecomes:\n    from . import spam\n'
from .. import fixer_base
from os.path import dirname, join, exists, sep
from ..fixer_util import FromImport, syms, token
from typing import Generator, List, Union

def traverse_imports(names) -> Generator[str, None, None]:
    '\n    Walks over all the names imported in a dotted_as_names node.\n    '
    pending: List = [names]
    while pending:
        node = pending.pop()
        if (node.type == token.NAME):
            (yield node.value)
        elif (node.type == syms.dotted_name):
            (yield ''.join([ch.value for ch in node.children]))
        elif (node.type == syms.dotted_as_name):
            pending.append(node.children[0])
        elif (node.type == syms.dotted_as_names):
            pending.extend(node.children[::(- 2)])
        else:
            raise AssertionError('unknown node type')

class FixImport(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "\n    import_from< 'from' imp=any 'import' ['('] any [')'] >\n    |\n    import_name< 'import' imp=any >\n    "

    def start_tree(self, tree, name: str) -> None:
        super(FixImport, self).start_tree(tree, name)
        self.skip: bool = ('absolute_import' in tree.future_features)

    def transform(self, node, results) -> Union[None, FromImport]:
        if self.skip:
            return
        imp = results['imp']
        if (node.type == syms.import_from):
            while (not hasattr(imp, 'value')):
                imp = imp.children[0]
            if self.probably_a_local_import(imp.value):
                imp.value = ('.' + imp.value)
                imp.changed()
        else:
            have_local: bool = False
            have_absolute: bool = False
            for mod_name in traverse_imports(imp):
                if self.probably_a_local_import(mod_name):
                    have_local = True
                else:
                    have_absolute = True
            if have_absolute:
                if have_local:
                    self.warning(node, 'absolute and local imports together')
                return
            new = FromImport('.', [imp])
            new.prefix = node.prefix
            return new

    def probably_a_local_import(self, imp_name: str) -> bool:
        if imp_name.startswith('.'):
            return False
        imp_name = imp_name.split('.', 1)[0]
        base_path = dirname(self.filename)
        base_path = join(base_path, imp_name)
        if (not exists(join(dirname(base_path), '__init__.py'))):
            return False
        for ext in ['.py', sep, '.pyc', '.so', '.sl', '.pyd']:
            if exists((base_path + ext)):
                return True
        return False
