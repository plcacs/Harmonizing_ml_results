
from lib2to3 import pytree, fixer_base
from lib2to3.fixer_util import Name, Attr, Call, Comma, Newline, syms
from typing import Any, Optional, Dict

class FixExitfunc(fixer_base.BaseFix):
    keep_line_order: bool = True
    BM_compatible: bool = True
    PATTERN: str = "\n              (\n                  sys_import=import_name<'import'\n                      ('sys'\n                      |\n                      dotted_as_names< (any ',')* 'sys' (',' any)* >\n                      )\n                  >\n              |\n                  expr_stmt<\n                      power< 'sys' trailer< '.' 'exitfunc' > >\n                  '=' func=any >\n              )\n              "

    def __init__(self, *args: Any) -> None:
        super(FixExitfunc, self).__init__(*args)

    def start_tree(self, tree: pytree.Node, filename: str) -> None:
        super(FixExitfunc, self).start_tree(tree, filename)
        self.sys_import: Optional[pytree.Node] = None

    def transform(self, node: pytree.Node, results: Dict[str, Any]) -> Optional[pytree.Node]:
        if 'sys_import' in results:
            if self.sys_import is None:
                self.sys_import = results['sys_import']
            return
        func = results['func'].clone()
        func.prefix = ''
        register = pytree.Node(syms.power, Attr(Name('atexit'), Name('register')))
        call = Call(register, [func], node.prefix)
        node.replace(call)
        if self.sys_import is None:
            self.warning(node, "Can't find sys import; Please add an atexit import at the top of your file.")
            return
        names: pytree.Node = self.sys_import.children[1]
        if names.type == syms.dotted_as_names:
            names.append_child(Comma())
            names.append_child(Name('atexit', ' '))
        else:
            containing_stmt: pytree.Node = self.sys_import.parent
            position: int = containing_stmt.children.index(self.sys_import)
            stmt_container: pytree.Node = containing_stmt.parent
            new_import: pytree.Node = pytree.Node(syms.import_name, [Name('import'), Name('atexit', ' ')])
            new: pytree.Node = pytree.Node(syms.simple_stmt, [new_import])
            containing_stmt.insert_child(position + 1, Newline())
            containing_stmt.insert_child(position + 2, new)
