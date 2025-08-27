from lib2to3 import pytree, fixer_base
from lib2to3.fixer_util import Name, Attr, Call, Comma, Newline, syms
from lib2to3.pgen2 import token
from lib2to3.pygram import python_symbols as syms
from typing import Dict, Any, Optional, cast

class FixExitfunc(fixer_base.BaseFix):
    keep_line_order: bool = True
    BM_compatible: bool = True
    PATTERN: str = "\n              (\n                  sys_import=import_name<'import'\n                      ('sys'\n                      |\n                      dotted_as_names< (any ',')* 'sys' (',' any)* >\n                      )\n                  >\n              |\n                  expr_stmt<\n                      power< 'sys' trailer< '.' 'exitfunc' > >\n                  '=' func=any >\n              )\n              "

    def __init__(self, *args: Any) -> None:
        super(FixExitfunc, self).__init__(*args)
        self.sys_import: Optional[pytree.Node] = None

    def start_tree(self, tree: pytree.Node, filename: str) -> None:
        super(FixExitfunc, self).start_tree(tree, filename)
        self.sys_import = None

    def transform(self, node: pytree.Node, results: Dict[str, Any]) -> None:
        if 'sys_import' in results:
            if self.sys_import is None:
                self.sys_import = cast(pytree.Node, results['sys_import'])
            return
        
        func = cast(pytree.Node, results['func']).clone()
        func.prefix = ''
        register = pytree.Node(syms.power, [Attr(Name('atexit'), Name('register'))])
        call = Call(register, [func], node.prefix)
        node.replace(call)
        
        if self.sys_import is None:
            self.warning(node, "Can't find sys import; Please add an atexit import at the top of your file.")
            return
        
        names = self.sys_import.children[1]
        if names.type == syms.dotted_as_names:
            names.append_child(Comma())
            names.append_child(Name('atexit', ' '))
        else:
            containing_stmt = self.sys_import.parent
            position = containing_stmt.children.index(self.sys_import)
            stmt_container = containing_stmt.parent
            new_import = pytree.Node(syms.import_name, [Name('import'), Name('atexit', ' ')])
            new = pytree.Node(syms.simple_stmt, [new_import])
            containing_stmt.insert_child(position + 1, Newline())
            containing_stmt.insert_child(position + 2, new)
