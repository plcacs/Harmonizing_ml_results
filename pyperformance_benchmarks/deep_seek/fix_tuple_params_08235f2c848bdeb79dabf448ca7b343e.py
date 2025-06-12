from .. import pytree
from ..pgen2 import token
from .. import fixer_base
from ..fixer_util import Assign, Name, Newline, Number, Subscript, syms
from typing import Any, Dict, List, Optional, Union, cast

def is_docstring(stmt: pytree.Base) -> bool:
    return (isinstance(stmt, pytree.Node) and (stmt.children[0].type == token.STRING))

class FixTupleParams(fixer_base.BaseFix):
    run_order: int = 4
    BM_compatible: bool = True
    PATTERN: str = """
              funcdef< 'def' any parameters< '(' args=any ')' >
                       ['->' any] ':' suite=any+ >
              |
              lambda=
              lambdef< 'lambda' args=vfpdef< '(' inner=any ')' >
                       ':' body=any
              >
              """

    def transform(self, node: pytree.Base, results: Dict[str, Any]) -> Optional[pytree.Base]:
        if 'lambda' in results:
            return self.transform_lambda(node, results)
        new_lines: List[pytree.Base] = []
        suite: List[pytree.Base] = results['suite']
        args: pytree.Base = results['args']
        if suite[0].children[1].type == token.INDENT:
            start: int = 2
            indent: str = suite[0].children[1].value
            end: pytree.Base = Newline()
        else:
            start = 0
            indent = '; '
            end = pytree.Leaf(token.INDENT, '')

        def handle_tuple(tuple_arg: pytree.Base, add_prefix: bool = False) -> None:
            n: pytree.Base = Name(self.new_name())
            arg: pytree.Base = tuple_arg.clone()
            arg.prefix = ''
            stmt: pytree.Base = Assign(arg, n.clone())
            if add_prefix:
                n.prefix = ' '
            tuple_arg.replace(n)
            new_lines.append(pytree.Node(syms.simple_stmt, [stmt, end.clone()]))

        if args.type == syms.tfpdef:
            handle_tuple(args)
        elif args.type == syms.typedargslist:
            for i, arg in enumerate(args.children):
                if arg.type == syms.tfpdef:
                    handle_tuple(arg, add_prefix=(i > 0))
        if not new_lines:
            return None
        for line in new_lines:
            line.parent = suite[0]
        after = start
        if start == 0:
            new_lines[0].prefix = ' '
        elif is_docstring(suite[0].children[start]):
            new_lines[0].prefix = indent
            after = start + 1
        for line in new_lines:
            line.parent = suite[0]
        suite[0].children[after:after] = new_lines
        for i in range(after + 1, after + len(new_lines) + 1):
            suite[0].children[i].prefix = indent
        suite[0].changed()
        return None

    def transform_lambda(self, node: pytree.Base, results: Dict[str, Any]) -> pytree.Base:
        args: pytree.Base = results['args']
        body: pytree.Base = results['body']
        inner: pytree.Base = simplify_args(results['inner'])
        if inner.type == token.NAME:
            inner = inner.clone()
            inner.prefix = ' '
            args.replace(inner)
            return node
        params: Union[str, List[Any]] = find_params(args)
        to_index: Dict[str, List[pytree.Base]] = map_to_index(params)
        tup_name: str = self.new_name(tuple_name(params))
        new_param: pytree.Base = Name(tup_name, prefix=' ')
        args.replace(new_param.clone())
        for n in body.post_order():
            if n.type == token.NAME and n.value in to_index:
                subscripts: List[pytree.Base] = [c.clone() for c in to_index[n.value]]
                new: pytree.Base = pytree.Node(syms.power, [new_param.clone()] + subscripts)
                new.prefix = n.prefix
                n.replace(new)
        return node

def simplify_args(node: pytree.Base) -> pytree.Base:
    if node.type in (syms.vfplist, token.NAME):
        return node
    elif node.type == syms.vfpdef:
        while node.type == syms.vfpdef:
            node = node.children[1]
        return node
    raise RuntimeError(f'Received unexpected node {node}')

def find_params(node: pytree.Base) -> Union[str, List[Any]]:
    if node.type == syms.vfpdef:
        return find_params(node.children[1])
    elif node.type == token.NAME:
        return node.value
    return [find_params(c) for c in node.children if c.type != token.COMMA]

def map_to_index(param_list: Union[str, List[Any]], prefix: Optional[List[pytree.Base]] = None, d: Optional[Dict[str, List[pytree.Base]]] = None) -> Dict[str, List[pytree.Base]]:
    if d is None:
        d = {}
    if prefix is None:
        prefix = []
    if isinstance(param_list, str):
        d[param_list] = prefix.copy()
        return d
    for i, obj in enumerate(param_list):
        trailer: List[pytree.Base] = [Subscript(Number(str(i)))]
        if isinstance(obj, list):
            map_to_index(obj, prefix + trailer, d=d)
        else:
            d[obj] = prefix + trailer
    return d

def tuple_name(param_list: Union[str, List[Any]]) -> str:
    l: List[str] = []
    if isinstance(param_list, str):
        return param_list
    for obj in param_list:
        if isinstance(obj, list):
            l.append(tuple_name(obj))
        else:
            l.append(obj)
    return '_'.join(l)
