from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
from .. import pytree
from ..pgen2 import token
from .. import fixer_base
from ..fixer_util import Assign, Name, Newline, Number, Subscript, syms


def is_docstring(stmt: pytree.Node) -> bool:
    return isinstance(stmt, pytree.Node) and (stmt.children[0].type == token.STRING)


class FixTupleParams(fixer_base.BaseFix):
    run_order: int = 4
    BM_compatible: bool = True
    PATTERN: str = """
        funcdef< 'def' any parameters< '(' args=any ')' >
               ['->' any] ':' suite=any+ >
        |
        lambda= 
        lambdef< 'lambda' args=vfpdef< '(' inner=any ')' >
                 ':' body=any >
    """

    def transform(self, node: pytree.Node, results: Dict[str, Any]) -> None:
        if 'lambda' in results:
            return self.transform_lambda(node, results)
        new_lines: List[pytree.Node] = []
        suite: List[pytree.Node] = results['suite']
        args: pytree.Node = results['args']
        if suite[0].children[1].type == token.INDENT:
            start: int = 2
            indent: str = suite[0].children[1].value
            end: pytree.Leaf = Newline()
        else:
            start = 0
            indent = '; '
            end = pytree.Leaf(token.INDENT, '')

        def handle_tuple(tuple_arg: pytree.Node, add_prefix: bool = False) -> None:
            n = Name(self.new_name())
            arg = tuple_arg.clone()
            arg.prefix = ''
            stmt: pytree.Node = Assign(arg, n.clone())
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
            return
        for line in new_lines:
            line.parent = suite[0]
        after: int = start
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

    def transform_lambda(self, node: pytree.Node, results: Dict[str, Any]) -> None:
        args: pytree.Node = results['args']
        body: pytree.Node = results['body']
        inner: pytree.Node = simplify_args(results['inner'])
        if inner.type == token.NAME:
            inner = inner.clone()
            inner.prefix = ' '
            args.replace(inner)
            return
        params: Union[str, List[Any]] = find_params(args)
        to_index: Dict[str, List[pytree.Node]] = map_to_index(params)
        tup_name: str = self.new_name(tuple_name(params))
        new_param: Name = Name(tup_name, prefix=' ')
        args.replace(new_param.clone())
        for n in body.post_order():
            if n.type == token.NAME and n.value in to_index:
                subscripts: List[pytree.Node] = [c.clone() for c in to_index[n.value]]
                new = pytree.Node(syms.power, [new_param.clone()] + subscripts)
                new.prefix = n.prefix
                n.replace(new)


def simplify_args(node: pytree.Node) -> pytree.Node:
    if node.type in (syms.vfplist, token.NAME):
        return node
    elif node.type == syms.vfpdef:
        while node.type == syms.vfpdef:
            node = node.children[1]
        return node
    raise RuntimeError(f'Received unexpected node {node}')


def find_params(node: pytree.Node) -> Union[str, List[Any]]:
    if node.type == syms.vfpdef:
        return find_params(node.children[1])
    elif node.type == token.NAME:
        return node.value
    return [find_params(c) for c in node.children if c.type != token.COMMA]


def map_to_index(
    param_list: Union[str, List[Any]],
    prefix: Optional[List[pytree.Node]] = None,
    d: Optional[Dict[str, List[pytree.Node]]] = None
) -> Dict[str, List[pytree.Node]]:
    if d is None:
        d = {}
    if prefix is None:
        prefix = []
    for i, obj in enumerate(param_list if isinstance(param_list, list) else [param_list]):
        trailer: List[pytree.Node] = [Subscript(Number(str(i)))]
        if isinstance(obj, list):
            map_to_index(obj, trailer, d=d)
        else:
            d[obj] = prefix + trailer
    return d


def tuple_name(param_list: Union[str, List[Any]]) -> str:
    l: List[str] = []
    if isinstance(param_list, list):
        for obj in param_list:
            if isinstance(obj, list):
                l.append(tuple_name(obj))
            else:
                l.append(obj)
    else:
        l.append(param_list)
    return '_'.join(l)
