'Fixer for function definitions with tuple parameters.\n\ndef func(((a, b), c), d):\n    ...\n\n    ->\n\ndef func(x, d):\n    ((a, b), c) = x\n    ...\n\nIt will also support lambdas:\n\n    lambda (x, y): x + y -> lambda t: t[0] + t[1]\n\n    # The parens are a syntax error in Python 3\n    lambda (x): x + y -> lambda x: x + y\n'
from .. import pytree
from ..pgen2 import token
from .. import fixer_base
from ..fixer_util import Assign, Name, Newline, Number, Subscript, syms
from typing import Any, Dict, List, Optional, Union, Sequence, Tuple, Set

def is_docstring(stmt: Any) -> bool:
    return (isinstance(stmt, pytree.Node) and (stmt.children[0].type == token.STRING))

class FixTupleParams(fixer_base.BaseFix):
    run_order: int = 4
    BM_compatible: bool = True
    PATTERN: str = "\n              funcdef< 'def' any parameters< '(' args=any ')' >\n                       ['->' any] ':' suite=any+ >\n              |\n              lambda=\n              lambdef< 'lambda' args=vfpdef< '(' inner=any ')' >\n                       ':' body=any\n              >\n              "

    def transform(self, node: Any, results: Dict[str, Any]) -> Optional[Any]:
        if ('lambda' in results):
            return self.transform_lambda(node, results)
        new_lines: List[Any] = []
        suite: List[Any] = results['suite']
        args: Any = results['args']
        if (suite[0].children[1].type == token.INDENT):
            start: int = 2
            indent: str = suite[0].children[1].value
            end: Any = Newline()
        else:
            start = 0
            indent = '; '
            end = pytree.Leaf(token.INDENT, '')

        def handle_tuple(tuple_arg: Any, add_prefix: bool = False) -> None:
            n: Any = Name(self.new_name())
            arg: Any = tuple_arg.clone()
            arg.prefix = ''
            stmt: Any = Assign(arg, n.clone())
            if add_prefix:
                n.prefix = ' '
            tuple_arg.replace(n)
            new_lines.append(pytree.Node(syms.simple_stmt, [stmt, end.clone()]))
        if (args.type == syms.tfpdef):
            handle_tuple(args)
        elif (args.type == syms.typedargslist):
            for (i, arg) in enumerate(args.children):
                if (arg.type == syms.tfpdef):
                    handle_tuple(arg, add_prefix=(i > 0))
        if (not new_lines):
            return
        for line in new_lines:
            line.parent = suite[0]
        after: int = start
        if (start == 0):
            new_lines[0].prefix = ' '
        elif is_docstring(suite[0].children[start]):
            new_lines[0].prefix = indent
            after = (start + 1)
        for line in new_lines:
            line.parent = suite[0]
        suite[0].children[after:after] = new_lines
        for i in range((after + 1), ((after + len(new_lines)) + 1)):
            suite[0].children[i].prefix = indent
        suite[0].changed()

    def transform_lambda(self, node: Any, results: Dict[str, Any]) -> None:
        args: Any = results['args']
        body: Any = results['body']
        inner: Any = simplify_args(results['inner'])
        if (inner.type == token.NAME):
            inner = inner.clone()
            inner.prefix = ' '
            args.replace(inner)
            return
        params: Any = find_params(args)
        to_index: Dict[str, List[Any]] = map_to_index(params)
        tup_name: str = self.new_name(tuple_name(params))
        new_param: Any = Name(tup_name, prefix=' ')
        args.replace(new_param.clone())
        for n in body.post_order():
            if ((n.type == token.NAME) and (n.value in to_index)):
                subscripts: List[Any] = [c.clone() for c in to_index[n.value]]
                new: Any = pytree.Node(syms.power, ([new_param.clone()] + subscripts))
                new.prefix = n.prefix
                n.replace(new)

def simplify_args(node: Any) -> Any:
    if (node.type in (syms.vfplist, token.NAME)):
        return node
    elif (node.type == syms.vfpdef):
        while (node.type == syms.vfpdef):
            node = node.children[1]
        return node
    raise RuntimeError(('Received unexpected node %s' % node))

def find_params(node: Any) -> Union[str, List[Any]]:
    if (node.type == syms.vfpdef):
        return find_params(node.children[1])
    elif (node.type == token.NAME):
        return node.value
    return [find_params(c) for c in node.children if (c.type != token.COMMA)]

def map_to_index(param_list: Union[str, List[Any]], prefix: Optional[List[Any]] = None, d: Optional[Dict[str, List[Any]]] = None) -> Dict[str, List[Any]]:
    if (d is None):
        d = {}
    if prefix is None:
        prefix = []
    for (i, obj) in enumerate(param_list):
        trailer: List[Any] = [Subscript(Number(str(i)))]
        if isinstance(obj, list):
            map_to_index(obj, prefix + trailer, d=d)
        else:
            d[obj] = prefix + trailer
    return d

def tuple_name(param_list: Union[str, List[Any]]) -> str:
    l: List[str] = []
    for obj in param_list:
        if isinstance(obj, list):
            l.append(tuple_name(obj))
        else:
            l.append(obj)
    return '_'.join(l)
