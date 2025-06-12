from typing import Any, Generator, List, Optional, Union
from .pgen2 import token
from .pytree import Leaf, Node
from .pygram import python_symbols as syms
from . import patcomp

def KeywordArg(keyword: Leaf, value: Leaf) -> Node:
    return Node(syms.argument, [keyword, Leaf(token.EQUAL, '='), value])

def LParen() -> Leaf:
    return Leaf(token.LPAR, '(')

def RParen() -> Leaf:
    return Leaf(token.RPAR, ')')

def Assign(target: Union[Leaf, List[Leaf]], source: Union[Leaf, List[Leaf]]) -> Node:
    if not isinstance(target, list):
        target = [target]
    if not isinstance(source, list):
        source.prefix = ' '
        source = [source]
    return Node(syms.atom, target + [Leaf(token.EQUAL, '=', prefix=' ')] + source)

def Name(name: str, prefix: Optional[str] = None) -> Leaf:
    return Leaf(token.NAME, name, prefix=prefix)

def Attr(obj: Leaf, attr: Leaf) -> List[Union[Leaf, Node]]:
    return [obj, Node(syms.trailer, [Dot(), attr])]

def Comma() -> Leaf:
    return Leaf(token.COMMA, ',')

def Dot() -> Leaf:
    return Leaf(token.DOT, '.')

def ArgList(args: Optional[List[Leaf]], lparen: Leaf = LParen(), rparen: Leaf = RParen()) -> Node:
    node = Node(syms.trailer, [lparen.clone(), rparen.clone()])
    if args:
        node.insert_child(1, Node(syms.arglist, args))
    return node

def Call(func_name: Leaf, args: Optional[List[Leaf]] = None, prefix: Optional[str] = None) -> Node:
    node = Node(syms.power, [func_name, ArgList(args)])
    if prefix is not None:
        node.prefix = prefix
    return node

def Newline() -> Leaf:
    return Leaf(token.NEWLINE, '\n')

def BlankLine() -> Leaf:
    return Leaf(token.NEWLINE, '')

def Number(n: str, prefix: Optional[str] = None) -> Leaf:
    return Leaf(token.NUMBER, n, prefix=prefix)

def Subscript(index_node: Leaf) -> Node:
    return Node(syms.trailer, [Leaf(token.LBRACE, '['), index_node, Leaf(token.RBRACE, ']')])

def String(string: str, prefix: Optional[str] = None) -> Leaf:
    return Leaf(token.STRING, string, prefix=prefix)

def ListComp(xp: Leaf, fp: Leaf, it: Leaf, test: Optional[Leaf] = None) -> Node:
    xp.prefix = ''
    fp.prefix = ' '
    it.prefix = ' '
    for_leaf = Leaf(token.NAME, 'for')
    for_leaf.prefix = ' '
    in_leaf = Leaf(token.NAME, 'in')
    in_leaf.prefix = ' '
    inner_args = [for_leaf, fp, in_leaf, it]
    if test:
        test.prefix = ' '
        if_leaf = Leaf(token.NAME, 'if')
        if_leaf.prefix = ' '
        inner_args.append(Node(syms.comp_if, [if_leaf, test]))
    inner = Node(syms.listmaker, [xp, Node(syms.comp_for, inner_args)])
    return Node(syms.atom, [Leaf(token.LBRACE, '['), inner, Leaf(token.RBRACE, ']')])

def FromImport(package_name: str, name_leafs: List[Leaf]) -> Node:
    for leaf in name_leafs:
        leaf.remove()
    children = [Leaf(token.NAME, 'from'), Leaf(token.NAME, package_name, prefix=' '), Leaf(token.NAME, 'import', prefix=' '), Node(syms.import_as_names, name_leafs)]
    imp = Node(syms.import_from, children)
    return imp

def ImportAndCall(node: Node, results: dict, names: List[str]) -> Node:
    obj = results['obj'].clone()
    if obj.type == syms.arglist:
        newarglist = obj.clone()
    else:
        newarglist = Node(syms.arglist, [obj.clone()])
    after = results['after']
    if after:
        after = [n.clone() for n in after]
    new = Node(syms.power, Attr(Name(names[0]), Name(names[1])) + [Node(syms.trailer, [results['lpar'].clone(), newarglist, results['rpar'].clone()])] + after)
    new.prefix = node.prefix
    return new

def is_tuple(node: Node) -> bool:
    if isinstance(node, Node) and (node.children == [LParen(), RParen()]):
        return True
    return (isinstance(node, Node) and len(node.children) == 3 and isinstance(node.children[0], Leaf) and isinstance(node.children[1], Node) and isinstance(node.children[2], Leaf) and node.children[0].value == '(' and node.children[2].value == ')')

def is_list(node: Node) -> bool:
    return (isinstance(node, Node) and len(node.children) > 1 and isinstance(node.children[0], Leaf) and isinstance(node.children[-1], Leaf) and node.children[0].value == '[' and node.children[-1].value == ']')

def parenthesize(node: Node) -> Node:
    return Node(syms.atom, [LParen(), node, RParen()])

consuming_calls = {'sorted', 'list', 'set', 'any', 'all', 'tuple', 'sum', 'min', 'max', 'enumerate'}

def attr_chain(obj: Any, attr: str) -> Generator[Any, None, None]:
    next = getattr(obj, attr)
    while next:
        yield next
        next = getattr(next, attr)

p0 = "for_stmt< 'for' any 'in' node=any ':' any* >\n        | comp_for< 'for' any 'in' node=any any* >\n     "
p1 = "\npower<\n    ( 'iter' | 'list' | 'tuple' | 'sorted' | 'set' | 'sum' |\n      'any' | 'all' | 'enumerate' | (any* trailer< '.' 'join' >) )\n    trailer< '(' node=any ')' >\n    any*\n>\n"
p2 = "\npower<\n    ( 'sorted' | 'enumerate' )\n    trailer< '(' arglist<node=any any*> ')' >\n    any*\n>\n"
pats_built = False

def in_special_context(node: Node) -> bool:
    global p0, p1, p2, pats_built
    if not pats_built:
        p0 = patcomp.compile_pattern(p0)
        p1 = patcomp.compile_pattern(p1)
        p2 = patcomp.compile_pattern(p2)
        pats_built = True
    patterns = [p0, p1, p2]
    for pattern, parent in zip(patterns, attr_chain(node, 'parent')):
        results = {}
        if pattern.match(parent, results) and results['node'] is node:
            return True
    return False

def is_probably_builtin(node: Node) -> bool:
    prev = node.prev_sibling
    if prev is not None and prev.type == token.DOT:
        return False
    parent = node.parent
    if parent.type in (syms.funcdef, syms.classdef):
        return False
    if parent.type == syms.expr_stmt and parent.children[0] is node:
        return False
    if parent.type == syms.parameters or (parent.type == syms.typedargslist and ((prev is not None and prev.type == token.COMMA) or parent.children[0] is node)):
        return False
    return True

def find_indentation(node: Node) -> str:
    while node is not None:
        if node.type == syms.suite and len(node.children) > 2:
            indent = node.children[1]
            if indent.type == token.INDENT:
                return indent.value
        node = node.parent
    return ''

def make_suite(node: Node) -> Node:
    if node.type == syms.suite:
        return node
    node = node.clone()
    parent, node.parent = node.parent, None
    suite = Node(syms.suite, [node])
    suite.parent = parent
    return suite

def find_root(node: Node) -> Node:
    while node.type != syms.file_input:
        node = node.parent
        if not node:
            raise ValueError('root found before file_input node was found.')
    return node

def does_tree_import(package: Optional[str], name: str, node: Node) -> bool:
    binding = find_binding(name, find_root(node), package)
    return bool(binding)

def is_import(node: Node) -> bool:
    return node.type in (syms.import_name, syms.import_from)

def touch_import(package: Optional[str], name: str, node: Node) -> None:
    def is_import_stmt(node: Node) -> bool:
        return node.type == syms.simple_stmt and node.children and is_import(node.children[0])

    root = find_root(node)
    if does_tree_import(package, name, root):
        return
    insert_pos = offset = 0
    for idx, node in enumerate(root.children):
        if not is_import_stmt(node):
            continue
        for offset, node2 in enumerate(root.children[idx:]):
            if not is_import_stmt(node2):
                break
        insert_pos = idx + offset
        break
    if insert_pos == 0:
        for idx, node in enumerate(root.children):
            if node.type == syms.simple_stmt and node.children and node.children[0].type == token.STRING:
                insert_pos = idx + 1
                break
    if package is None:
        import_ = Node(syms.import_name, [Leaf(token.NAME, 'import'), Leaf(token.NAME, name, prefix=' ')])
    else:
        import_ = FromImport(package, [Leaf(token.NAME, name, prefix=' ')])
    children = [import_, Newline()]
    root.insert_child(insert_pos, Node(syms.simple_stmt, children))

_def_syms = {syms.classdef, syms.funcdef}

def find_binding(name: str, node: Node, package: Optional[str] = None) -> Optional[Node]:
    for child in node.children:
        ret = None
        if child.type == syms.for_stmt:
            if _find(name, child.children[1]):
                return child
            n = find_binding(name, make_suite(child.children[-1]), package)
            if n:
                ret = n
        elif child.type in (syms.if_stmt, syms.while_stmt):
            n = find_binding(name, make_suite(child.children[-1]), package)
            if n:
                ret = n
        elif child.type == syms.try_stmt:
            n = find_binding(name, make_suite(child.children[2]), package)
            if n:
                ret = n
            else:
                for i, kid in enumerate(child.children[3:]):
                    if kid.type == token.COLON and kid.value == ':':
                        n = find_binding(name, make_suite(child.children[i + 4]), package)
                        if n:
                            ret = n
        elif child.type in _def_syms and child.children[1].value == name:
            ret = child
        elif _is_import_binding(child, name, package):
            ret = child
        elif child.type == syms.simple_stmt:
            ret = find_binding(name, child, package)
        elif child.type == syms.expr_stmt:
            if _find(name, child.children[0]):
                ret = child
        if ret:
            if not package:
                return ret
            if is_import(ret):
                return ret
    return None

_block_syms = {syms.funcdef, syms.classdef, syms.trailer}

def _find(name: str, node: Node) -> Optional[Leaf]:
    nodes = [node]
    while nodes:
        node = nodes.pop()
        if node.type > 256 and node.type not in _block_syms:
            nodes.extend(node.children)
        elif node.type == token.NAME and node.value == name:
            return node
    return None

def _is_import_binding(node: Node, name: str, package: Optional[str] = None) -> Optional[Node]:
    if node.type == syms.import_name and not package:
        imp = node.children[1]
        if imp.type == syms.dotted_as_names:
            for child in imp.children:
                if child.type == syms.dotted_as_name:
                    if child.children[2].value == name:
                        return node
                elif child.type == token.NAME and child.value == name:
                    return node
        elif imp.type == syms.dotted_as_name:
            last = imp.children[-1]
            if last.type == token.NAME and last.value == name:
                return node
        elif imp.type == token.NAME and imp.value == name:
            return node
    elif node.type == syms.import_from:
        if package and str(node.children[1]).strip() != package:
            return None
        n = node.children[3]
        if package and _find('as', n):
            return None
        elif n.type == syms.import_as_names and _find(name, n):
            return node
        elif n.type == syms.import_as_name:
            child = n.children[2]
            if child.type == token.NAME and child.value == name:
                return node
        elif n.type == token.NAME and n.value == name:
            return node
        elif package and n.type == token.STAR:
            return node
    return None
