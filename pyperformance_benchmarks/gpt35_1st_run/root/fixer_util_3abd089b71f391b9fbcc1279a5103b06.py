from typing import List, Union

def KeywordArg(keyword: Node, value: Leaf) -> Node:
    return Node(syms.argument, [keyword, Leaf(token.EQUAL, '='), value])

def LParen() -> Leaf:
    return Leaf(token.LPAR, '(')

def RParen() -> Leaf:
    return Leaf(token.RPAR, ')')

def Assign(target: Union[List, Node], source: Union[List, Leaf]) -> Node:
    if (not isinstance(target, list)):
        target = [target]
    if (not isinstance(source, list)):
        source.prefix = ' '
        source = [source]
    return Node(syms.atom, ((target + [Leaf(token.EQUAL, '=', prefix=' ')]) + source))

def Name(name: str, prefix: str = None) -> Leaf:
    return Leaf(token.NAME, name, prefix=prefix)

def Attr(obj: Node, attr: Leaf) -> List:
    return [obj, Node(syms.trailer, [Dot(), attr])]

def Comma() -> Leaf:
    return Leaf(token.COMMA, ',')

def Dot() -> Leaf:
    return Leaf(token.DOT, '.')

def ArgList(args: List, lparen: Leaf = LParen(), rparen: Leaf = RParen()) -> Node:
    node = Node(syms.trailer, [lparen.clone(), rparen.clone()])
    if args:
        node.insert_child(1, Node(syms.arglist, args))
    return node

def Call(func_name: Node, args: List = None, prefix: str = None) -> Node:
    node = Node(syms.power, [func_name, ArgList(args)])
    if (prefix is not None):
        node.prefix = prefix
    return node

def Newline() -> Leaf:
    return Leaf(token.NEWLINE, '\n')

def BlankLine() -> Leaf:
    return Leaf(token.NEWLINE, '')

def Number(n: str, prefix: str = None) -> Leaf:
    return Leaf(token.NUMBER, n, prefix=prefix)

def Subscript(index_node: Node) -> Node:
    return Node(syms.trailer, [Leaf(token.LBRACE, '['), index_node, Leaf(token.RBRACE, ']')]

def String(string: str, prefix: str = None) -> Leaf:
    return Leaf(token.STRING, string, prefix=prefix)

def ListComp(xp: Node, fp: Node, it: Node, test: Node = None) -> Node:
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
    return Node(syms.atom, [Leaf(token.LBRACE, '['), inner, Leaf(token.RBRACE, ']')]

def FromImport(package_name: str, name_leafs: List) -> Node:
    for leaf in name_leafs:
        leaf.remove()
    children = [Leaf(token.NAME, 'from'), Leaf(token.NAME, package_name, prefix=' '), Leaf(token.NAME, 'import', prefix=' '), Node(syms.import_as_names, name_leafs)]
    imp = Node(syms.import_from, children)
    return imp

def ImportAndCall(node: Node, results: dict, names: List) -> Node:
    obj = results['obj'].clone()
    if (obj.type == syms.arglist):
        newarglist = obj.clone()
    else:
        newarglist = Node(syms.arglist, [obj.clone()])
    after = results['after']
    if after:
        after = [n.clone() for n in after]
    new = Node(syms.power, ((Attr(Name(names[0]), Name(names[1])) + [Node(syms.trailer, [results['lpar'].clone(), newarglist, results['rpar'].clone()])]) + after)
    new.prefix = node.prefix
    return new

def is_tuple(node: Node) -> bool:
    if (isinstance(node, Node) and (node.children == [LParen(), RParen()])):
        return True
    return (isinstance(node, Node) and (len(node.children) == 3) and isinstance(node.children[0], Leaf) and isinstance(node.children[1], Node) and isinstance(node.children[2], Leaf) and (node.children[0].value == '(') and (node.children[2].value == ')'))

def is_list(node: Node) -> bool:
    return (isinstance(node, Node) and (len(node.children) > 1) and isinstance(node.children[0], Leaf) and isinstance(node.children[(- 1)], Leaf) and (node.children[0].value == '[') and (node.children[(- 1)].value == ']'))

def parenthesize(node: Node) -> Node:
    return Node(syms.atom, [LParen(), node, RParen()])
