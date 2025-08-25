'Pattern compiler.\n\nThe grammar is taken from PatternGrammar.txt.\n\nThe compiler compiles a pattern to a pytree.*Pattern instance.\n'
__author__ = 'Guido van Rossum <guido@python.org>'
import io
from typing import Optional, Iterator, Tuple, Any, List, Union, Dict
from .pgen2 import driver, literals, token, tokenize, parse, grammar
from . import pytree
from . import pygram

class PatternSyntaxError(Exception):
    pass

def tokenize_wrapper(input: str) -> Iterator[Tuple[int, str, Tuple[int, int], Tuple[int, int], str]]:
    'Tokenizes a string suppressing significant whitespace.'
    skip: set = {token.NEWLINE, token.INDENT, token.DEDENT}
    tokens = tokenize.generate_tokens(io.StringIO(input).readline)
    for quintuple in tokens:
        (type_, value, start, end, line_text) = quintuple
        if type_ not in skip:
            yield quintuple

class PatternCompiler:
    def __init__(self, grammar_file: Optional[str] = None) -> None:
        'Initializer.\n\n        Takes an optional alternative filename for the pattern grammar.\n        '
        if grammar_file is None:
            self.grammar: Any = pygram.pattern_grammar
            self.syms: Any = pygram.pattern_symbols
        else:
            self.grammar = driver.load_grammar(grammar_file)
            self.syms = pygram.Symbols(self.grammar)
        self.pygrammar: Any = pygram.python_grammar
        self.pysyms: Any = pygram.python_symbols
        self.driver: driver.Driver = driver.Driver(self.grammar, convert=pattern_convert)

    def compile_pattern(self, input: str, debug: bool = False, with_tree: bool = False) -> Union[pytree.Pattern, Tuple[pytree.Pattern, Any]]:
        'Compiles a pattern string to a nested pytree.*Pattern object.'
        tokens = tokenize_wrapper(input)
        try:
            root = self.driver.parse_tokens(tokens, debug=debug)
        except parse.ParseError as e:
            raise PatternSyntaxError(str(e)) from None
        if with_tree:
            return (self.compile_node(root), root)
        else:
            return self.compile_node(root)

    def compile_node(self, node: Union[pytree.Node, pytree.Leaf]) -> pytree.Pattern:
        'Compiles a node, recursively.\n\n        This is one big switch on the node type.\n        '
        if node.type == self.syms.Matcher:
            node = node.children[0]
        if node.type == self.syms.Alternatives:
            alts: List[pytree.Pattern] = [self.compile_node(ch) for ch in node.children[::2]]
            if len(alts) == 1:
                return alts[0]
            p: pytree.WildcardPattern = pytree.WildcardPattern([[a] for a in alts], min=1, max=1)
            return p.optimize()
        if node.type == self.syms.Alternative:
            units: List[pytree.Pattern] = [self.compile_node(ch) for ch in node.children]
            if len(units) == 1:
                return units[0]
            p: pytree.WildcardPattern = pytree.WildcardPattern([units], min=1, max=1)
            return p.optimize()
        if node.type == self.syms.NegatedUnit:
            pattern: pytree.Pattern = self.compile_basic(node.children[1:])
            p: pytree.NegatedPattern = pytree.NegatedPattern(pattern)
            return p.optimize()
        assert node.type == self.syms.Unit
        name: Optional[str] = None
        nodes: List[Union[pytree.Node, pytree.Leaf]] = node.children
        if len(nodes) >= 3 and nodes[1].type == token.EQUAL:
            name = nodes[0].value
            nodes = nodes[2:]
        repeat: Optional[Union[pytree.Node, pytree.Leaf]] = None
        if len(nodes) >= 2 and nodes[-1].type == self.syms.Repeater:
            repeat = nodes[-1]
            nodes = nodes[:-1]
        pattern: pytree.Pattern = self.compile_basic(nodes, repeat)
        if repeat is not None:
            assert repeat.type == self.syms.Repeater
            children = repeat.children
            child = children[0]
            if child.type == token.STAR:
                min_val: int = 0
                max_val: int = pytree.HUGE
            elif child.type == token.PLUS:
                min_val = 1
                max_val = pytree.HUGE
            elif child.type == token.LBRACE:
                assert children[-1].type == token.RBRACE
                assert len(children) in (3, 5)
                min_val: int = self.get_int(children[1])
                max_val: int = min_val
                if len(children) == 5:
                    max_val = self.get_int(children[3])
            else:
                assert False
            if (min_val != 1) or (max_val != 1):
                pattern = pattern.optimize()
                pattern = pytree.WildcardPattern([[pattern]], min=min_val, max=max_val)
        if name is not None:
            pattern.name = name
        return pattern.optimize()

    def compile_basic(self, nodes: List[Union[pytree.Node, pytree.Leaf]], repeat: Optional[Union[pytree.Node, pytree.Leaf]] = None) -> pytree.Pattern:
        assert len(nodes) >= 1
        node: Union[pytree.Node, pytree.Leaf] = nodes[0]
        if node.type == token.STRING:
            value: str = str(literals.evalString(node.value))
            return pytree.LeafPattern(_type_of_literal(value), value)
        elif node.type == token.NAME:
            value: str = node.value
            if value.isupper():
                if value not in TOKEN_MAP:
                    raise PatternSyntaxError(f"Invalid token: {value!r}")
                if nodes[1:]:
                    raise PatternSyntaxError("Can't have details for token")
                return pytree.LeafPattern(TOKEN_MAP[value])
            else:
                if value == 'any':
                    type_val: Optional[int] = None
                elif not value.startswith('_'):
                    type_val = getattr(self.pysyms, value, None)
                    if type_val is None:
                        raise PatternSyntaxError(f"Invalid symbol: {value!r}")
                else:
                    type_val = None
                if nodes[1:]:
                    content: Optional[List[pytree.Pattern]] = [self.compile_node(nodes[1].children[1])]
                else:
                    content = None
                return pytree.NodePattern(type_val, content)
        elif node.value == '(':
            return self.compile_node(nodes[1])
        elif node.value == '[':
            assert repeat is None
            subpattern: pytree.Pattern = self.compile_node(nodes[1])
            return pytree.WildcardPattern([[subpattern]], min=0, max=1)
        assert False, node

    def get_int(self, node: Union[pytree.Node, pytree.Leaf]) -> int:
        assert node.type == token.NUMBER
        return int(node.value)

TOKEN_MAP: Dict[str, Optional[int]] = {
    'NAME': token.NAME,
    'STRING': token.STRING,
    'NUMBER': token.NUMBER,
    'TOKEN': None
}

def _type_of_literal(value: str) -> Optional[int]:
    if value[0].isalpha():
        return token.NAME
    elif value in grammar.opmap:
        return grammar.opmap[value]
    else:
        return None

def pattern_convert(grammar: Any, raw_node_info: Tuple[int, Any, Any, Any]) -> Union[pytree.Node, pytree.Leaf]:
    'Converts raw node information to a Node or Leaf instance.'
    (type_, value, context, children) = raw_node_info
    if children or (type_ in grammar.number2symbol):
        return pytree.Node(type_, children, context=context)
    else:
        return pytree.Leaf(type_, value, context=context)

def compile_pattern(pattern: str) -> pytree.Pattern:
    return PatternCompiler().compile_pattern(pattern)
