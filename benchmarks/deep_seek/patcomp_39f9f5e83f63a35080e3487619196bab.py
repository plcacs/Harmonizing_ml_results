"""Pattern compiler.

The grammar is taken from PatternGrammar.txt.

The compiler compiles a pattern to a pytree.*Pattern instance.
"""
__author__ = 'Guido van Rossum <guido@python.org>'
import io
from typing import Optional, List, Tuple, Iterator, Dict, Any, Union
from .pgen2 import driver, literals, token, tokenize, parse, grammar
from . import pytree
from . import pygram

class PatternSyntaxError(Exception):
    pass

def tokenize_wrapper(input: str) -> Iterator[Tuple[int, str, Tuple[int, int], Tuple[int, int], str]]:
    """Tokenizes a string suppressing significant whitespace."""
    skip = {token.NEWLINE, token.INDENT, token.DEDENT}
    tokens = tokenize.generate_tokens(io.StringIO(input).readline)
    for quintuple in tokens:
        type, value, start, end, line_text = quintuple
        if type not in skip:
            yield quintuple

class PatternCompiler:
    def __init__(self, grammar_file: Optional[str] = None) -> None:
        """Initializer.

        Takes an optional alternative filename for the pattern grammar.
        """
        if grammar_file is None:
            self.grammar = pygram.pattern_grammar
            self.syms = pygram.pattern_symbols
        else:
            self.grammar = driver.load_grammar(grammar_file)
            self.syms = pygram.Symbols(self.grammar)
        self.pygrammar = pygram.python_grammar
        self.pysyms = pygram.python_symbols
        self.driver = driver.Driver(self.grammar, convert=pattern_convert)

    def compile_pattern(self, input: str, debug: bool = False, with_tree: bool = False) -> Union[pytree.Pattern, Tuple[pytree.Pattern, Any]]:
        """Compiles a pattern string to a nested pytree.*Pattern object."""
        tokens = tokenize_wrapper(input)
        try:
            root = self.driver.parse_tokens(tokens, debug=debug)
        except parse.ParseError as e:
            raise PatternSyntaxError(str(e)) from None
        if with_tree:
            return (self.compile_node(root), root)
        else:
            return self.compile_node(root)

    def compile_node(self, node: Any) -> pytree.Pattern:
        """Compiles a node, recursively.

        This is one big switch on the node type.
        """
        if node.type == self.syms.Matcher:
            node = node.children[0]
        if node.type == self.syms.Alternatives:
            alts = [self.compile_node(ch) for ch in node.children[::2]]
            if len(alts) == 1:
                return alts[0]
            p = pytree.WildcardPattern([[a] for a in alts], min=1, max=1)
            return p.optimize()
        if node.type == self.syms.Alternative:
            units = [self.compile_node(ch) for ch in node.children]
            if len(units) == 1:
                return units[0]
            p = pytree.WildcardPattern([units], min=1, max=1)
            return p.optimize()
        if node.type == self.syms.NegatedUnit:
            pattern = self.compile_basic(node.children[1:])
            p = pytree.NegatedPattern(pattern)
            return p.optimize()
        assert node.type == self.syms.Unit
        name = None
        nodes = node.children
        if len(nodes) >= 3 and nodes[1].type == token.EQUAL:
            name = nodes[0].value
            nodes = nodes[2:]
        repeat = None
        if len(nodes) >= 2 and nodes[-1].type == self.syms.Repeater:
            repeat = nodes[-1]
            nodes = nodes[:-1]
        pattern = self.compile_basic(nodes, repeat)
        if repeat is not None:
            assert repeat.type == self.syms.Repeater
            children = repeat.children
            child = children[0]
            if child.type == token.STAR:
                min = 0
                max = pytree.HUGE
            elif child.type == token.PLUS:
                min = 1
                max = pytree.HUGE
            elif child.type == token.LBRACE:
                assert children[-1].type == token.RBRACE
                assert len(children) in (3, 5)
                min = max = self.get_int(children[1])
                if len(children) == 5:
                    max = self.get_int(children[3])
            else:
                assert False
            if min != 1 or max != 1:
                pattern = pattern.optimize()
                pattern = pytree.WildcardPattern([[pattern]], min=min, max=max)
        if name is not None:
            pattern.name = name
        return pattern.optimize()

    def compile_basic(self, nodes: List[Any], repeat: Optional[Any] = None) -> pytree.Pattern:
        assert len(nodes) >= 1
        node = nodes[0]
        if node.type == token.STRING:
            value = str(literals.evalString(node.value))
            return pytree.LeafPattern(_type_of_literal(value), value)
        elif node.type == token.NAME:
            value = node.value
            if value.isupper():
                if value not in TOKEN_MAP:
                    raise PatternSyntaxError(f'Invalid token: {value!r}')
                if nodes[1:]:
                    raise PatternSyntaxError("Can't have details for token")
                return pytree.LeafPattern(TOKEN_MAP[value])
            else:
                if value == 'any':
                    type = None
                elif not value.startswith('_'):
                    type = getattr(self.pysyms, value, None)
                    if type is None:
                        raise PatternSyntaxError(f'Invalid symbol: {value!r}')
                if nodes[1:]:
                    content = [self.compile_node(nodes[1].children[1])]
                else:
                    content = None
                return pytree.NodePattern(type, content)
        elif node.value == '(':
            return self.compile_node(nodes[1])
        elif node.value == '[':
            assert repeat is None
            subpattern = self.compile_node(nodes[1])
            return pytree.WildcardPattern([[subpattern]], min=0, max=1)
        assert False, node

    def get_int(self, node: Any) -> int:
        assert node.type == token.NUMBER
        return int(node.value)

TOKEN_MAP: Dict[str, Optional[int]] = {'NAME': token.NAME, 'STRING': token.STRING, 'NUMBER': token.NUMBER, 'TOKEN': None}

def _type_of_literal(value: str) -> Optional[int]:
    if value[0].isalpha():
        return token.NAME
    elif value in grammar.opmap:
        return grammar.opmap[value]
    else:
        return None

def pattern_convert(grammar: Any, raw_node_info: Tuple[int, str, Any, List[Any]]) -> Union[pytree.Node, pytree.Leaf]:
    """Converts raw node information to a Node or Leaf instance."""
    type, value, context, children = raw_node_info
    if children or type in grammar.number2symbol:
        return pytree.Node(type, children, context=context)
    else:
        return pytree.Leaf(type, value, context=context)

def compile_pattern(pattern: str) -> pytree.Pattern:
    return PatternCompiler().compile_pattern(pattern)
