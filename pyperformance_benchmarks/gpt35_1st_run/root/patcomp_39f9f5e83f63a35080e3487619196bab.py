from typing import Generator, Tuple, Union

class PatternCompiler:
    def __init__(self, grammar_file: str = None) -> None:
        ...
    
    def compile_pattern(self, input: str, debug: bool = False, with_tree: bool = False) -> Union[pytree.Pattern, Tuple[pytree.Pattern, pytree.Node]]:
        ...
    
    def compile_node(self, node: pytree.Node) -> pytree.Pattern:
        ...
    
    def compile_basic(self, nodes: List[pytree.Node], repeat: pytree.Node = None) -> pytree.Pattern:
        ...
    
    def get_int(self, node: pytree.Node) -> int:
        ...
