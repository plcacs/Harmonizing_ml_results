from typing import Callable, Iterator, Union, Any, Optional, Tuple, List, Dict

Results: Dict[str, NL]
Convert: Callable[[Grammar, RawNode], Union[Node, Leaf]]
DFA: List[List[Tuple[int, int]]]
DFAS: Tuple[DFA, Dict[int, int]]
