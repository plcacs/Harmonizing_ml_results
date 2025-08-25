from typing import List, Dict

class BMNode:
    transition_table: Dict[str, 'BMNode']
    fixers: List
    id: int
    content: str

class BottomMatcher:
    match: set
    root: BMNode
    nodes: List[BMNode]
    fixers: List
    logger: logging.Logger

    def add_fixer(self, fixer: Any) -> None:
        ...

    def add(self, pattern: List, start: BMNode) -> List[BMNode]:
        ...

    def run(self, leaves: List) -> Dict:
        ...
