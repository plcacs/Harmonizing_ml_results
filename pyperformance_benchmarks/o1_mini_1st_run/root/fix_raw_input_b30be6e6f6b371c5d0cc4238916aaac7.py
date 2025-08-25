from .. import fixer_base
from ..fixer_util import Name
from typing import Any, Dict, Optional
from lib2to3.pytree import Node

class FixRawInput(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = """
              power< name='raw_input' trailer< '(' [any] ')' > any* >
              """

    def transform(self, node: Node, results: Dict[str, Any]) -> Optional[Node]:
        name = results['name']
        name.replace(Name('input', prefix=name.prefix))
