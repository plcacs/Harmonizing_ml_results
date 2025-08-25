'Fixer for __nonzero__ -> __bool__ methods.'
from .. import fixer_base
from ..fixer_util import Name
from typing import Any, Dict, Optional
from lib2to3.fixer_base import BaseFix
from lib2to3.pytree import Node

class FixNonzero(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = """
        classdef< 'class' any+ ':' 
                  suite< any*
                         funcdef< 'def' name='__nonzero__' 
                                  parameters< '(' NAME ')' > any+ >
                         any* > >
    """

    def transform(self, node: Node, results: Dict[str, Any]) -> Optional[Node]:
        name = results['name']
        new = Name('__bool__', prefix=name.prefix)
        name.replace(new)
