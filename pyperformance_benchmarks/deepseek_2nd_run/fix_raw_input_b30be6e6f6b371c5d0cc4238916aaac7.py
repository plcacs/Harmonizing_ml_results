'Fixer that changes raw_input(...) into input(...).'
from .. import fixer_base
from ..fixer_util import Name
from typing import Any, Dict

class FixRawInput(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "\n              power< name='raw_input' trailer< '(' [any] ')' > any* >\n              "

    def transform(self, node: Any, results: Dict[str, Any]) -> None:
        name: Any = results['name']
        name.replace(Name('input', prefix=name.prefix))
