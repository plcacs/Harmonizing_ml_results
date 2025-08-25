from .. import fixer_base
from ..fixer_util import Name
from typing import Any

class FixRawInput(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "\n              power< name: Name='raw_input' trailer< '(' [any] ')' > any* >\n              "

    def transform(self, node: Any, results: Any) -> None:
        name: Name = results['name']
        name.replace(Name('input', prefix=name.prefix))
