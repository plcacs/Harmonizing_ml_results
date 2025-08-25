from .. import fixer_base
from ..fixer_util import Name
from typing import Any

class FixBuffer(fixer_base.BaseFix):
    BM_compatible: bool = True
    explicit: bool = True
    PATTERN: str = "\n              power< name: Name='buffer' trailer< '(' [Any] ')' > Any* >\n              "

    def transform(self, node: Any, results: Any) -> None:
        name: Name = results['name']
        name.replace(Name('memoryview', prefix=name.prefix))
