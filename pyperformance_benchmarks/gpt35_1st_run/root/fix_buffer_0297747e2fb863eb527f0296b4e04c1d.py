from typing import Any
from lib2to3.fixer_base import BaseFix
from lib2to3.fixes import fixer_util
from lib2to3.fixer_util import Name

class FixBuffer(BaseFix):
    BM_compatible: bool = True
    explicit: bool = True
    PATTERN: str = "\n              power< name: Name='buffer' trailer< '(' [Any] ')' > Any* >\n              "

    def transform(self, node: Any, results: Any) -> None:
        name: Name = results['name']
        name.replace(Name('memoryview', prefix=name.prefix))
