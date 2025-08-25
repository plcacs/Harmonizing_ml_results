'Fixer that changes buffer(...) into memoryview(...).'
from .. import fixer_base
from ..fixer_util import Name
from typing import Dict, Any, Optional

class FixBuffer(fixer_base.BaseFix):
    BM_compatible: bool = True
    explicit: bool = True
    PATTERN: str = "\n              power< name='buffer' trailer< '(' [any] ')' > any* >\n              "

    def transform(self, node: Any, results: Dict[str, Any]) -> Optional[Any]:
        name = results['name']
        name.replace(Name('memoryview', prefix=name.prefix))
