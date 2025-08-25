from .. import fixer_base
from ..fixer_util import Name
from typing import Dict, Any, Optional

class FixGetcwdu(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = "\n              power< 'os' trailer< dot='.' name='getcwdu' > any* >\n              "

    def transform(self, node: Any, results: Dict[str, Any]) -> Optional[Any]:
        name = results['name']
        name.replace(Name('getcwd', prefix=name.prefix))
