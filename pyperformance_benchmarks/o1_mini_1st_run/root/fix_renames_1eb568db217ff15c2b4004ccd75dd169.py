from typing import Dict, Iterator, Tuple, Any
from .. import fixer_base
from ..fixer_util import Name, attr_chain

MAPPING: Dict[str, Dict[str, str]] = {'sys': {'maxint': 'maxsize'}}
LOOKUP: Dict[Tuple[str, str], str] = {}

def alternates(members: Iterator[str]) -> str:
    return '(' + '|'.join(map(repr, members)) + ')'

def build_pattern() -> Iterator[str]:
    for module, replace in list(MAPPING.items()):
        for old_attr, new_attr in list(replace.items()):
            LOOKUP[(module, old_attr)] = new_attr
            yield (
                "\n                  import_from< 'from' module_name=%r 'import'\n"
                "                      ( attr_name=%r | import_as_name< attr_name=%r 'as' any >) >\n"
                "                  " % (module, old_attr, old_attr)
            )
            yield (
                "\n                  power< module_name=%r trailer< '.' attr_name=%r > any* >\n"
                "                  " % (module, old_attr)
            )

class FixRenames(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = '|'.join(build_pattern())
    order: str = 'pre'

    def match(self, node: Any) -> Any:
        match = super(FixRenames, self).match
        results = match(node)
        if results:
            if any(match(obj) for obj in attr_chain(node, 'parent')):
                return False
            return results
        return False

    def transform(self, node: Any, results: Dict[str, Any]) -> None:
        mod_name = results.get('module_name')
        attr_name = results.get('attr_name')
        if mod_name and attr_name:
            new_attr = LOOKUP[(mod_name.value, attr_name.value)]
            attr_name.replace(Name(new_attr, prefix=attr_name.prefix))
