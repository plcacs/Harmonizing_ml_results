'Fix incompatible renames\n\nFixes:\n  * sys.maxint -> sys.maxsize\n'
from typing import Dict, Iterator, List, Optional, Tuple, Union
from .. import fixer_base
from ..fixer_util import Name, attr_chain

MAPPING: Dict[str, Dict[str, str]] = {'sys': {'maxint': 'maxsize'}}
LOOKUP: Dict[Tuple[str, str], str] = {}

def alternates(members: List[str]) -> str:
    return (('(' + '|'.join(map(repr, members))) + ')')

def build_pattern() -> Iterator[str]:
    for (module, replace) in list(MAPPING.items()):
        for (old_attr, new_attr) in list(replace.items()):
            LOOKUP[(module, old_attr)] = new_attr
            (yield ("\n                  import_from< 'from' module_name=%r 'import'\n                      ( attr_name=%r | import_as_name< attr_name=%r 'as' any >) >\n                  " % (module, old_attr, old_attr)))
            (yield ("\n                  power< module_name=%r trailer< '.' attr_name=%r > any* >\n                  " % (module, old_attr)))

class FixRenames(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = '|'.join(build_pattern())
    order: str = 'pre'

    def match(self, node: fixer_base.Node) -> Union[Dict[str, fixer_base.Node], bool]:
        match = super(FixRenames, self).match
        results = match(node)
        if results:
            if any((match(obj) for obj in attr_chain(node, 'parent'))):
                return False
            return results
        return False

    def transform(self, node: fixer_base.Node, results: Dict[str, fixer_base.Node]) -> None:
        mod_name: Optional[fixer_base.Node] = results.get('module_name')
        attr_name: Optional[fixer_base.Node] = results.get('attr_name')
        if (mod_name and attr_name):
            new_attr: str = LOOKUP[(mod_name.value, attr_name.value)]
            attr_name.replace(Name(new_attr, prefix=attr_name.prefix))
