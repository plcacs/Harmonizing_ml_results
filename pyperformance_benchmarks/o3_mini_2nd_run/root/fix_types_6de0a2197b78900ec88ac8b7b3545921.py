from typing import Dict, List, Any, Optional
from .. import fixer_base
from ..fixer_util import Name

_TYPE_MAPPING: Dict[str, str] = {
    'BooleanType': 'bool', 
    'BufferType': 'memoryview', 
    'ClassType': 'type', 
    'ComplexType': 'complex', 
    'DictType': 'dict', 
    'DictionaryType': 'dict', 
    'EllipsisType': 'type(Ellipsis)', 
    'FloatType': 'float', 
    'IntType': 'int', 
    'ListType': 'list', 
    'LongType': 'int', 
    'ObjectType': 'object', 
    'NoneType': 'type(None)', 
    'NotImplementedType': 'type(NotImplemented)', 
    'SliceType': 'slice', 
    'StringType': 'bytes', 
    'StringTypes': '(str,)', 
    'TupleType': 'tuple', 
    'TypeType': 'type', 
    'UnicodeType': 'str', 
    'XRangeType': 'range'
}

_pats: List[str] = [("power< 'types' trailer< '.' name='%s' > >" % t) for t in _TYPE_MAPPING]

class FixTypes(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = '|'.join(_pats)

    def transform(self, node: Any, results: Dict[str, Any]) -> Optional[Name]:
        new_value: Optional[str] = _TYPE_MAPPING.get(results['name'].value)
        if new_value:
            return Name(new_value, prefix=node.prefix)
        return None