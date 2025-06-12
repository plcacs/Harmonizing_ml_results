"""Fixer for removing uses of the types module.

These work for only the known names in the types module.  The forms above
can include types. or not.  ie, It is assumed the module is imported either as:

    import types
    from types import ... # either * or specific types

The import statements are not modified.

There should be another fixer that handles at least the following constants:

   type([]) -> list
   type(()) -> tuple
   type('') -> str
"""
from typing import Dict, List, Optional, Any
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
_pats: List[str] = [f"power< 'types' trailer< '.' name='{t}' > >" for t in _TYPE_MAPPING]

class FixTypes(fixer_base.BaseFix):
    BM_compatible: bool = True
    PATTERN: str = '|'.join(_pats)

    def transform(self, node: Any, results: Dict[str, Any]) -> Optional[Name]:
        new_value: Optional[str] = _TYPE_MAPPING.get(results['name'].value)
        if new_value:
            return Name(new_value, prefix=node.prefix)
        return None
