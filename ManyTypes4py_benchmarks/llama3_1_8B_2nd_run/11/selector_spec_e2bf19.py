import os
import re
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple, Union
from dbt.exceptions import InvalidSelectorError
from dbt.flags import get_flags
from dbt_common.dataclass_schema import StrEnum, dbtClassMixin
from dbt_common.exceptions import DbtRuntimeError
from .graph import UniqueId
from .selector_methods import MethodName

RAW_SELECTOR_PATTERN = re.compile('\\A(?P<childrens_parents>(\\@))?(?P<parents>((?P<parents_depth>(\\d*))\\+))?((?P<method>([\\w.]+)):)?(?P<value>(.*?))(?P<children>(\\+(?P<children_depth>(\\d*))))?\\Z')
SELECTOR_METHOD_SEPARATOR = '.'

class IndirectSelection(StrEnum):
    Eager = 'eager'
    Cautious = 'cautious'
    Buildable = 'buildable'
    Empty = 'empty'

def _probably_path(value: str) -> bool:
    """Decide if the value is probably a path. Windows has two path separators, so
    we should check both sep ('\\') and altsep ('/') there.
    """
    if os.path.sep in value:
        return True
    elif os.path.altsep is not None and os.path.altsep in value:
        return True
    else:
        return False

def _match_to_int(match: re.Match, key: str) -> Optional[int]:
    raw = match.get(key)
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise DbtRuntimeError(f'Invalid node spec - could not handle parent depth {raw}') from exc

SelectionSpec = Union['SelectionCriteria', 'SelectionIntersection', 'SelectionDifference', 'SelectionUnion']

@dataclass
class SelectionCriteria:
    indirect_selection: IndirectSelection
    raw: Optional[str]

    def __post_init__(self):
        if self.children and self.childrens_parents:
            raise DbtRuntimeError(f'Invalid node spec {self.raw} - "@" prefix and "+" suffix are incompatible')

    @classmethod
    def default_method(cls, value: str) -> MethodName:
        if _probably_path(value):
            return MethodName.Path
        elif value.lower().endswith(('.sql', '.py', '.csv')):
            return MethodName.File
        else:
            return MethodName.FQN

    @classmethod
    def parse_method(cls, groupdict: Dict[str, str]) -> Tuple[MethodName, List[str]]:
        raw_method = groupdict.get('method')
        if raw_method is None:
            return (cls.default_method(groupdict['value']), [])
        method_parts = raw_method.split(SELECTOR_METHOD_SEPARATOR)
        try:
            method_name = MethodName(method_parts[0])
        except ValueError as exc:
            raise InvalidSelectorError(f"'{method_parts[0]}' is not a valid method name") from exc
        method_arguments = method_parts[1:]
        return (method_name, method_arguments)

    @classmethod
    def selection_criteria_from_dict(cls, raw: str, dct: Dict[str, str]) -> 'SelectionCriteria':
        if 'value' not in dct:
            raise DbtRuntimeError(f'Invalid node spec "{raw}" - no search value!')
        method_name, method_arguments = cls.parse_method(dct)
        parents_depth = _match_to_int(dct, 'parents_depth')
        children_depth = _match_to_int(dct, 'children_depth')
        indirect_selection = IndirectSelection(dct.get('indirect_selection', get_flags().INDIRECT_SELECTION))
        return cls(raw=raw, method=method_name, method_arguments=method_arguments, value=dct['value'], childrens_parents=bool(dct.get('childrens_parents')), parents=bool(dct.get('parents')), parents_depth=parents_depth, children=bool(dct.get('children')), children_depth=children_depth, indirect_selection=indirect_selection)

    @classmethod
    def dict_from_single_spec(cls, raw: str) -> Dict[str, Any]:
        result = RAW_SELECTOR_PATTERN.match(raw)
        if result is None:
            return {'error': 'Invalid selector spec'}
        dct = result.groupdict()
        method_name, method_arguments = cls.parse_method(dct)
        meth_name = str(method_name)
        if method_arguments:
            meth_name += '.' + '.'.join(method_arguments)
        dct['method'] = meth_name
        dct = {k: v for k, v in dct.items() if v is not None and v != ''}
        if 'childrens_parents' in dct:
            dct['childrens_parents'] = bool(dct.get('childrens_parents'))
        if 'parents' in dct:
            dct['parents'] = bool(dct.get('parents'))
        if 'children' in dct:
            dct['children'] = bool(dct.get('children'))
        return dct

    @classmethod
    def from_single_spec(cls, raw: str) -> 'SelectionCriteria':
        result = RAW_SELECTOR_PATTERN.match(raw)
        if result is None:
            raise DbtRuntimeError(f'Invalid selector spec "{raw}"')
        return cls.selection_criteria_from_dict(raw, result.groupdict())

class BaseSelectionGroup(dbtClassMixin, Iterable[SelectionSpec], metaclass=ABCMeta):
    def __init__(self, components: List[SelectionSpec], indirect_selection: IndirectSelection = IndirectSelection.Eager, expect_exists: bool = False, raw: Optional[str] = None):
        self.components = list(components)
        self.expect_exists = expect_exists
        self.raw = raw
        self.indirect_selection = indirect_selection

    def __iter__(self) -> Iterator[SelectionSpec]:
        for component in self.components:
            yield component

    @abstractmethod
    def combine_selections(self, selections: Iterable[SelectionSpec]) -> Set[SelectionSpec]:
        raise NotImplementedError('_combine_selections not implemented!')

    def combined(self, selections: Iterable[SelectionSpec]) -> Set[SelectionSpec]:
        if not selections:
            return set()
        return self.combine_selections(selections)

class SelectionIntersection(BaseSelectionGroup):
    def combine_selections(self, selections: Iterable[SelectionSpec]) -> Set[SelectionSpec]:
        return set.intersection(*selections)

class SelectionDifference(BaseSelectionGroup):
    def combine_selections(self, selections: Iterable[SelectionSpec]) -> Set[SelectionSpec]:
        return set.difference(*selections)

class SelectionUnion(BaseSelectionGroup):
    def combine_selections(self, selections: Iterable[SelectionSpec]) -> Set[SelectionSpec]:
        return set.union(*selections)
