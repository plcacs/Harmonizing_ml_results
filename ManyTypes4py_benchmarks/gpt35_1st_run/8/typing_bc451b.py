import abc
import os
import random
import string
import sys
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from enum import Enum
from itertools import count
from types import FrameType
from typing import Any, Callable, ClassVar, Dict, Iterator, List, NamedTuple, Optional, Set, Tuple, Type, TypeVar, Union
from mode.utils.objects import DICT_TYPES, LIST_TYPES, SET_TYPES, TUPLE_TYPES, cached_property, is_optional, is_union, qualname
from mode.utils.typing import Counter
from faust.types.models import CoercionHandler, CoercionMapping, IsInstanceArgT, ModelT
from faust.utils import codegen
from faust.utils.functional import translate
from faust.utils.iso8601 import parse as parse_iso8601
from faust.utils.json import str_to_decimal

__all__: List[str] = ['NodeType', 'TypeExpression']

T = TypeVar('T')
MISSING: object = object()
TUPLE_NAME_COUNTER: count = count(0)
JSON_TYPES: Tuple[type, ...] = (str, list, dict, int, float, Decimal)
LITERAL_TYPES: Tuple[type, ...] = (str, bytes, float, int)
DEBUG: bool = bool(os.environ.get('TYPEXPR_DEBUG', False))
QUALNAME_TRANSLATION_TABLE: Dict[str, str] = {'.': '__', '@': '__', '>': '', '<': ''}

def qualname_to_identifier(s: str) -> str:
    return translate(QUALNAME_TRANSLATION_TABLE, s)

_getframe = getattr(sys, '_getframe')

class NodeType(Enum):
    ROOT: str = 'ROOT'
    UNION: str = 'UNION'
    ANY: str = 'ANY'
    LITERAL: str = 'LITERAL'
    DATETIME: str = 'DATETIME'
    DECIMAL: str = 'DECIMAL'
    NAMEDTUPLE: str = 'NAMEDTUPLE'
    TUPLE: str = 'TUPLE'
    SET: str = 'SET'
    DICT: str = 'DICT'
    LIST: str = 'LIST'
    MODEL: str = 'MODEL'
    USER: str = 'USER'

USER_TYPES: Set[NodeType] = frozenset({NodeType.DATETIME, NodeType.DECIMAL, NodeType.USER, NodeType.MODEL})
GENERIC_TYPES: Set[NodeType] = frozenset({NodeType.TUPLE, NodeType.SET, NodeType.DICT, NodeType.LIST, NodeType.NAMEDTUPLE})
NONFIELD_TYPES: Set[NodeType] = frozenset({NodeType.NAMEDTUPLE, NodeType.MODEL, NodeType.USER})

class TypeInfo(NamedTuple):
    type: Type
    args: Tuple
    is_optional: bool

def _TypeInfo_from_type(typ: Type, *, optional: bool = False) -> TypeInfo:
    return TypeInfo(type=typ, args=tuple(getattr(typ, '__args__', None) or ()), is_optional=optional

class Variable:

    def __init__(self, name: str, *, getitem: Optional[str] = None):
        self.name: str = name
        self.getitem: Optional[str] = getitem

    def __str__(self) -> str:
        if self.getitem is not None:
            return f'{self.name}[{self.getitem}]'
        else:
            return self.name

    def __repr__(self) -> str:
        return f'<{type(self).__name__}: {self}>'

    def __getitem__(self, name: str) -> 'Variable':
        return self.clone(getitem=name)

    def clone(self, *, name: Optional[str] = None, getitem: object = MISSING) -> 'Variable':
        return type(self)(name=name if name is not None else self.name, getitem=getitem if getitem is not MISSING else self.getitem)

    def next_identifier(self) -> 'Variable':
        name: str = self.name
        next_ord: int = ord(name[-1]) + 1
        if next_ord > 122:
            name = name + 'a'
        return self.clone(name=name[:-1] + chr(next_ord), getitem=None)

class Node(abc.ABC):
    BUILTIN_TYPES: Dict[Type, 'Node'] = {}
    use_origin: bool = False

    def __init_subclass__(cls):
        cls._register()

    @classmethod
    def _register(cls):
        cls.BUILTIN_TYPES[cls.type] = cls

    @classmethod
    def create_if_compatible(cls, typ: Type, *, root: 'RootNode') -> Optional['Node']:
        if cls.compatible_types:
            target_type: Type = typ
            if cls.use_origin:
                target_type = getattr(typ, '__origin__', None) or typ
            if cls._issubclass(target_type, cls.compatible_types):
                return cls(typ, root=root)
        return None

    @classmethod
    def _issubclass(cls, typ: Type, types: Union[Type, Tuple[Type, ...]]) -> bool:
        try:
            return issubclass(typ, types)
        except (AttributeError, TypeError):
            return False

    @classmethod
    def inspect_type(cls, typ: Type) -> TypeInfo:
        optional: bool = is_optional(typ)
        if optional:
            args: Tuple = getattr(typ, '__args__', ())
            union_args: List[Type] = []
            found_none: bool = False
            for arg in args:
                if _is_NoneType(arg):
                    found_none = True
                else:
                    union_args.append(arg)
            if len(union_args) == 1:
                assert found_none
                return _TypeInfo_from_type(union_args[0], optional=True)
            return _TypeInfo_from_type(typ, optional=True)
        return _TypeInfo_from_type(typ, optional=False)

    def __init__(self, expr: Type, root: 'RootNode') -> None:
        assert root is not None
        assert root.type is NodeType.ROOT
        self.expr: Type = expr
        self.root: 'RootNode' = root
        self.root.type_stats[self.type] += 1
        assert self.root.type_stats[NodeType.ROOT] == 1
        self.__post_init__()
        if DEBUG:
            print(f'NODE {self!r}')

    def __post_init__(self) -> None:
        ...

    def random_identifier(self, n: int = 8) -> str:
        return ''.join((random.choice(string.ascii_letters) for _ in range(n)))

    @abc.abstractmethod
    def build(self, var: Variable, *args: Type) -> str:
        ...

    def __repr__(self) -> str:
        return f'<{type(self).__name__}: {self.expr!r}'

class AnyNode(Node):
    type: NodeType = NodeType.ANY
    compatible_types: Tuple[Type, ...] = ()

    @classmethod
    def create_if_compatible(cls, typ: Type, *, root: 'RootNode') -> Optional['Node']:
        if typ is Any:
            return cls(typ, root=root)
        return None

    def build(self, var: Variable, *args: Type) -> str:
        return f'{var}'

class UnionNode(Node):
    type: NodeType = NodeType.UNION
    compatible_types: Tuple[Type, ...] = ()
    use_origin: bool = True

    @classmethod
    def _maybe_unroll_union(cls, info: TypeInfo) -> TypeInfo:
        if is_union(info.type):
            assert len(info.args) > 1
            if cls._all_types_match(ModelT, info.args):
                return _TypeInfo_from_type(ModelT, optional=info.is_optional)
            elif cls._all_types_match(LITERAL_TYPES, info.args):
                return _TypeInfo_from_type(info.args[0], optional=info.is_optional)
        return info

    @classmethod
    def create_if_compatible(cls, typ: Type, *, root: 'RootNode') -> Optional['Node']:
        if is_union(typ):
            return cls(typ, root=root)
        return None

    @classmethod
    def _all_types_match(cls, typ: Type, union_args: Tuple[Type, ...]) -> bool:
        return all((cls._issubclass(x, typ) for x in cls._filter_NoneType(union_args)))

    @classmethod
    def _filter_NoneType(cls, union_args: Tuple[Type, ...]) -> Iterator[Type]:
        return (x for x in union_args if not _is_NoneType(x))

    def build(self, var: Variable, *args: Type) -> str:
        raise NotImplementedError(f'Union of types {args!r} not supported')

class LiteralNode(Node):
    type: NodeType = NodeType.LITERAL
    compatible_types: Tuple[Type, ...] = LITERAL_TYPES

    def build(self, var: Variable, *args: Type) -> str:
        return f'{var}'

class DecimalNode(Node):
    type: NodeType = NodeType.DECIMAL
    compatible_types: Tuple[Type, ...] = (Decimal,)

    def __post_init__(self) -> None:
        self.root.found_types[self.type].add(self.expr)

    def build(self, var: Variable, *args: Type) -> str:
        self.root.add_closure('_Decimal_', '__Decimal__', self._maybe_coerce)
        return f'_Decimal_({var})'

    @staticmethod
    def _maybe_coerce(value: Optional[Any] = None) -> Optional[Decimal]:
        if value is not None:
            if not isinstance(value, Decimal):
                return str_to_decimal(value)
            return value
        return None

class DatetimeNode(Node):
    type: NodeType = NodeType.DATETIME
    compatible_types: Tuple[Type, ...] = (datetime,)

    def __post_init__(self) -> None:
        self.root.found_types[self.type].add(self.expr)

    def build(self, var: Variable, *args: Type) -> str:
        self.root.add_closure('_iso8601_parse_', '__iso8601_parse__', self._maybe_coerce)
        return f'_iso8601_parse_({var})'

    def _maybe_coerce(self, value: Optional[Any] = None) -> Optional[datetime]:
        if value is not None:
            if isinstance(value, str):
                return self.root.date_parser(value)
            return value
        return None

class NamedTupleNode(Node):
    type: NodeType = NodeType.NAMEDTUPLE
    compatible_types: Tuple[Type, ...] = TUPLE_TYPES

    @classmethod
    def create_if_compatible(cls, typ: Type, *, root: 'RootNode') -> Optional['Node']:
        if cls._issubclass(typ, cls.compatible_types):
            if '_asdict' in typ.__dict__ and '_make' in typ.__dict__ and ('_fields' in typ.__dict__):
                return cls(typ, root=root)
        return None

    def build(self, var: Variable, *args: Type) -> str:
        self.root.add_closure(self.local_name, self.global_name, self.expr)
        tup = self.expr
        fields = ', '.join(('{0}={1}'.format(field, self.root.build(var[i], typ)) for i, (field, typ) in enumerate(tup.__annotations__.items()))
        return f'{self.local_name}({fields})'

    def next_namedtuple_name(self, typ: Type) -> str:
        num: int = next(TUPLE_NAME_COUNTER)
        return f'namedtuple_{num}_{typ.__name__}'

    @cached_property
    def local_name(self) -> str:
        return self.next_namedtuple_name(self.expr)

    @cached_property
    def global_name(self) -> str:
        return '_' + self.local_name + '_'

class TupleNode(Node):
    type: NodeType = NodeType.TUPLE
    compatible_types: Tuple[Type, ...] = TUPLE_TYPES
    use_origin: bool = True

    def build(self, var: Variable, *args: Type) -> str:
        if not args:
            return self._build_untyped_tuple(var)
        for position, arg in enumerate(args):
            if arg is Ellipsis:
                assert position == 1
                return self._build_vararg_tuple(var, args[0])
        return self._build_tuple_literal(var, *args)

    def _build_tuple_literal(self, var: Variable, *member_args: Type) -> str:
        source = '(' + ', '.join((self.root.build(var[i], arg) for i, arg in enumerate(member_args))) + ')'
        if ',' not in source:
            return source[:-1] + ',)'
        return source

    def _build_untyped_tuple(self, var: Variable) -> str:
        return f'tuple({var})'

    def _build_vararg_tuple(self, var: Variable, member_type: Type) -> str:
        item_var = var.next_identifier()
        handler = self.root.build(item_var, member_type)
        return f'tuple({handler} for {item_var} in {var})'

class SetNode(Node):
    type: NodeType = NodeType.SET
    compatible_types: Tuple[Type, ...] = SET_TYPES
    use_origin: bool = True

    def build(self, var: Variable, *args: Type) -> str:
        if not args:
            return f'set({var})'
        return self._build_set_expression(var, *args)

    def _build_set_expression(self, var: Variable, member_type: Type) -> str:
        member_var = var.next_identifier()
        handler = self.root.build(member_var, member_type)
        return f'{{{handler} for {member_var} in {var}}}'

class DictNode(Node):
    type: NodeType = NodeType.DICT
    compatible_types: Tuple[Type, ...] = DICT_TYPES
    use_origin: bool = True

    def build(self, var: Variable, *args: Type) -> str:
        if not args:
            return f'dict({var})'
        return self._build_dict_expression(var, *args)

    def _build_dict_expression(self, var: Variable, key_type: Type, value_type: Type) -> str:
        key_var = var.next_identifier()
        value_var = key_var.next_identifier()
        key_handler = self.root.build(key_var, key_type)
        value_handler = self.root.build(value_var, value_type)
        return f'{{{key_handler}: {value_handler} for {key_var}, {value_var} in {var}.items()}}'

class ListNode(Node):
    type: NodeType = NodeType.LIST
    compatible_types: Tuple[Type, ...] = LIST_TYPES
    use_origin: bool = True

    def build(self, var: Variable, *args: Type) -> str:
        if not args:
            return f'list({var})'
        return self._build_list_expression(var, *args)

    def _build_list_expression(self, var: Variable, item_type: Type) -> str:
        item_var = var.next_identifier()
        handler = self.root.build(item_var, item_type)
        return f'[{handler} for {item_var} in {var}]'

class ModelNode(Node):
    type: NodeType = NodeType.MODEL
    compatible_types: Tuple[Type, ...] = ()

    def __post_init__(self) -> None:
        self.root.found_types[self.type].add(self.expr)

    @classmethod
    def create_if_compatible(cls, typ: Type, *, root: 'RootNode') -> Optional['Node']:
        if cls._is_model(typ):
            return cls(typ, root=root)
        return None

    @classmethod
    def _is_model(cls, typ: Type) -> bool:
        try:
            if issubclass(typ, ModelT):
                return True
        except TypeError:
            pass
        return False

    def build(self, var: Variable, *args: Type) -> str:
        model_name = self._ensure_model_name(self.expr)
        return f'{model_name}._from_data_field({var})'

    def _ensure_model_name(self, typ: Type) -> str:
        try:
            namespace = typ._options.namespace
        except AttributeError:
            model_name = '_Model_'
            model_global_name = '__Model__'
            self.root.add_closure(model_name, model_global_name, self.Model)
        else:
            model_name = qualname_to_identifier(qualname(namespace))
            model_global_name = '__' + model_name + '__'
            self.root.add_closure(model_name, model_global_name, self.expr)
        return model_name

    @cached_property
    def Model(self) -> Type:
        from .base import Model
        return Model

class UserNode(Node):
    type: NodeType = NodeType.USER
    compatible_types: Tuple[Type, ...] = ()

    def __init__(self, expr: Type, root: 'RootNode', *, user_types: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None, handler: Callable):
        super().__init__(expr, root)
        self.handler: Callable = handler
        self.handler_name: str = qualname_to_identifier(qualname(self.handler))
        self.handler_global_name: str = '__' + self.handler_name + '__'

    def __post_init__(self) -> None:
        self.root.found_types[self.type].add(self.expr)

    def _maybe_coerce(self, value: Optional[Any]) -> Optional[Any]:
        if value is None:
            return None
        if isinstance(value, JSON_TYPES):
            return self.handler(value)
        return value

    def build(self, var: Variable, *args: Type) -> str:
        self.root.add_closure(self.handler_name, self.handler_global_name, self._maybe_coerce)
        return f'{self.handler_name}({var})'

class RootNode(Node):
    DEFAULT_NODE: Optional[Type['Node']] = None
    type: NodeType = NodeType.ROOT

    @classmethod
    def _register(cls) -> None:
        ...

    def add_closure(self, local_name: str, global_name: str, obj: Any) -> None:
        self.globals[global_name] = obj
        self.closures[local_name] = global_name

    def __init__(self, expr: Type, root: Optional['RootNode'] = None, *, user_types: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None, date_parser: Optional[Callable] = None) -> None:
        assert self.type == NodeType.ROOT
        self.type_stats: Counter = Counter()
        self.user_types: Dict[Union[Type, Tuple[Type, ...]], Callable] = user_types or {}
        self.globals: Dict[str, Any] = {}
        self.closures: Dict[str, str] = {}
        if date_parser is not None:
            self.date_parser: Callable = date_parser
        else:
            self.date_parser: Callable = parse_iso8601
        self.found_types: Dict[NodeType, Set[Type]] = defaultdict(set)
        super().__init__(expr, root=self)

    def find_compatible_node_or_default(self, info: TypeInfo) -> 'Node':
        node = self.find_compatible_node(info)
        if node is None:
            return self.new_default_node(info.type)
        else:
            return node