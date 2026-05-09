import sys
from configparser import ConfigParser
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from mypy.errorcodes import ErrorCode
from mypy.nodes import ARG_NAMED, ARG_NAMED_OPT, ARG_OPT, ARG_POS, ARG_STAR2, MDEF, Argument, AssignmentStmt, Block, CallExpr, ClassDef, Context, Decorator, EllipsisExpr, FuncBase, FuncDef, JsonDict, MemberExpr, NameExpr, PassStmt, PlaceholderNode, RefExpr, StrExpr, SymbolNode, SymbolTableNode, TempNode, TypeInfo, TypeVarExpr, Var
from mypy.options import Options
from mypy.plugin import CheckerPluginInterface, ClassDefContext, FunctionContext, MethodContext, Plugin, ReportConfigContext, SemanticAnalyzerPluginInterface
from mypy.plugins import dataclasses
from mypy.semanal import set_callable_name
from mypy.server.trigger import make_wildcard_trigger
from mypy.types import AnyType, CallableType, Instance, NoneType, Overloaded, ProperType, Type, TypeOfAny, TypeType, TypeVarId, TypeVarType, UnionType, get_proper_type
from mypy.typevars import fill_typevars
from mypy.version import __version__ as mypy_version
from pydantic.v1.utils import is_valid_field
from dataclasses import dataclass
from typing import TypeVar

class PydanticPlugin(Plugin):
    def __init__(self, options: Options) -> None:
        ...

    def get_base_class_hook(self, fullname: str) -> Optional[Callable[[Context], None]]:
        ...

    def get_metaclass_hook(self, fullname: str) -> Optional[Callable[[Context], None]]:
        ...

    def get_function_hook(self, fullname: str) -> Optional[Callable[[Context], None]]:
        ...

    def get_method_hook(self, fullname: str) -> Optional[Callable[[Context], None]]:
        ...

    def get_class_decorator_hook(self, fullname: str) -> Optional[Callable[[Context], None]]:
        ...

    def report_config_data(self, ctx: Context) -> Dict[str, Any]:
        ...

class PydanticPluginConfig:
    __slots__ = ('init_forbid_extra', 'init_typed', 'warn_required_dynamic_aliases', 'warn_untyped_fields', 'debug_dataclass_transform')

    def __init__(self, options: Options) -> None:
        ...

    def to_data(self) -> Dict[str, Any]:
        ...

class PydanticModelTransformer:
    tracked_config_fields: Set[str] = {'extra', 'allow_mutation', 'frozen', 'orm_mode', 'allow_population_by_field_name', 'alias_generator'}

    def __init__(self, ctx: Context, plugin_config: PydanticPluginConfig) -> None:
        ...

    def transform(self) -> None:
        ...

    def collect_config(self) -> ModelConfigData:
        ...

    def collect_fields(self, model_config: ModelConfigData) -> List[PydanticModelField]:
        ...

    def add_initializer(self, fields: List[PydanticModelField], config: ModelConfigData, is_settings: bool) -> None:
        ...

    def add_construct_method(self, fields: List[PydanticModelField]) -> None:
        ...

    def set_frozen(self, fields: List[PydanticModelField], frozen: bool) -> None:
        ...

    def get_config_update(self, substmt: AssignmentStmt) -> Optional[ModelConfigData]:
        ...

    def get_is_required(self, cls: TypeInfo, stmt: AssignmentStmt, lhs: NameExpr) -> bool:
        ...

    def get_alias_info(self, stmt: AssignmentStmt) -> Tuple[Optional[str], bool]:
        ...

    def get_field_arguments(self, fields: List[PydanticModelField], typed: bool, force_all_optional: bool, use_alias: bool) -> List[Argument]:
        ...

    def should_init_forbid_extra(self, fields: List[PydanticModelField], config: ModelConfigData) -> bool:
        ...

class PydanticModelField:
    def __init__(self, name: str, is_required: bool, alias: Optional[str], has_dynamic_alias: bool, line: int, column: int) -> None:
        ...

    def to_var(self, info: TypeInfo, use_alias: bool) -> Var:
        ...

    def to_argument(self, info: TypeInfo, typed: bool, force_optional: bool, use_alias: bool) -> Argument:
        ...

    def serialize(self) -> Dict[str, Any]:
        ...

    @classmethod
    def deserialize(cls, info: TypeInfo, data: Dict[str, Any]) -> 'PydanticModelField':
        ...

class ModelConfigData:
    def __init__(self, forbid_extra: Optional[bool] = None, allow_mutation: Optional[bool] = None, frozen: Optional[bool] = None, orm_mode: Optional[bool] = None, allow_population_by_field_name: Optional[bool] = None, has_alias_generator: Optional[bool] = None) -> None:
        ...

    def set_values_dict(self) -> Dict[str, Any]:
        ...

    def update(self, config: ModelConfigData) -> None:
        ...

    def setdefault(self, key: str, value: Any) -> None:
        ...

def error_from_orm(model_name: str, api: 'mypy.api', context: 'mypy.context') -> None:
    ...

def error_invalid_config_value(name: str, api: 'mypy.api', context: 'mypy.context') -> None:
    ...

def error_required_dynamic_aliases(api: 'mypy.api', context: 'mypy.context') -> None:
    ...

def error_unexpected_behavior(detail: str, api: 'mypy.api', context: 'mypy.context') -> None:
    ...

def error_untyped_fields(api: 'mypy.api', context: 'mypy.context') -> None:
    ...

def error_default_and_default_factory_specified(api: 'mypy.api', context: 'mypy.context') -> None:
    ...

def add_method(ctx: Context, name: str, args: List[Argument], return_type: Type, self_type: Optional[Type] = None, tvar_def: Optional[TypeVarType] = None, is_classmethod: bool = False, is_new: bool = False) -> None:
    ...

def get_fullname(x: Any) -> str:
    ...

def get_name(x: Any) -> str:
    ...
