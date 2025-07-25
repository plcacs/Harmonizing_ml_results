import sys
from configparser import ConfigParser
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type as TypingType, Union
from mypy.errorcodes import ErrorCode
from mypy.nodes import ARG_NAMED, ARG_NAMED_OPT, ARG_OPT, ARG_POS, ARG_STAR2, MDEF, Argument, AssignmentStmt, Block, CallExpr, ClassDef, Context, Decorator, EllipsisExpr, FuncBase, FuncDef, JsonDict, MemberExpr, NameExpr, PassStmt, PlaceholderNode, RefExpr, StrExpr, SymbolNode, SymbolTableNode, TempNode, TypeInfo, TypeVarExpr, Var
from mypy.options import Options
from mypy.plugin import CheckerPluginInterface, ClassDefContext, FunctionContext, MethodContext, Plugin, ReportConfigContext, SemanticAnalyzerPluginInterface
from mypy.plugins import dataclasses
from mypy.semanal import set_callable_name
from mypy.server.trigger import make_wildcard_trigger
from mypy.types import AnyType, CallableType, Instance, NoneType, Overloaded, ProperType, Type, TypeOfAny, TypeType, TypeVarId, TypeVarType, UnionType, get_proper_type
from mypy.typevars import fill_typevars
from mypy.util import get_unique_redefinition_name
from mypy.version import __version__ as mypy_version
from pydantic.v1.utils import is_valid_field

try:
    from mypy.types import TypeVarDef
except ImportError:
    from mypy.types import TypeVarType as TypeVarDef

CONFIGFILE_KEY = 'pydantic-mypy'
METADATA_KEY = 'pydantic-mypy-metadata'
_NAMESPACE = __name__[:-5]
BASEMODEL_FULLNAME = f'{_NAMESPACE}.main.BaseModel'
BASESETTINGS_FULLNAME = f'{_NAMESPACE}.env_settings.BaseSettings'
MODEL_METACLASS_FULLNAME = f'{_NAMESPACE}.main.ModelMetaclass'
FIELD_FULLNAME = f'{_NAMESPACE}.fields.Field'
DATACLASS_FULLNAME = f'{_NAMESPACE}.dataclasses.dataclass'

def parse_mypy_version(version: str) -> Tuple[int, ...]:
    return tuple(map(int, version.partition('+')[0].split('.')))

MYPY_VERSION_TUPLE = parse_mypy_version(mypy_version)
BUILTINS_NAME = 'builtins' if MYPY_VERSION_TUPLE >= (0, 930) else '__builtins__'
__version__ = 2

def plugin(version: str) -> TypingType[Plugin]:
    return PydanticPlugin

class PydanticPlugin(Plugin):

    def __init__(self, options: Options) -> None:
        self.plugin_config = PydanticPluginConfig(options)
        self._plugin_data = self.plugin_config.to_data()
        super().__init__(options)

    def get_base_class_hook(self, fullname: str) -> Optional[Callable[[ClassDefContext], None]]:
        sym = self.lookup_fully_qualified(fullname)
        if sym and isinstance(sym.node, TypeInfo):
            if any((get_fullname(base) == BASEMODEL_FULLNAME for base in sym.node.mro)):
                return self._pydantic_model_class_maker_callback
        return None

    def get_metaclass_hook(self, fullname: str) -> Optional[Callable[[ClassDefContext], None]]:
        if fullname == MODEL_METACLASS_FULLNAME:
            return self._pydantic_model_metaclass_marker_callback
        return None

    def get_function_hook(self, fullname: str) -> Optional[Callable[[FunctionContext], Type]]:
        sym = self.lookup_fully_qualified(fullname)
        if sym and sym.fullname == FIELD_FULLNAME:
            return self._pydantic_field_callback
        return None

    def get_method_hook(self, fullname: str) -> Optional[Callable[[MethodContext], Type]]:
        if fullname.endswith('.from_orm'):
            return from_orm_callback
        return None

    def get_class_decorator_hook(self, fullname: str) -> Optional[Callable[[ClassDefContext], None]]:
        if fullname == DATACLASS_FULLNAME and MYPY_VERSION_TUPLE < (1, 1):
            return dataclasses.dataclass_class_maker_callback
        return None

    def report_config_data(self, ctx: ReportConfigContext) -> Dict[str, Any]:
        return self._plugin_data

    def _pydantic_model_class_maker_callback(self, ctx: ClassDefContext) -> None:
        transformer = PydanticModelTransformer(ctx, self.plugin_config)
        transformer.transform()

    def _pydantic_model_metaclass_marker_callback(self, ctx: ClassDefContext) -> None:
        if self.plugin_config.debug_dataclass_transform:
            return
        info_metaclass = ctx.cls.info.declared_metaclass
        assert info_metaclass, "callback not passed from 'get_metaclass_hook'"
        if getattr(info_metaclass.type, 'dataclass_transform_spec', None):
            info_metaclass.type.dataclass_transform_spec = None

    def _pydantic_field_callback(self, ctx: FunctionContext) -> Type:
        default_any_type = ctx.default_return_type
        assert ctx.callee_arg_names[0] == 'default', '"default" is no longer first argument in Field()'
        assert ctx.callee_arg_names[1] == 'default_factory', '"default_factory" is no longer second argument in Field()'
        default_args = ctx.args[0]
        default_factory_args = ctx.args[1]
        if default_args and default_factory_args:
            error_default_and_default_factory_specified(ctx.api, ctx.context)
            return default_any_type
        if default_args:
            default_type = ctx.arg_types[0][0]
            default_arg = default_args[0]
            if not isinstance(default_arg, EllipsisExpr):
                return default_type
        elif default_factory_args:
            default_factory_type = ctx.arg_types[1][0]
            if isinstance(default_factory_type, Overloaded):
                if MYPY_VERSION_TUPLE > (0, 910):
                    default_factory_type = default_factory_type.items[0]
                else:
                    default_factory_type = default_factory_type.items()[0]
            if isinstance(default_factory_type, CallableType):
                ret_type = default_factory_type.ret_type
                args = getattr(ret_type, 'args', None)
                if args:
                    if all((isinstance(arg, TypeVarType) for arg in args)):
                        ret_type.args = tuple((default_any_type for _ in args))
                return ret_type
        return default_any_type

class PydanticPluginConfig:
    __slots__ = ('init_forbid_extra', 'init_typed', 'warn_required_dynamic_aliases', 'warn_untyped_fields', 'debug_dataclass_transform')

    def __init__(self, options: Options) -> None:
        if options.config_file is None:
            return
        toml_config = parse_toml(options.config_file)
        if toml_config is not None:
            config = toml_config.get('tool', {}).get('pydantic-mypy', {})
            for key in self.__slots__:
                setting = config.get(key, False)
                if not isinstance(setting, bool):
                    raise ValueError(f'Configuration value must be a boolean for key: {key}')
                setattr(self, key, setting)
        else:
            plugin_config = ConfigParser()
            plugin_config.read(options.config_file)
            for key in self.__slots__:
                setting = plugin_config.getboolean(CONFIGFILE_KEY, key, fallback=False)
                setattr(self, key, setting)

    def to_data(self) -> Dict[str, Any]:
        return {key: getattr(self, key) for key in self.__slots__}

def from_orm_callback(ctx: MethodContext) -> Type:
    ctx_type = ctx.type
    if isinstance(ctx_type, TypeType):
        ctx_type = ctx_type.item
    if isinstance(ctx_type, CallableType) and isinstance(ctx_type.ret_type, Instance):
        model_type = ctx_type.ret_type
    elif isinstance(ctx_type, Instance):
        model_type = ctx_type
    else:
        detail = f'ctx.type: {ctx_type} (of type {ctx_type.__class__.__name__})'
        error_unexpected_behavior(detail, ctx.api, ctx.context)
        return ctx.default_return_type
    pydantic_metadata = model_type.type.metadata.get(METADATA_KEY)
    if pydantic_metadata is None:
        return ctx.default_return_type
    orm_mode = pydantic_metadata.get('config', {}).get('orm_mode')
    if orm_mode is not True:
        error_from_orm(get_name(model_type.type), ctx.api, ctx.context)
    return ctx.default_return_type

class PydanticModelTransformer:
    tracked_config_fields = {'extra', 'allow_mutation', 'frozen', 'orm_mode', 'allow_population_by_field_name', 'alias_generator'}

    def __init__(self, ctx: ClassDefContext, plugin_config: PydanticPluginConfig) -> None:
        self._ctx = ctx
        self.plugin_config = plugin_config

    def transform(self) -> None:
        ctx = self._ctx
        info = ctx.cls.info
        self.adjust_validator_signatures()
        config = self.collect_config()
        fields = self.collect_fields(config)
        is_settings = any((get_fullname(base) == BASESETTINGS_FULLNAME for base in info.mro[:-1]))
        self.add_initializer(fields, config, is_settings)
        self.add_construct_method(fields)
        self.set_frozen(fields, frozen=config.allow_mutation is False or config.frozen is True)
        info.metadata[METADATA_KEY] = {'fields': {field.name: field.serialize() for field in fields}, 'config': config.set_values_dict()}

    def adjust_validator_signatures(self) -> None:
        for name, sym in self._ctx.cls.info.names.items():
            if isinstance(sym.node, Decorator):
                first_dec = sym.node.original_decorators[0]
                if isinstance(first_dec, CallExpr) and isinstance(first_dec.callee, NameExpr) and (first_dec.callee.fullname == f'{_NAMESPACE}.class_validators.validator'):
                    sym.node.func.is_class = True

    def collect_config(self) -> 'ModelConfigData':
        ctx = self._ctx
        cls = ctx.cls
        config = ModelConfigData()
        for stmt in cls.defs.body:
            if not isinstance(stmt, ClassDef):
                continue
            if stmt.name == 'Config':
                for substmt in stmt.defs.body:
                    if not isinstance(substmt, AssignmentStmt):
                        continue
                    config.update(self.get_config_update(substmt))
                if config.has_alias_generator and (not config.allow_population_by_field_name) and self.plugin_config.warn_required_dynamic_aliases:
                    error_required_dynamic_aliases(ctx.api, stmt)
        for info in cls.info.mro[1:]:
            if METADATA_KEY not in info.metadata:
                continue
            ctx.api.add_plugin_dependency(make_wildcard_trigger(get_fullname(info)))
            for name, value in info.metadata[METADATA_KEY]['config'].items():
                config.setdefault(name, value)
        return config

    def collect_fields(self, model_config: 'ModelConfigData') -> List['PydanticModelField']:
        ctx = self._ctx
        cls = self._ctx.cls
        fields = []
        known_fields = set()
        for stmt in cls.defs.body:
            if not isinstance(stmt, AssignmentStmt):
                continue
            lhs = stmt.lvalues[0]
            if not isinstance(lhs, NameExpr) or not is_valid_field(lhs.name):
                continue
            if not stmt.new_syntax and self.plugin_config.warn_untyped_fields:
                error_untyped_fields(ctx.api, stmt)
            sym = cls.info.names.get(lhs.name)
            if sym is None:
                continue
            node = sym.node
            if isinstance(node, PlaceholderNode):
                continue
            if not isinstance(node, Var):
                continue
            if node.is_classvar:
                continue
            is_required = self.get_is_required(cls, stmt, lhs)
            alias, has_dynamic_alias = self.get_alias_info(stmt)
            if has_dynamic_alias and (not model_config.allow_population_by_field_name) and self.plugin_config.warn_required_dynamic_aliases:
                error_required_dynamic_aliases(ctx.api, stmt)
            fields.append(PydanticModelField(name=lhs.name, is_required=is_required, alias=alias, has_dynamic_alias=has_dynamic_alias, line=stmt.line, column=stmt.column))
            known_fields.add(lhs.name)
        all_fields = fields.copy()
        for info in cls.info.mro[1:]:
            if METADATA_KEY not in info.metadata:
                continue
            superclass_fields = []
            ctx.api.add_plugin_dependency(make_wildcard_trigger(get_fullname(info)))
            for name, data in info.metadata[METADATA_KEY]['fields'].items():
                if name not in known_fields:
                    field = PydanticModelField.deserialize(info, data)
                    known_fields.add(name)
                    superclass_fields.append(field)
                else:
                    field, = (a for a in all_fields if a.name == name)
                    all_fields.remove(field)
                    superclass_fields.append(field)
            all_fields = superclass_fields + all_fields
        return all_fields

    def add_initializer(self, fields: List['PydanticModelField'], config: 'ModelConfigData', is_settings: bool) -> None:
        ctx = self._ctx
        typed = self.plugin_config.init_typed
        use_alias = config.allow_population_by_field_name is not True
        force_all_optional = is_settings or bool(config.has_alias_generator and (not config.allow_population_by_field_name))
        init_arguments = self.get_field_arguments(fields, typed=typed, force_all_optional=force_all_optional, use_alias=use_alias)
        if not self.should_init_forbid_extra(fields, config):
            var = Var('kwargs')
            init_arguments.append(Argument(var, AnyType(TypeOfAny.explicit), None, ARG_STAR2))
        if '__init__' not in ctx.cls.info.names:
            add_method(ctx, '__init__', init_arguments, NoneType())

    def add_construct_method(self, fields: List['PydanticModelField']) -> None:
        ctx = self._ctx
        set_str = ctx.api.named_type(f'{BUILTINS_NAME}.set', [ctx.api.named_type(f'{BUILTINS_NAME}.str')])
        optional_set_str = UnionType([set_str, NoneType()])
        fields_set_argument = Argument(Var('_fields_set', optional_set_str), optional_set_str, None, ARG_OPT)
        construct_arguments = self.get_field_arguments(fields, typed=True, force_all_optional=False, use_alias=False)
        construct_arguments = [fields_set_argument] + construct_arguments
        obj_type = ctx.api.named_type(f'{BUILTINS_NAME}.object')
        self_tvar_name = '_PydanticBaseModel'
        tvar_fullname = ctx.cls.fullname + '.' + self_tvar_name
        if MYPY_VERSION_TUPLE >= (1, 4):
            tvd = TypeVarType(self_tvar_name, tvar_fullname, TypeVarId(-1, namespace=ctx.cls.fullname + '.construct') if MYPY_VERSION_TUPLE >= (1, 11) else TypeVarId(-1), [], obj_type, AnyType(TypeOfAny.from_omitted_generics))
            self_tvar_expr = TypeVarExpr(self_tvar_name, tvar_fullname, [], obj_type, AnyType(TypeOfAny.from_omitted_generics))
        else:
            tvd = TypeVarDef(self_tvar_name, tvar_fullname, -1, [], obj_type)
            self_tvar_expr = TypeVarExpr(self_tvar_name, tvar_fullname, [], obj_type)
        ctx.cls.info.names[self_tvar_name] = SymbolTableNode(MDEF, self_tvar_expr)
        if isinstance(tvd, TypeVarType):
            self_type = tvd
        else:
            self_type = TypeVarType(tvd)
        add_method(ctx, 'construct', construct_arguments, return_type=self_type, self_type=self_type, tvar_def=tvd, is_classmethod=True)

    def set_frozen(self, fields: List['PydanticModelField'], frozen: bool) -> None:
        ctx = self._ctx
        info = ctx.cls.info
        for field in fields:
            sym_node = info.names.get(field.name)
            if sym_node is not None:
                var = sym_node.node
                if isinstance(var, Var):
                    var.is_property = frozen
                elif isinstance(var, PlaceholderNode) and (not ctx.api.final_iteration):
                    ctx.api.defer()
                else:
                    try:
                        var_str = str(var)
                    except TypeError:
                        var_str = repr(var)
                    detail = f'sym_node.node: {var_str} (of type {var.__class__})'
                    error_unexpected_behavior(detail, ctx.api, ctx.cls)
            else:
                var = field.to_var(info, use_alias=False)
                var.info = info
                var.is_property = frozen
                var._fullname = get_fullname(info) + '.' + get_name(var)
                info.names[get_name(var)] = SymbolTableNode(MDEF, var)

    def get_config_update(self, substmt: AssignmentStmt) -> Optional['ModelConfigData']:
        lhs = substmt.lvalues[0]
        if not (isinstance(lhs, NameExpr) and lhs.name in self.tracked_config_fields):
            return None
        if lhs.name == 'extra':
            if isinstance(substmt.rvalue, StrExpr):
                forbid_extra = substmt.rvalue.value == 'forbid'
            elif isinstance(substmt.rvalue, MemberExpr):
                forbid_extra = substmt.rvalue.name == 'forbid'
            else:
                error_invalid_config_value(lhs.name, self._ctx.api, substmt)
                return None
            return ModelConfigData(forbid_extra=forbid_extra)
        if lhs.name == 'alias_generator':
            has_alias_generator = True
            if isinstance(substmt.rvalue, NameExpr) and substmt.rvalue.fullname == 'builtins.None':
                has_alias_generator = False
            return ModelConfigData(has_alias_generator=has_alias_generator)
        if isinstance(substmt.rvalue, NameExpr) and substmt.rvalue.fullname in ('builtins.True', 'builtins.False'):
            return ModelConfigData(**{lhs.name: substmt.rvalue.fullname == 'builtins.True'})
        error_invalid_config_value(lhs.name, self._ctx.api, substmt)
        return None

    @staticmethod
    def get_is_required(cls: ClassDef, stmt: AssignmentStmt, lhs: NameExpr) -> bool:
        expr = stmt.rvalue
        if isinstance(expr, TempNode):
            value_type = get_proper_type(cls.info[lhs.name].type)
            return not PydanticModelTransformer.type_has_implicit_default(value_type)
        if isinstance(expr, CallExpr) and isinstance(expr.callee, RefExpr) and (expr.callee.fullname == FIELD_FULLNAME):
            for arg, name in zip(expr.args, expr.arg_names):
                if name is None or name == 'default':
                    return arg.__class__ is EllipsisExpr
                if name == 'default_factory':
                    return False
            value_type = get_proper_type(cls.info[lhs.name].type)
            return not PydanticModelTransformer.type_has_implicit_default(value_type)
        return isinstance(expr, EllipsisExpr)

    @staticmethod
    def type_has_implicit_default(type_: Type) -> bool:
        if isinstance(type_, AnyType):
            return True
        if isinstance(type_, UnionType) and any((isinstance(item, NoneType) or isinstance(item, AnyType) for item in type_.items)):
            return True
        return False

    @staticmethod
    def get_alias_info(stmt: AssignmentStmt) -> Tuple[Optional[str], bool]:
        expr = stmt.rvalue
        if isinstance(expr, TempNode):
            return (None, False)
        if not (isinstance(expr, CallExpr) and isinstance(expr.callee, RefExpr) and (expr.callee.fullname == FIELD_FULLNAME)):
            return (None, False)
        for i, arg_name in enumerate(expr.arg_names):
            if arg_name != 'alias':
                continue
            arg = expr.args[i]
            if isinstance(arg, StrExpr):
                return (arg.value, False)
            else:
                return (None, True)
        return (None, False)

    def get_field_arguments(self, fields: List['PydanticModelField'], typed: bool, force_all_optional: bool, use_alias: bool) -> List[Argument]:
        info = self._ctx.cls.info
        arguments = [field.to_argument(info, typed=typed, force_optional=force_all_optional, use_alias=use_alias) for field in fields if not (use_alias and field.has_dynamic_alias)]
        return arguments

    def should_init_forbid_extra(self, fields: List['PydanticModelField'], config: 'ModelConfigData') -> bool:
        if not config.allow_population_by_field_name:
            if self.is_dynamic_alias_present(fields, bool(config.has_alias_generator)):
                return False
        if config.forbid_extra:
            return True
        return self.plugin_config.init_forbid_extra

    @staticmethod
    def is_dynamic_alias_present(fields: List['PydanticModelField'], has_alias_generator: bool) -> bool:
        for field in fields:
            if field.has_dynamic_alias:
                return True
        if has_alias_generator:
            for field in fields:
                if field.alias is None:
                    return True
        return False

class PydanticModelField:

    def __init__(self, name: str, is_required: bool, alias: Optional[str], has_dynamic_alias: bool, line: int, column: int) -> None:
        self.name = name
        self.is_required = is_required
        self.alias = alias
        self.has_dynamic_alias = has_dynamic_alias
        self.line = line
        self.column = column

    def to_var(self, info: TypeInfo, use_alias: bool) -> Var:
        name = self.name
        if use_alias and self.alias is not None:
            name = self.alias
        return Var(name, info[self.name].type)

    def to_argument(self, info: TypeInfo, typed: bool, force_optional: bool, use_alias: bool) -> Argument:
        if typed and info[self.name].type is not None:
            type_annotation = info[self.name].type
        else:
            type_annotation = AnyType(TypeOfAny.explicit)
        return Argument(variable=self.to_var(info, use_alias), type_annotation=type_annotation, initializer=None, kind=ARG_NAMED_OPT if force_optional or not self.is_required else ARG_NAMED)

    def serialize(self) -> Dict[str, Any]:
        return self.__dict__

    @classmethod
    def deserialize(cls, info: TypeInfo, data: Dict[str, Any]) -> 'PydanticModelField':
        return cls(**data)

class ModelConfigData:

    def __init__(self, forbid_extra: Optional[bool] = None, allow_mutation: Optional[bool] = None, frozen: Optional[bool] = None, orm_mode: Optional[bool] = None, allow_population_by_field_name: Optional[bool] = None, has_alias_generator: Optional[bool] = None) -> None:
        self.forbid_extra = forbid_extra
        self.allow_mutation = allow_mutation
        self.frozen = frozen
        self.orm_mode = orm_mode
        self.allow_population_by_field_name = allow_population_by_field_name
        self.has_alias_generator = has_alias_generator

    def set_values_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    def update(self, config: Optional['ModelConfigData']) -> None:
        if config is None:
            return
        for k, v in config.set_values_dict().items():
            setattr(self, k, v)

    def setdefault(self, key: str, value: Any) -> None:
        if getattr(self, key) is None:
            setattr(self, key, value)

ERROR_ORM = ErrorCode('pydantic-orm', 'Invalid from_orm call', 'Pydantic')
ERROR_CONFIG = ErrorCode('pydantic-config', 'Invalid config value', 'Pydantic')
ERROR_ALIAS = ErrorCode('pydantic-alias', 'Dynamic alias disallowed', 'Pydantic')
ERROR_UNEXPECTED = ErrorCode('pydantic-unexpected', 'Unexpected behavior', 'Pydantic')
ERROR_UNTYPED = ErrorCode('pydantic-field', 'Untyped field disallowed', 'Pydantic')
ERROR_FIELD_DEFAULTS = ErrorCode('pydantic-field', 'Invalid Field defaults', 'Pydantic')

def error_from_orm(model_name: str, api: CheckerPluginInterface, context: Context) -> None:
    api.fail(f'"{model_name}" does not have orm_mode=True', context, code=ERROR_ORM)

def error_invalid_config_value(name: str, api: CheckerPluginInterface, context: Context) -> None:
    api.fail(f'Invalid value for "Config.{name}"', context, code=ERROR_CONFIG)

def error_required_dynamic_aliases(api: CheckerPluginInterface, context: Context) -> None:
    api.fail('Required dynamic aliases disallowed', context, code=ERROR_ALIAS)

def error_unexpected_behavior(detail: str, api: CheckerPluginInterface, context: Context) -> None:
    link = 'https://github.com/pydantic/pydantic/issues/new/choose'
    full_message = f'The pydantic mypy plugin ran into unexpected behavior: {detail}\n'
    full_message += f'Please consider reporting this bug at {link} so we can try to fix it!'
    api.fail(full_message, context, code=ERROR_UNEXPECTED)

def error_untyped_fields(api: CheckerPluginInterface, context: Context) -> None:
    api.fail('Untyped fields disallowed', context, code=ERROR_UNTYPED)

def error_default_and_default_factory_specified(api: CheckerPluginInterface, context: Context) -> None:
    api.fail('Field default and default_factory cannot be specified together', context, code=ERROR_FIELD_DEFAULTS)

def add_method(ctx: ClassDefContext, name: str, args: List[Argument], return_type: Type, self_type: Optional[Type] = None, tvar_def: Optional[TypeVarType] = None, is_classmethod: bool = False, is_new: bool = False) -> None:
    info = ctx.cls.info
    if name in info.names:
        sym = info.names[name]
        if sym.plugin_generated and isinstance(sym.node, FuncDef):
            ctx.cls.defs.body.remove(sym.node)
    self_type = self_type or fill_typevars(info)
    if is_classmethod or is_new:
        first = [Argument(Var('_cls'), TypeType.make_normalized(self_type), None, ARG_POS)]
    else:
        self_type = self_type or fill_typevars(info)
        first = [Argument(Var('__pydantic_self__'), self_type, None, ARG_POS)]
    args = first + args
    arg_types, arg_names, arg_kinds = ([], [], [])
    for arg in args:
        assert arg.type_annotation, 'All arguments must be fully typed.'
        arg_types.append(arg.type_annotation)
        arg_names.append(get_name(arg.variable))
        arg_kinds.append(arg.kind)
    function_type = ctx.api.named_type(f'{BUILTINS_NAME}.function')
    signature = CallableType(arg_types, arg_kinds, arg_names, return_type, function_type)
    if tvar_def:
        signature.variables = [tvar_def]
    func = FuncDef(name, args, Block([PassStmt()]))
    func.info = info
    func.type = set_callable_name(signature, func)
    func.is_class = is_classmethod
    func._fullname = get_fullname(info) + '.' + name
    func.line = info.line
    if name in info.names:
        r_name = get_unique_redefinition_name(name, info.names)
        info.names[r_name] = info.names[name]
    if is_classmethod:
        func.is_decorated = True
        v = Var(name, func.type)
        v.info = info
        v._fullname = func._fullname
        v.is_classmethod = True
        dec = Decorator(func, [NameExpr('classmethod')], v)
        dec.line = info.line
        sym = SymbolTableNode(MDEF, dec)
    else:
        sym = SymbolTableNode(MDEF, func)
    sym.plugin_generated = True
    info.names[name] = sym
    info.defn.defs.body.append(func)

def get_fullname(x: Any) -> str:
    fn = x.fullname
    if callable(fn):
        return fn()
    return fn

def get_name(x: Any) -> str:
    fn = x.name
    if callable(fn):
        return fn()
    return fn

def parse_toml(config_file: str) -> Optional[Dict[str, Any]]:
    if not config_file.endswith('.toml'):
        return None
    read_mode = 'rb'
    if sys.version_info >= (3, 11):
        import tomllib as toml_
    else:
        try:
            import tomli as toml_
        except ImportError:
            read_mode = 'r'
            try:
                import toml as toml_
            except ImportError:
                import warnings
                warnings.warn('No TOML parser installed, cannot read configuration from `pyproject.toml`.')
                return None
    with open(config_file, read_mode) as rf:
        return toml_.load(rf)
