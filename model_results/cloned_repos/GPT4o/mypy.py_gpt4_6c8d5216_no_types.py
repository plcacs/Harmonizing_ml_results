"""This module includes classes and functions designed specifically for use with the mypy plugin."""
from __future__ import annotations
import sys
from collections.abc import Iterator
from configparser import ConfigParser
from typing import Any, Callable, cast
from mypy.errorcodes import ErrorCode
from mypy.expandtype import expand_type, expand_type_by_instance
from mypy.nodes import ARG_NAMED, ARG_NAMED_OPT, ARG_OPT, ARG_POS, ARG_STAR2, INVARIANT, MDEF, Argument, AssignmentStmt, Block, CallExpr, ClassDef, Context, Decorator, DictExpr, EllipsisExpr, Expression, FuncDef, IfStmt, JsonDict, MemberExpr, NameExpr, PassStmt, PlaceholderNode, RefExpr, Statement, StrExpr, SymbolTableNode, TempNode, TypeAlias, TypeInfo, Var
from mypy.options import Options
from mypy.plugin import CheckerPluginInterface, ClassDefContext, MethodContext, Plugin, ReportConfigContext, SemanticAnalyzerPluginInterface
from mypy.plugins.common import deserialize_and_fixup_type
from mypy.semanal import set_callable_name
from mypy.server.trigger import make_wildcard_trigger
from mypy.state import state
from mypy.typeops import map_type_from_supertype
from mypy.types import AnyType, CallableType, Instance, NoneType, Type, TypeOfAny, TypeType, TypeVarType, UnionType, get_proper_type
from mypy.typevars import fill_typevars
from mypy.util import get_unique_redefinition_name
from mypy.version import __version__ as mypy_version
from pydantic._internal import _fields
from pydantic.version import parse_mypy_version
CONFIGFILE_KEY = 'pydantic-mypy'
METADATA_KEY = 'pydantic-mypy-metadata'
BASEMODEL_FULLNAME = 'pydantic.main.BaseModel'
BASESETTINGS_FULLNAME = 'pydantic_settings.main.BaseSettings'
ROOT_MODEL_FULLNAME = 'pydantic.root_model.RootModel'
MODEL_METACLASS_FULLNAME = (
    'pydantic._internal._model_construction.ModelMetaclass')
FIELD_FULLNAME = 'pydantic.fields.Field'
DATACLASS_FULLNAME = 'pydantic.dataclasses.dataclass'
MODEL_VALIDATOR_FULLNAME = 'pydantic.functional_validators.model_validator'
DECORATOR_FULLNAMES = {'pydantic.functional_validators.field_validator',
    'pydantic.functional_validators.model_validator',
    'pydantic.functional_serializers.serializer',
    'pydantic.functional_serializers.model_serializer',
    'pydantic.deprecated.class_validators.validator',
    'pydantic.deprecated.class_validators.root_validator'}
IMPLICIT_CLASSMETHOD_DECORATOR_FULLNAMES = DECORATOR_FULLNAMES - {
    'pydantic.functional_serializers.model_serializer'}
MYPY_VERSION_TUPLE = parse_mypy_version(mypy_version)
BUILTINS_NAME = 'builtins'
__version__ = 2


def plugin(version):
    """`version` is the mypy version string.

    We might want to use this to print a warning if the mypy version being used is
    newer, or especially older, than we expect (or need).

    Args:
        version: The mypy version string.

    Return:
        The Pydantic mypy plugin type.
    """
    return PydanticPlugin


class PydanticPlugin(Plugin):
    """The Pydantic mypy plugin."""

    def __init__(self, options):
        self.plugin_config = PydanticPluginConfig(options)
        self._plugin_data = self.plugin_config.to_data()
        super().__init__(options)

    def get_base_class_hook(self, fullname):
        """Update Pydantic model class."""
        sym = self.lookup_fully_qualified(fullname)
        if sym and isinstance(sym.node, TypeInfo):
            if sym.node.has_base(BASEMODEL_FULLNAME):
                return self._pydantic_model_class_maker_callback
        return None

    def get_metaclass_hook(self, fullname):
        """Update Pydantic `ModelMetaclass` definition."""
        if fullname == MODEL_METACLASS_FULLNAME:
            return self._pydantic_model_metaclass_marker_callback
        return None

    def get_method_hook(self, fullname):
        """Adjust return type of `from_orm` method call."""
        if fullname.endswith('.from_orm'):
            return from_attributes_callback
        return None

    def report_config_data(self, ctx):
        """Return all plugin config data.

        Used by mypy to determine if cache needs to be discarded.
        """
        return self._plugin_data

    def _pydantic_model_class_maker_callback(self, ctx):
        transformer = PydanticModelTransformer(ctx.cls, ctx.reason, ctx.api,
            self.plugin_config)
        transformer.transform()

    def _pydantic_model_metaclass_marker_callback(self, ctx):
        """Reset dataclass_transform_spec attribute of ModelMetaclass.

        Let the plugin handle it. This behavior can be disabled
        if 'debug_dataclass_transform' is set to True', for testing purposes.
        """
        if self.plugin_config.debug_dataclass_transform:
            return
        info_metaclass = ctx.cls.info.declared_metaclass
        assert info_metaclass, "callback not passed from 'get_metaclass_hook'"
        if getattr(info_metaclass.type, 'dataclass_transform_spec', None):
            info_metaclass.type.dataclass_transform_spec = None


class PydanticPluginConfig:
    """A Pydantic mypy plugin config holder.

    Attributes:
        init_forbid_extra: Whether to add a `**kwargs` at the end of the generated `__init__` signature.
        init_typed: Whether to annotate fields in the generated `__init__`.
        warn_required_dynamic_aliases: Whether to raise required dynamic aliases error.
        debug_dataclass_transform: Whether to not reset `dataclass_transform_spec` attribute
            of `ModelMetaclass` for testing purposes.
    """
    __slots__ = ('init_forbid_extra', 'init_typed',
        'warn_required_dynamic_aliases', 'debug_dataclass_transform')
    init_forbid_extra: bool
    init_typed: bool
    warn_required_dynamic_aliases: bool
    debug_dataclass_transform: bool

    def __init__(self, options):
        if options.config_file is None:
            return
        toml_config = parse_toml(options.config_file)
        if toml_config is not None:
            config = toml_config.get('tool', {}).get('pydantic-mypy', {})
            for key in self.__slots__:
                setting = config.get(key, False)
                if not isinstance(setting, bool):
                    raise ValueError(
                        f'Configuration value must be a boolean for key: {key}'
                        )
                setattr(self, key, setting)
        else:
            plugin_config = ConfigParser()
            plugin_config.read(options.config_file)
            for key in self.__slots__:
                setting = plugin_config.getboolean(CONFIGFILE_KEY, key,
                    fallback=False)
                setattr(self, key, setting)

    def to_data(self):
        """Returns a dict of config names to their values."""
        return {key: getattr(self, key) for key in self.__slots__}


def from_attributes_callback(ctx):
    """Raise an error if from_attributes is not enabled."""
    model_type: Instance
    ctx_type = ctx.type
    if isinstance(ctx_type, TypeType):
        ctx_type = ctx_type.item
    if isinstance(ctx_type, CallableType) and isinstance(ctx_type.ret_type,
        Instance):
        model_type = ctx_type.ret_type
    elif isinstance(ctx_type, Instance):
        model_type = ctx_type
    else:
        detail = (
            f'ctx.type: {ctx_type} (of type {ctx_type.__class__.__name__})')
        error_unexpected_behavior(detail, ctx.api, ctx.context)
        return ctx.default_return_type
    pydantic_metadata = model_type.type.metadata.get(METADATA_KEY)
    if pydantic_metadata is None:
        return ctx.default_return_type
    if not model_type.type.has_base(BASEMODEL_FULLNAME):
        return ctx.default_return_type
    from_attributes = pydantic_metadata.get('config', {}).get('from_attributes'
        )
    if from_attributes is not True:
        error_from_attributes(model_type.type.name, ctx.api, ctx.context)
    return ctx.default_return_type


class PydanticModelField:
    """Based on mypy.plugins.dataclasses.DataclassAttribute."""

    def __init__(self, name, alias, is_frozen, has_dynamic_alias,
        has_default, strict, line, column, type, info):
        self.name = name
        self.alias = alias
        self.is_frozen = is_frozen
        self.has_dynamic_alias = has_dynamic_alias
        self.has_default = has_default
        self.strict = strict
        self.line = line
        self.column = column
        self.type = type
        self.info = info

    def to_argument(self, current_info, typed, model_strict, force_optional,
        use_alias, api, force_typevars_invariant, is_root_model_root):
        """Based on mypy.plugins.dataclasses.DataclassAttribute.to_argument."""
        variable = self.to_var(current_info, api, use_alias,
            force_typevars_invariant)
        strict = model_strict if self.strict is None else self.strict
        if typed or strict:
            type_annotation = self.expand_type(current_info, api)
        else:
            type_annotation = AnyType(TypeOfAny.explicit)
        return Argument(variable=variable, type_annotation=type_annotation,
            initializer=None, kind=ARG_OPT if is_root_model_root else 
            ARG_NAMED_OPT if force_optional or self.has_default else ARG_NAMED)

    def expand_type(self, current_info, api, force_typevars_invariant=False):
        """Based on mypy.plugins.dataclasses.DataclassAttribute.expand_type."""
        if force_typevars_invariant:
            if isinstance(self.type, TypeVarType):
                modified_type = self.type.copy_modified()
                modified_type.variance = INVARIANT
                self.type = modified_type
        if self.type is not None and self.info.self_type is not None:
            with state.strict_optional_set(api.options.strict_optional):
                filled_with_typevars = fill_typevars(current_info)
                assert isinstance(filled_with_typevars, Instance)
                if force_typevars_invariant:
                    for arg in filled_with_typevars.args:
                        if isinstance(arg, TypeVarType):
                            arg.variance = INVARIANT
                expanded_type = expand_type(self.type, {self.info.self_type
                    .id: filled_with_typevars})
                if isinstance(expanded_type, Instance) and is_root_model(
                    expanded_type.type):
                    root_type = cast(Type, expanded_type.type['root'].type)
                    expanded_root_type = expand_type_by_instance(root_type,
                        expanded_type)
                    expanded_type = UnionType([expanded_type,
                        expanded_root_type])
                return expanded_type
        return self.type

    def to_var(self, current_info, api, use_alias, force_typevars_invariant
        =False):
        """Based on mypy.plugins.dataclasses.DataclassAttribute.to_var."""
        if use_alias and self.alias is not None:
            name = self.alias
        else:
            name = self.name
        return Var(name, self.expand_type(current_info, api,
            force_typevars_invariant))

    def serialize(self):
        """Based on mypy.plugins.dataclasses.DataclassAttribute.serialize."""
        assert self.type
        return {'name': self.name, 'alias': self.alias, 'is_frozen': self.
            is_frozen, 'has_dynamic_alias': self.has_dynamic_alias,
            'has_default': self.has_default, 'strict': self.strict, 'line':
            self.line, 'column': self.column, 'type': self.type.serialize()}

    @classmethod
    def deserialize(cls, info, data, api):
        """Based on mypy.plugins.dataclasses.DataclassAttribute.deserialize."""
        data = data.copy()
        typ = deserialize_and_fixup_type(data.pop('type'), api)
        return cls(type=typ, info=info, **data)

    def expand_typevar_from_subtype(self, sub_type, api):
        """Expands type vars in the context of a subtype when an attribute is inherited
        from a generic super type.
        """
        if self.type is not None:
            with state.strict_optional_set(api.options.strict_optional):
                self.type = map_type_from_supertype(self.type, sub_type,
                    self.info)


class PydanticModelClassVar:
    """Based on mypy.plugins.dataclasses.DataclassAttribute.

    ClassVars are ignored by subclasses.

    Attributes:
        name: the ClassVar name
    """

    def __init__(self, name):
        self.name = name

    @classmethod
    def deserialize(cls, data):
        """Based on mypy.plugins.dataclasses.DataclassAttribute.deserialize."""
        data = data.copy()
        return cls(**data)

    def serialize(self):
        """Based on mypy.plugins.dataclasses.DataclassAttribute.serialize."""
        return {'name': self.name}


class PydanticModelTransformer:
    """Transform the BaseModel subclass according to the plugin settings.

    Attributes:
        tracked_config_fields: A set of field configs that the plugin has to track their value.
    """
    tracked_config_fields: set[str] = {'extra', 'frozen', 'from_attributes',
        'populate_by_name', 'alias_generator', 'strict'}

    def __init__(self, cls, reason, api, plugin_config):
        self._cls = cls
        self._reason = reason
        self._api = api
        self.plugin_config = plugin_config

    def transform(self):
        """Configures the BaseModel subclass according to the plugin settings.

        In particular:

        * determines the model config and fields,
        * adds a fields-aware signature for the initializer and construct methods
        * freezes the class if frozen = True
        * stores the fields, config, and if the class is settings in the mypy metadata for access by subclasses
        """
        info = self._cls.info
        is_a_root_model = is_root_model(info)
        config = self.collect_config()
        fields, class_vars = self.collect_fields_and_class_vars(config,
            is_a_root_model)
        if fields is None or class_vars is None:
            return False
        for field in fields:
            if field.type is None:
                return False
        is_settings = info.has_base(BASESETTINGS_FULLNAME)
        self.add_initializer(fields, config, is_settings, is_a_root_model)
        self.add_model_construct_method(fields, config, is_settings,
            is_a_root_model)
        self.set_frozen(fields, self._api, frozen=config.frozen is True)
        self.adjust_decorator_signatures()
        info.metadata[METADATA_KEY] = {'fields': {field.name: field.
            serialize() for field in fields}, 'class_vars': {class_var.name:
            class_var.serialize() for class_var in class_vars}, 'config':
            config.get_values_dict()}
        return True

    def adjust_decorator_signatures(self):
        """When we decorate a function `f` with `pydantic.validator(...)`, `pydantic.field_validator`
        or `pydantic.serializer(...)`, mypy sees `f` as a regular method taking a `self` instance,
        even though pydantic internally wraps `f` with `classmethod` if necessary.

        Teach mypy this by marking any function whose outermost decorator is a `validator()`,
        `field_validator()` or `serializer()` call as a `classmethod`.
        """
        for sym in self._cls.info.names.values():
            if isinstance(sym.node, Decorator):
                first_dec = sym.node.original_decorators[0]
                if (isinstance(first_dec, CallExpr) and isinstance(
                    first_dec.callee, NameExpr) and first_dec.callee.
                    fullname in IMPLICIT_CLASSMETHOD_DECORATOR_FULLNAMES and
                    not (first_dec.callee.fullname ==
                    MODEL_VALIDATOR_FULLNAME and any(first_dec.arg_names[i] ==
                    'mode' and isinstance(arg, StrExpr) and arg.value ==
                    'after' for i, arg in enumerate(first_dec.args)))):
                    sym.node.func.is_class = True

    def collect_config(self):
        """Collects the values of the config attributes that are used by the plugin, accounting for parent classes."""
        cls = self._cls
        config = ModelConfigData()
        has_config_kwargs = False
        has_config_from_namespace = False
        for name, expr in cls.keywords.items():
            config_data = self.get_config_update(name, expr)
            if config_data:
                has_config_kwargs = True
                config.update(config_data)
        stmt: Statement | None = None
        for stmt in cls.defs.body:
            if not isinstance(stmt, (AssignmentStmt, ClassDef)):
                continue
            if isinstance(stmt, AssignmentStmt):
                lhs = stmt.lvalues[0]
                if not isinstance(lhs, NameExpr) or lhs.name != 'model_config':
                    continue
                if isinstance(stmt.rvalue, CallExpr):
                    for arg_name, arg in zip(stmt.rvalue.arg_names, stmt.
                        rvalue.args):
                        if arg_name is None:
                            continue
                        config.update(self.get_config_update(arg_name, arg,
                            lax_extra=True))
                elif isinstance(stmt.rvalue, DictExpr):
                    for key_expr, value_expr in stmt.rvalue.items:
                        if not isinstance(key_expr, StrExpr):
                            continue
                        config.update(self.get_config_update(key_expr.value,
                            value_expr))
            elif isinstance(stmt, ClassDef):
                if stmt.name != 'Config':
                    continue
                for substmt in stmt.defs.body:
                    if not isinstance(substmt, AssignmentStmt):
                        continue
                    lhs = substmt.lvalues[0]
                    if not isinstance(lhs, NameExpr):
                        continue
                    config.update(self.get_config_update(lhs.name, substmt.
                        rvalue))
            if has_config_kwargs:
                self._api.fail(
                    'Specifying config in two places is ambiguous, use either Config attribute or class kwargs'
                    , cls)
                break
            has_config_from_namespace = True
        if has_config_kwargs or has_config_from_namespace:
            if (stmt and config.has_alias_generator and not config.
                populate_by_name and self.plugin_config.
                warn_required_dynamic_aliases):
                error_required_dynamic_aliases(self._api, stmt)
        for info in cls.info.mro[1:]:
            if METADATA_KEY not in info.metadata:
                continue
            self._api.add_plugin_dependency(make_wildcard_trigger(info.
                fullname))
            for name, value in info.metadata[METADATA_KEY]['config'].items():
                config.setdefault(name, value)
        return config

    def collect_fields_and_class_vars(self, model_config, is_root_model):
        """Collects the fields for the model, accounting for parent classes."""
        cls = self._cls
        found_fields: dict[str, PydanticModelField] = {}
        found_class_vars: dict[str, PydanticModelClassVar] = {}
        for info in reversed(cls.info.mro[1:-1]):
            if METADATA_KEY not in info.metadata:
                continue
            self._api.add_plugin_dependency(make_wildcard_trigger(info.
                fullname))
            for name, data in info.metadata[METADATA_KEY]['fields'].items():
                field = PydanticModelField.deserialize(info, data, self._api)
                field.expand_typevar_from_subtype(cls.info, self._api)
                found_fields[name] = field
                sym_node = cls.info.names.get(name)
                if sym_node and sym_node.node and not isinstance(sym_node.
                    node, Var):
                    self._api.fail(
                        'BaseModel field may only be overridden by another field'
                        , sym_node.node)
            for name, data in info.metadata[METADATA_KEY]['class_vars'].items(
                ):
                found_class_vars[name] = PydanticModelClassVar.deserialize(data
                    )
        current_field_names: set[str] = set()
        current_class_vars_names: set[str] = set()
        for stmt in self._get_assignment_statements_from_block(cls.defs):
            maybe_field = self.collect_field_or_class_var_from_stmt(stmt,
                model_config, found_class_vars)
            if maybe_field is None:
                continue
            lhs = stmt.lvalues[0]
            assert isinstance(lhs, NameExpr)
            if isinstance(maybe_field, PydanticModelField):
                if is_root_model and lhs.name != 'root':
                    error_extra_fields_on_root_model(self._api, stmt)
                else:
                    current_field_names.add(lhs.name)
                    found_fields[lhs.name] = maybe_field
            elif isinstance(maybe_field, PydanticModelClassVar):
                current_class_vars_names.add(lhs.name)
                found_class_vars[lhs.name] = maybe_field
        return list(found_fields.values()), list(found_class_vars.values())

    def _get_assignment_statements_from_if_statement(self, stmt):
        for body in stmt.body:
            if not body.is_unreachable:
                yield from self._get_assignment_statements_from_block(body)
        if stmt.else_body is not None and not stmt.else_body.is_unreachable:
            yield from self._get_assignment_statements_from_block(stmt.
                else_body)

    def _get_assignment_statements_from_block(self, block):
        for stmt in block.body:
            if isinstance(stmt, AssignmentStmt):
                yield stmt
            elif isinstance(stmt, IfStmt):
                yield from self._get_assignment_statements_from_if_statement(
                    stmt)

    def collect_field_or_class_var_from_stmt(self, stmt, model_config,
        class_vars):
        """Get pydantic model field from statement.

        Args:
            stmt: The statement.
            model_config: Configuration settings for the model.
            class_vars: ClassVars already known to be defined on the model.

        Returns:
            A pydantic model field if it could find the field in statement. Otherwise, `None`.
        """
        cls = self._cls
        lhs = stmt.lvalues[0]
        if not isinstance(lhs, NameExpr) or not _fields.is_valid_field_name(lhs
            .name) or lhs.name == 'model_config':
            return None
        if not stmt.new_syntax:
            if isinstance(stmt.rvalue, CallExpr) and isinstance(stmt.rvalue
                .callee, CallExpr) and isinstance(stmt.rvalue.callee.callee,
                NameExpr
                ) and stmt.rvalue.callee.callee.fullname in DECORATOR_FULLNAMES:
                return None
            if lhs.name in class_vars:
                return None
            error_untyped_fields(self._api, stmt)
            return None
        lhs = stmt.lvalues[0]
        if not isinstance(lhs, NameExpr):
            return None
        if not _fields.is_valid_field_name(lhs.name
            ) or lhs.name == 'model_config':
            return None
        sym = cls.info.names.get(lhs.name)
        if sym is None:
            return None
        node = sym.node
        if isinstance(node, PlaceholderNode):
            return None
        if isinstance(node, TypeAlias):
            self._api.fail(
                'Type aliases inside BaseModel definitions are not supported at runtime'
                , node)
            return None
        if not isinstance(node, Var):
            return None
        if node.is_classvar:
            return PydanticModelClassVar(lhs.name)
        node_type = get_proper_type(node.type)
        if isinstance(node_type, Instance
            ) and node_type.type.fullname == 'dataclasses.InitVar':
            self._api.fail('InitVar is not supported in BaseModel', node)
        has_default = self.get_has_default(stmt)
        strict = self.get_strict(stmt)
        if sym.type is None and node.is_final and node.is_inferred:
            typ = self._api.analyze_simple_literal_type(stmt.rvalue,
                is_final=True)
            if typ:
                node.type = typ
            else:
                self._api.fail(
                    'Need type argument for Final[...] with non-literal default in BaseModel'
                    , stmt)
                node.type = AnyType(TypeOfAny.from_error)
        if node.is_final and has_default:
            return PydanticModelClassVar(lhs.name)
        alias, has_dynamic_alias = self.get_alias_info(stmt)
        if (has_dynamic_alias and not model_config.populate_by_name and
            self.plugin_config.warn_required_dynamic_aliases):
            error_required_dynamic_aliases(self._api, stmt)
        is_frozen = self.is_field_frozen(stmt)
        init_type = self._infer_dataclass_attr_init_type(sym, lhs.name, stmt)
        return PydanticModelField(name=lhs.name, has_dynamic_alias=
            has_dynamic_alias, has_default=has_default, strict=strict,
            alias=alias, is_frozen=is_frozen, line=stmt.line, column=stmt.
            column, type=init_type, info=cls.info)

    def _infer_dataclass_attr_init_type(self, sym, name, context):
        """Infer __init__ argument type for an attribute.

        In particular, possibly use the signature of __set__.
        """
        default = sym.type
        if sym.implicit:
            return default
        t = get_proper_type(sym.type)
        if not isinstance(t, Instance):
            return default
        setter = t.type.get('__set__')
        if setter:
            if isinstance(setter.node, FuncDef):
                super_info = t.type.get_containing_type_info('__set__')
                assert super_info
                if setter.type:
                    setter_type = get_proper_type(map_type_from_supertype(
                        setter.type, t.type, super_info))
                else:
                    return AnyType(TypeOfAny.unannotated)
                if isinstance(setter_type, CallableType
                    ) and setter_type.arg_kinds == [ARG_POS, ARG_POS, ARG_POS]:
                    return expand_type_by_instance(setter_type.arg_types[2], t)
                else:
                    self._api.fail(
                        f'Unsupported signature for "__set__" in "{t.type.name}"'
                        , context)
            else:
                self._api.fail(f'Unsupported "__set__" in "{t.type.name}"',
                    context)
        return default

    def add_initializer(self, fields, config, is_settings, is_root_model):
        """Adds a fields-aware `__init__` method to the class.

        The added `__init__` will be annotated with types vs. all `Any` depending on the plugin settings.
        """
        if '__init__' in self._cls.info.names and not self._cls.info.names[
            '__init__'].plugin_generated:
            return
        typed = self.plugin_config.init_typed
        model_strict = bool(config.strict)
        use_alias = config.populate_by_name is not True
        requires_dynamic_aliases = bool(config.has_alias_generator and not
            config.populate_by_name)
        args = self.get_field_arguments(fields, typed=typed, model_strict=
            model_strict, requires_dynamic_aliases=requires_dynamic_aliases,
            use_alias=use_alias, is_settings=is_settings, is_root_model=
            is_root_model, force_typevars_invariant=True)
        if is_settings:
            base_settings_node = self._api.lookup_fully_qualified(
                BASESETTINGS_FULLNAME).node
            assert isinstance(base_settings_node, TypeInfo)
            if '__init__' in base_settings_node.names:
                base_settings_init_node = base_settings_node.names['__init__'
                    ].node
                assert isinstance(base_settings_init_node, FuncDef)
                if (base_settings_init_node is not None and 
                    base_settings_init_node.type is not None):
                    func_type = base_settings_init_node.type
                    assert isinstance(func_type, CallableType)
                    for arg_idx, arg_name in enumerate(func_type.arg_names):
                        if arg_name is None or arg_name.startswith('__'
                            ) or not arg_name.startswith('_'):
                            continue
                        analyzed_variable_type = self._api.anal_type(func_type
                            .arg_types[arg_idx])
                        variable = Var(arg_name, analyzed_variable_type)
                        args.append(Argument(variable,
                            analyzed_variable_type, None, ARG_OPT))
        if not self.should_init_forbid_extra(fields, config):
            var = Var('kwargs')
            args.append(Argument(var, AnyType(TypeOfAny.explicit), None,
                ARG_STAR2))
        add_method(self._api, self._cls, '__init__', args=args, return_type
            =NoneType())

    def add_model_construct_method(self, fields, config, is_settings,
        is_root_model):
        """Adds a fully typed `model_construct` classmethod to the class.

        Similar to the fields-aware __init__ method, but always uses the field names (not aliases),
        and does not treat settings fields as optional.
        """
        set_str = self._api.named_type(f'{BUILTINS_NAME}.set', [self._api.
            named_type(f'{BUILTINS_NAME}.str')])
        optional_set_str = UnionType([set_str, NoneType()])
        fields_set_argument = Argument(Var('_fields_set', optional_set_str),
            optional_set_str, None, ARG_OPT)
        with state.strict_optional_set(self._api.options.strict_optional):
            args = self.get_field_arguments(fields, typed=True,
                model_strict=bool(config.strict), requires_dynamic_aliases=
                False, use_alias=False, is_settings=is_settings,
                is_root_model=is_root_model)
        if not self.should_init_forbid_extra(fields, config):
            var = Var('kwargs')
            args.append(Argument(var, AnyType(TypeOfAny.explicit), None,
                ARG_STAR2))
        args = args + [fields_set_argument] if is_root_model else [
            fields_set_argument] + args
        add_method(self._api, self._cls, 'model_construct', args=args,
            return_type=fill_typevars(self._cls.info), is_classmethod=True)

    def set_frozen(self, fields, api, frozen):
        """Marks all fields as properties so that attempts to set them trigger mypy errors.

        This is the same approach used by the attrs and dataclasses plugins.
        """
        info = self._cls.info
        for field in fields:
            sym_node = info.names.get(field.name)
            if sym_node is not None:
                var = sym_node.node
                if isinstance(var, Var):
                    var.is_property = frozen or field.is_frozen
                elif isinstance(var, PlaceholderNode
                    ) and not self._api.final_iteration:
                    self._api.defer()
                else:
                    try:
                        var_str = str(var)
                    except TypeError:
                        var_str = repr(var)
                    detail = (
                        f'sym_node.node: {var_str} (of type {var.__class__})')
                    error_unexpected_behavior(detail, self._api, self._cls)
            else:
                var = field.to_var(info, api, use_alias=False)
                var.info = info
                var.is_property = frozen
                var._fullname = info.fullname + '.' + var.name
                info.names[var.name] = SymbolTableNode(MDEF, var)

    def get_config_update(self, name, arg, lax_extra=False):
        """Determines the config update due to a single kwarg in the ConfigDict definition.

        Warns if a tracked config attribute is set to a value the plugin doesn't know how to interpret (e.g., an int)
        """
        if name not in self.tracked_config_fields:
            return None
        if name == 'extra':
            if isinstance(arg, StrExpr):
                forbid_extra = arg.value == 'forbid'
            elif isinstance(arg, MemberExpr):
                forbid_extra = arg.name == 'forbid'
            else:
                if not lax_extra:
                    error_invalid_config_value(name, self._api, arg)
                return None
            return ModelConfigData(forbid_extra=forbid_extra)
        if name == 'alias_generator':
            has_alias_generator = True
            if isinstance(arg, NameExpr) and arg.fullname == 'builtins.None':
                has_alias_generator = False
            return ModelConfigData(has_alias_generator=has_alias_generator)
        if isinstance(arg, NameExpr) and arg.fullname in ('builtins.True',
            'builtins.False'):
            return ModelConfigData(**{name: arg.fullname == 'builtins.True'})
        error_invalid_config_value(name, self._api, arg)
        return None

    @staticmethod
    def get_has_default(stmt):
        """Returns a boolean indicating whether the field defined in `stmt` is a required field."""
        expr = stmt.rvalue
        if isinstance(expr, TempNode):
            return False
        if isinstance(expr, CallExpr) and isinstance(expr.callee, RefExpr
            ) and expr.callee.fullname == FIELD_FULLNAME:
            for arg, name in zip(expr.args, expr.arg_names):
                if name is None or name == 'default':
                    return arg.__class__ is not EllipsisExpr
                if name == 'default_factory':
                    return not (isinstance(arg, NameExpr) and arg.fullname ==
                        'builtins.None')
            return False
        return not isinstance(expr, EllipsisExpr)

    @staticmethod
    def get_strict(stmt):
        """Returns a the `strict` value of a field if defined, otherwise `None`."""
        expr = stmt.rvalue
        if isinstance(expr, CallExpr) and isinstance(expr.callee, RefExpr
            ) and expr.callee.fullname == FIELD_FULLNAME:
            for arg, name in zip(expr.args, expr.arg_names):
                if name != 'strict':
                    continue
                if isinstance(arg, NameExpr):
                    if arg.fullname == 'builtins.True':
                        return True
                    elif arg.fullname == 'builtins.False':
                        return False
                return None
        return None

    @staticmethod
    def get_alias_info(stmt):
        """Returns a pair (alias, has_dynamic_alias), extracted from the declaration of the field defined in `stmt`.

        `has_dynamic_alias` is True if and only if an alias is provided, but not as a string literal.
        If `has_dynamic_alias` is True, `alias` will be None.
        """
        expr = stmt.rvalue
        if isinstance(expr, TempNode):
            return None, False
        if not (isinstance(expr, CallExpr) and isinstance(expr.callee,
            RefExpr) and expr.callee.fullname == FIELD_FULLNAME):
            return None, False
        if 'validation_alias' in expr.arg_names:
            arg = expr.args[expr.arg_names.index('validation_alias')]
        elif 'alias' in expr.arg_names:
            arg = expr.args[expr.arg_names.index('alias')]
        else:
            return None, False
        if isinstance(arg, StrExpr):
            return arg.value, False
        else:
            return None, True

    @staticmethod
    def is_field_frozen(stmt):
        """Returns whether the field is frozen, extracted from the declaration of the field defined in `stmt`.

        Note that this is only whether the field was declared to be frozen in a `<field_name> = Field(frozen=True)`
        sense; this does not determine whether the field is frozen because the entire model is frozen; that is
        handled separately.
        """
        expr = stmt.rvalue
        if isinstance(expr, TempNode):
            return False
        if not (isinstance(expr, CallExpr) and isinstance(expr.callee,
            RefExpr) and expr.callee.fullname == FIELD_FULLNAME):
            return False
        for i, arg_name in enumerate(expr.arg_names):
            if arg_name == 'frozen':
                arg = expr.args[i]
                return isinstance(arg, NameExpr
                    ) and arg.fullname == 'builtins.True'
        return False

    def get_field_arguments(self, fields, typed, model_strict, use_alias,
        requires_dynamic_aliases, is_settings, is_root_model,
        force_typevars_invariant=False):
        """Helper function used during the construction of the `__init__` and `model_construct` method signatures.

        Returns a list of mypy Argument instances for use in the generated signatures.
        """
        info = self._cls.info
        arguments = [field.to_argument(info, typed=typed, model_strict=
            model_strict, force_optional=requires_dynamic_aliases or
            is_settings, use_alias=use_alias, api=self._api,
            force_typevars_invariant=force_typevars_invariant,
            is_root_model_root=is_root_model and field.name == 'root') for
            field in fields if not (use_alias and field.has_dynamic_alias)]
        return arguments

    def should_init_forbid_extra(self, fields, config):
        """Indicates whether the generated `__init__` should get a `**kwargs` at the end of its signature.

        We disallow arbitrary kwargs if the extra config setting is "forbid", or if the plugin config says to,
        *unless* a required dynamic alias is present (since then we can't determine a valid signature).
        """
        if not config.populate_by_name:
            if self.is_dynamic_alias_present(fields, bool(config.
                has_alias_generator)):
                return False
        if config.forbid_extra:
            return True
        return self.plugin_config.init_forbid_extra

    @staticmethod
    def is_dynamic_alias_present(fields, has_alias_generator):
        """Returns whether any fields on the model have a "dynamic alias", i.e., an alias that cannot be
        determined during static analysis.
        """
        for field in fields:
            if field.has_dynamic_alias:
                return True
        if has_alias_generator:
            for field in fields:
                if field.alias is None:
                    return True
        return False


class ModelConfigData:
    """Pydantic mypy plugin model config class."""

    def __init__(self, forbid_extra=None, frozen=None, from_attributes=None,
        populate_by_name=None, has_alias_generator=None, strict=None):
        self.forbid_extra = forbid_extra
        self.frozen = frozen
        self.from_attributes = from_attributes
        self.populate_by_name = populate_by_name
        self.has_alias_generator = has_alias_generator
        self.strict = strict

    def get_values_dict(self):
        """Returns a dict of Pydantic model config names to their values.

        It includes the config if config value is not `None`.
        """
        return {k: v for k, v in self.__dict__.items() if v is not None}

    def update(self, config):
        """Update Pydantic model config values."""
        if config is None:
            return
        for k, v in config.get_values_dict().items():
            setattr(self, k, v)

    def setdefault(self, key, value):
        """Set default value for Pydantic model config if config value is `None`."""
        if getattr(self, key) is None:
            setattr(self, key, value)


def is_root_model(info):
    """Return whether the type info is a root model subclass (or the `RootModel` class itself)."""
    return info.has_base(ROOT_MODEL_FULLNAME)


ERROR_ORM = ErrorCode('pydantic-orm', 'Invalid from_attributes call',
    'Pydantic')
ERROR_CONFIG = ErrorCode('pydantic-config', 'Invalid config value', 'Pydantic')
ERROR_ALIAS = ErrorCode('pydantic-alias', 'Dynamic alias disallowed',
    'Pydantic')
ERROR_UNEXPECTED = ErrorCode('pydantic-unexpected', 'Unexpected behavior',
    'Pydantic')
ERROR_UNTYPED = ErrorCode('pydantic-field', 'Untyped field disallowed',
    'Pydantic')
ERROR_FIELD_DEFAULTS = ErrorCode('pydantic-field', 'Invalid Field defaults',
    'Pydantic')
ERROR_EXTRA_FIELD_ROOT_MODEL = ErrorCode('pydantic-field',
    'Extra field on RootModel subclass', 'Pydantic')


def error_from_attributes(model_name, api, context):
    """Emits an error when the model does not have `from_attributes=True`."""
    api.fail(f'"{model_name}" does not have from_attributes=True', context,
        code=ERROR_ORM)


def error_invalid_config_value(name, api, context):
    """Emits an error when the config value is invalid."""
    api.fail(f'Invalid value for "Config.{name}"', context, code=ERROR_CONFIG)


def error_required_dynamic_aliases(api, context):
    """Emits required dynamic aliases error.

    This will be called when `warn_required_dynamic_aliases=True`.
    """
    api.fail('Required dynamic aliases disallowed', context, code=ERROR_ALIAS)


def error_unexpected_behavior(detail, api, context):
    """Emits unexpected behavior error."""
    link = 'https://github.com/pydantic/pydantic/issues/new/choose'
    full_message = (
        f'The pydantic mypy plugin ran into unexpected behavior: {detail}\n')
    full_message += (
        f'Please consider reporting this bug at {link} so we can try to fix it!'
        )
    api.fail(full_message, context, code=ERROR_UNEXPECTED)


def error_untyped_fields(api, context):
    """Emits an error when there is an untyped field in the model."""
    api.fail('Untyped fields disallowed', context, code=ERROR_UNTYPED)


def error_extra_fields_on_root_model(api, context):
    """Emits an error when there is more than just a root field defined for a subclass of RootModel."""
    api.fail('Only `root` is allowed as a field of a `RootModel`', context,
        code=ERROR_EXTRA_FIELD_ROOT_MODEL)


def add_method(api, cls, name, args, return_type, self_type=None, tvar_def=
    None, is_classmethod=False):
    """Very closely related to `mypy.plugins.common.add_method_to_class`, with a few pydantic-specific changes."""
    info = cls.info
    if name in info.names:
        sym = info.names[name]
        if sym.plugin_generated and isinstance(sym.node, FuncDef):
            cls.defs.body.remove(sym.node)
    if isinstance(api, SemanticAnalyzerPluginInterface):
        function_type = api.named_type('builtins.function')
    else:
        function_type = api.named_generic_type('builtins.function', [])
    if is_classmethod:
        self_type = self_type or TypeType(fill_typevars(info))
        first = [Argument(Var('_cls'), self_type, None, ARG_POS, True)]
    else:
        self_type = self_type or fill_typevars(info)
        first = [Argument(Var('__pydantic_self__'), self_type, None, ARG_POS)]
    args = first + args
    arg_types, arg_names, arg_kinds = [], [], []
    for arg in args:
        assert arg.type_annotation, 'All arguments must be fully typed.'
        arg_types.append(arg.type_annotation)
        arg_names.append(arg.variable.name)
        arg_kinds.append(arg.kind)
    signature = CallableType(arg_types, arg_kinds, arg_names, return_type,
        function_type)
    if tvar_def:
        signature.variables = [tvar_def]
    func = FuncDef(name, args, Block([PassStmt()]))
    func.info = info
    func.type = set_callable_name(signature, func)
    func.is_class = is_classmethod
    func._fullname = info.fullname + '.' + name
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


def parse_toml(config_file):
    """Returns a dict of config keys to values.

    It reads configs from toml file and returns `None` if the file is not a toml file.
    """
    if not config_file.endswith('.toml'):
        return None
    if sys.version_info >= (3, 11):
        import tomllib as toml_
    else:
        try:
            import tomli as toml_
        except ImportError:
            import warnings
            warnings.warn(
                'No TOML parser installed, cannot read configuration from `pyproject.toml`.'
                )
            return None
    with open(config_file, 'rb') as rf:
        return toml_.load(rf)
