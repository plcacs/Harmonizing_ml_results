"""This module includes classes and functions designed specifically for use with the mypy plugin."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from configparser import ConfigParser
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union, cast

from mypy.errorcodes import ErrorCode
from mypy.expandtype import expand_type, expand_type_by_instance
from mypy.nodes import (
    ARG_NAMED,
    ARG_NAMED_OPT,
    ARG_OPT,
    ARG_POS,
    ARG_STAR2,
    INVARIANT,
    MDEF,
    Argument,
    AssignmentStmt,
    Block,
    CallExpr,
    ClassDef,
    Context,
    Decorator,
    DictExpr,
    EllipsisExpr,
    Expression,
    FuncDef,
    IfStmt,
    JsonDict,
    MemberExpr,
    NameExpr,
    PassStmt,
    PlaceholderNode,
    RefExpr,
    Statement,
    StrExpr,
    SymbolTableNode,
    TempNode,
    TypeAlias,
    TypeInfo,
    Var,
)
from mypy.options import Options
from mypy.plugin import (
    CheckerPluginInterface,
    ClassDefContext,
    MethodContext,
    Plugin,
    ReportConfigContext,
    SemanticAnalyzerPluginInterface,
)
from mypy.plugins.common import (
    deserialize_and_fixup_type,
)
from mypy.semanal import set_callable_name
from mypy.server.trigger import make_wildcard_trigger
from mypy.state import state
from mypy.typeops import map_type_from_supertype
from mypy.types import (
    AnyType,
    CallableType,
    Instance,
    NoneType,
    Type,
    TypeOfAny,
    TypeType,
    TypeVarType,
    UnionType,
    get_proper_type,
)
from mypy.typevars import fill_typevars
from mypy.util import get_unique_redefinition_name
from mypy.version import __version__ as mypy_version

from pydantic._internal import _fields
from pydantic.version import parse_mypy_version

CONFIGFILE_KEY: str = 'pydantic-mypy'
METADATA_KEY: str = 'pydantic-mypy-metadata'
BASEMODEL_FULLNAME: str = 'pydantic.main.BaseModel'
BASESETTINGS_FULLNAME: str = 'pydantic_settings.main.BaseSettings'
ROOT_MODEL_FULLNAME: str = 'pydantic.root_model.RootModel'
MODEL_METACLASS_FULLNAME: str = 'pydantic._internal._model_construction.ModelMetaclass'
FIELD_FULLNAME: str = 'pydantic.fields.Field'
DATACLASS_FULLNAME: str = 'pydantic.dataclasses.dataclass'
MODEL_VALIDATOR_FULLNAME: str = 'pydantic.functional_validators.model_validator'
DECORATOR_FULLNAMES: Set[str] = {
    'pydantic.functional_validators.field_validator',
    'pydantic.functional_validators.model_validator',
    'pydantic.functional_serializers.serializer',
    'pydantic.functional_serializers.model_serializer',
    'pydantic.deprecated.class_validators.validator',
    'pydantic.deprecated.class_validators.root_validator',
}
IMPLICIT_CLASSMETHOD_DECORATOR_FULLNAMES: Set[str] = DECORATOR_FULLNAMES - {'pydantic.functional_serializers.model_serializer'}


MYPY_VERSION_TUPLE: Tuple[int, ...] = parse_mypy_version(mypy_version)
BUILTINS_NAME: str = 'builtins'

# Increment version if plugin changes and mypy caches should be invalidated
__version__: int = 2


def plugin(version: str) -> type[Plugin]:
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

    def __init__(self, options: Options) -> None:
        self.plugin_config: PydanticPluginConfig = PydanticPluginConfig(options)
        self._plugin_data: Dict[str, Any] = self.plugin_config.to_data()
        super().__init__(options)

    def get_base_class_hook(self, fullname: str) -> Optional[Callable[[ClassDefContext], None]]:
        """Update Pydantic model class."""
        sym = self.lookup_fully_qualified(fullname)
        if sym and isinstance(sym.node, TypeInfo):  # pragma: no branch
            # No branching may occur if the mypy cache has not been cleared
            if sym.node.has_base(BASEMODEL_FULLNAME):
                return self._pydantic_model_class_maker_callback
        return None

    def get_metaclass_hook(self, fullname: str) -> Optional[Callable[[ClassDefContext], None]]:
        """Update Pydantic `ModelMetaclass` definition."""
        if fullname == MODEL_METACLASS_FULLNAME:
            return self._pydantic_model_metaclass_marker_callback
        return None

    def get_method_hook(self, fullname: str) -> Optional[Callable[[MethodContext], Type]]:
        """Adjust return type of `from_orm` method call."""
        if fullname.endswith('.from_orm'):
            return from_attributes_callback
        return None

    def report_config_data(self, ctx: ReportConfigContext) -> Dict[str, Any]:
        """Return all plugin config data.

        Used by mypy to determine if cache needs to be discarded.
        """
        return self._plugin_data

    def _pydantic_model_class_maker_callback(self, ctx: ClassDefContext) -> None:
        transformer: PydanticModelTransformer = PydanticModelTransformer(ctx.cls, ctx.reason, ctx.api, self.plugin_config)
        transformer.transform()

    def _pydantic_model_metaclass_marker_callback(self, ctx: ClassDefContext) -> None:
        """Reset dataclass_transform_spec attribute of ModelMetaclass.

        Let the plugin handle it. This behavior can be disabled
        if 'debug_dataclass_transform' is set to True', for testing purposes.
        """
        if self.plugin_config.debug_dataclass_transform:
            return
        info_metaclass: Optional[TypeInfo] = ctx.cls.info.declared_metaclass
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

    __slots__: Tuple[str, ...] = (
        'init_forbid_extra',
        'init_typed',
        'warn_required_dynamic_aliases',
        'debug_dataclass_transform',
    )
    init_forbid_extra: bool
    init_typed: bool
    warn_required_dynamic_aliases: bool
    debug_dataclass_transform: bool  # undocumented

    def __init__(self, options: Options) -> None:
        if options.config_file is None:  # pragma: no cover
            return

        toml_config: Optional[Dict[str, Any]] = parse_toml(options.config_file)
        if toml_config is not None:
            config: Dict[str, Any] = toml_config.get('tool', {}).get('pydantic-mypy', {})
            for key in self.__slots__:
                setting: bool = config.get(key, False)
                if not isinstance(setting, bool):
                    raise ValueError(f'Configuration value must be a boolean for key: {key}')
                setattr(self, key, setting)
        else:
            plugin_config: ConfigParser = ConfigParser()
            plugin_config.read(options.config_file)
            for key in self.__slots__:
                setting: bool = plugin_config.getboolean(CONFIGFILE_KEY, key, fallback=False)
                setattr(self, key, setting)

    def to_data(self) -> Dict[str, Any]:
        """Returns a dict of config names to their values."""
        return {key: getattr(self, key) for key in self.__slots__}


def from_attributes_callback(ctx: MethodContext) -> Type:
    """Raise an error if from_attributes is not enabled."""
    model_type: Instance
    ctx_type: Type = ctx.type
    if isinstance(ctx_type, TypeType):
        ctx_type = ctx_type.item
    if isinstance(ctx_type, CallableType) and isinstance(ctx_type.ret_type, Instance):
        model_type = ctx_type.ret_type  # called on the class
    elif isinstance(ctx_type, Instance):
        model_type = ctx_type  # called on an instance (unusual, but still valid)
    else:  # pragma: no cover
        detail: str = f'ctx.type: {ctx_type} (of type {ctx_type.__class__.__name__})'
        error_unexpected_behavior(detail, ctx.api, ctx.context)
        return ctx.default_return_type
    pydantic_metadata: Optional[Dict[str, Any]] = model_type.type.metadata.get(METADATA_KEY)
    if pydantic_metadata is None:
        return ctx.default_return_type
    if not model_type.type.has_base(BASEMODEL_FULLNAME):
        # not a Pydantic v2 model
        return ctx.default_return_type
    from_attributes: Optional[bool] = pydantic_metadata.get('config', {}).get('from_attributes')
    if from_attributes is not True:
        error_from_attributes(model_type.type.name, ctx.api, ctx.context)
    return ctx.default_return_type


class PydanticModelField:
    """Based on mypy.plugins.dataclasses.DataclassAttribute."""

    def __init__(
        self,
        name: str,
        alias: Optional[str],
        is_frozen: bool,
        has_dynamic_alias: bool,
        has_default: bool,
        strict: Optional[bool],
        line: int,
        column: int,
        type: Optional[Type],
        info: TypeInfo,
    ):
        self.name: str = name
        self.alias: Optional[str] = alias
        self.is_frozen: bool = is_frozen
        self.has_dynamic_alias: bool = has_dynamic_alias
        self.has_default: bool = has_default
        self.strict: Optional[bool] = strict
        self.line: int = line
        self.column: int = column
        self.type: Optional[Type] = type
        self.info: TypeInfo = info

    def to_argument(
        self,
        current_info: TypeInfo,
        typed: bool,
        model_strict: bool,
        force_optional: bool,
        use_alias: bool,
        api: SemanticAnalyzerPluginInterface,
        force_typevars_invariant: bool,
        is_root_model_root: bool,
    ) -> Argument:
        """Based on mypy.plugins.dataclasses.DataclassAttribute.to_argument."""
        variable: Var = self.to_var(current_info, api, use_alias, force_typevars_invariant)

        strict: bool = model_strict if self.strict is None else self.strict
        if typed or strict:
            type_annotation: Type = self.expand_type(current_info, api)
        else:
            type_annotation: Type = AnyType(TypeOfAny.explicit)

        return Argument(
            variable=variable,
            type_annotation=type_annotation,
            initializer=None,
            kind=ARG_OPT
            if is_root_model_root
            else (ARG_NAMED_OPT if force_optional or self.has_default else ARG_NAMED),
        )

    def expand_type(
        self, current_info: TypeInfo, api: SemanticAnalyzerPluginInterface, force_typevars_invariant: bool = False
    ) -> Optional[Type]:
        """Based on mypy.plugins.dataclasses.DataclassAttribute.expand_type."""
        if force_typevars_invariant:
            # In some cases, mypy will emit an error "Cannot use a covariant type variable as a parameter"
            # To prevent that, we add an option to replace typevars with invariant ones while building certain
            # method signatures (in particular, `__init__`). There may be a better way to do this, if this causes
            # us problems in the future, we should look into why the dataclasses plugin doesn't have this issue.
            if isinstance(self.type, TypeVarType):
                modified_type: TypeVarType = self.type.copy_modified()
                modified_type.variance = INVARIANT
                self.type = modified_type

        if self.type is not None and self.info.self_type is not None:
            # In general, it is not safe to call `expand_type()` during semantic analysis,
            # however this plugin is called very late, so all types should be fully ready.
            # Also, it is tricky to avoid eager expansion of Self types here (e.g. because
            # we serialize attributes).
            with state.strict_optional_set(api.options.strict_optional):
                filled_with_typevars: Type = fill_typevars(current_info)
                # Cannot be TupleType as current_info represents a Pydantic model:
                assert isinstance(filled_with_typevars, Instance)
                if force_typevars_invariant:
                    for arg in filled_with_typevars.args:
                        if isinstance(arg, TypeVarType):
                            arg.variance = INVARIANT

                expanded_type: Type = expand_type(self.type, {self.info.self_type.id: filled_with_typevars})
                if isinstance(expanded_type, Instance) and is_root_model(expanded_type.type):
                    # When a root model is used as a field, Pydantic allows both an instance of the root model
                    # as well as instances of the `root` field type:
                    root_type: Type = cast(Type, expanded_type.type['root'].type)
                    expanded_root_type: Type = expand_type_by_instance(root_type, expanded_type)
                    expanded_type = UnionType([expanded_type, expanded_root_type])
                return expanded_type
        return self.type

    def to_var(
        self,
        current_info: TypeInfo,
        api: SemanticAnalyzerPluginInterface,
        use_alias: bool,
        force_typevars_invariant: bool = False,
    ) -> Var:
        """Based on mypy.plugins.dataclasses.DataclassAttribute.to_var."""
        if use_alias and self.alias is not None:
            name: str = self.alias
        else:
            name: str = self.name

        return Var(name, self.expand_type(current_info, api, force_typevars_invariant))

    def serialize(self) -> JsonDict:
        """Based on mypy.plugins.dataclasses.DataclassAttribute.serialize."""
        assert self.type
        return {
            'name': self.name,
            'alias': self.alias,
            'is_frozen': self.is_frozen,
            'has_dynamic_alias': self.has_dynamic_alias,
            'has_default': self.has_default,
            'strict': self.strict,
            'line': self.line,
            'column': self.column,
            'type': self.type.serialize(),
        }

    @classmethod
    def deserialize(cls, info: TypeInfo, data: JsonDict, api: SemanticAnalyzerPluginInterface) -> PydanticModelField:
        """Based on mypy.plugins.dataclasses.DataclassAttribute.deserialize."""
        data = data.copy()
        typ: Type = deserialize_and_fixup_type(data.pop('type'), api)
        return cls(type=typ, info=info, **data)

    def expand_typevar_from_subtype(self, sub_type: TypeInfo, api: SemanticAnalyzerPluginInterface) -> None:
        """Expands type vars in the context of a subtype when an attribute is inherited
        from a generic super type.
        """
        if self.type is not None:
            with state.strict_optional_set(api.options.strict_optional):
                self.type = map_type_from_supertype(self.type, sub_type, self.info)


class PydanticModelClassVar:
    """Based on mypy.plugins.dataclasses.DataclassAttribute.

    ClassVars are ignored by subclasses.

    Attributes:
        name: the ClassVar name
    """

    def __init__(self, name: str):
        self.name: str = name

    @classmethod
    def deserialize(cls, data: JsonDict) -> PydanticModelClassVar:
        """Based on mypy.plugins.dataclasses.DataclassAttribute.deserialize."""
        data = data.copy()
        return cls(**data)

    def serialize(self) -> JsonDict:
        """Based on mypy.plugins.dataclasses.DataclassAttribute.serialize."""
        return {
            'name': self.name,
        }


class PydanticModelTransformer:
    """Transform the BaseModel subclass according to the plugin settings.

    Attributes:
        tracked_config_fields: A set of field configs that the plugin has to track their value.
    """

    tracked_config_fields: Set[str] = {
        'extra',
        'frozen',
        'from_attributes',
        'populate_by_name',
        'alias_generator',
        'strict',
    }

    def __init__(
        self,
        cls: ClassDef,
        reason: Union[Expression, Statement],
        api: SemanticAnalyzerPluginInterface,
        plugin_config: PydanticPluginConfig,
    ) -> None:
        self._cls: ClassDef = cls
        self._reason: Union[Expression, Statement] = reason
        self._api: SemanticAnalyzerPluginInterface = api

        self.plugin_config: PydanticPluginConfig = plugin_config

    def transform(self) -> bool:
        """Configures the BaseModel subclass according to the plugin settings.

        In particular:

        * determines the model config and fields,
        * adds a fields-aware signature for the initializer and construct methods
        * freezes the class if frozen = True
        * stores the fields, config, and if the class is settings in the mypy metadata for access by subclasses
        """
        info: TypeInfo = self._cls.info
        is_a_root_model: bool = is_root_model(info)
        config: ModelConfigData = self.collect_config()
        fields: Optional[List[PydanticModelField]]
        class_vars: Optional[List[PydanticModelClassVar]]
        fields, class_vars = self.collect_fields_and_class_vars(config, is_a_root_model)
        if fields is None or class_vars is None:
            # Some definitions are not ready. We need another pass.
            return False
        for field in fields:
            if field.type is None:
                return False

        is_settings: bool = info.has_base(BASESETTINGS_FULLNAME)
        self.add_initializer(fields, config, is_settings, is_a_root_model)
        self.add_model_construct_method(fields, config, is_settings, is_a_root_model)
        self.set_frozen(fields, self._api, frozen=config.frozen is True)

        self.adjust_decorator_signatures()

        info.metadata[METADATA_KEY] = {
            'fields': {field.name: field.serialize() for field in fields},
            'class_vars': {class_var.name: class_var.serialize() for class_var in class_vars},
            'config': config.get_values_dict(),
        }

        return True

    def adjust_decorator_signatures(self) -> None:
        """When we decorate a function `f` with `pydantic.validator(...)`, `pydantic.field_validator`
        or `pydantic.serializer(...)`, mypy sees `f` as a regular method taking a `self` instance,
        even though pydantic internally wraps `f` with `classmethod` if necessary.

        Teach mypy this by marking any function whose outermost decorator is a `validator()`,
        `field_validator()` or `serializer()` call as a `classmethod`.
        """
        for sym in self._cls.info.names.values():
            if isinstance(sym.node, Decorator):
                first_dec: CallExpr = sym.node.original_decorators[0]
                if (
                    isinstance(first_dec, CallExpr)
                    and isinstance(first_dec.callee, NameExpr)
                    and first_dec.callee.fullname in IMPLICIT_CLASSMETHOD_DECORATOR_FULLNAMES
                    # @model_validator(mode="after") is an exception, it expects a regular method
                    and not (
                        first_dec.callee.fullname == MODEL_VALIDATOR_FULLNAME
                        and any(
                            first_dec.arg_names[i] == 'mode' and isinstance(arg, StrExpr) and arg.value == 'after'
                            for i, arg in enumerate(first_dec.args)
                    )
                ):
                    # TODO: Only do this if the first argument of the decorated function is `cls`
                    sym.node.func.is_class = True

    def collect_config(self) -> ModelConfigData:  # noqa: C901 (ignore complexity)
        """Collects the values of the config attributes that are used by the plugin, accounting for parent classes."""
        cls: ClassDef = self._cls
        config: ModelConfigData = ModelConfigData()

        has_config_kwargs: bool = False
        has_config_from_namespace: bool = False

        # Handle `class MyModel(BaseModel, <name>=<expr>, ...):`
        for name, expr in cls.keywords.items():
            config_data: Optional[ModelConfigData] = self.get_config_update(name, expr)
            if config_data:
                has_config_kwargs = True
                config.update(config_data)

        # Handle `model_config`
        stmt: Optional[Statement] = None
        for stmt in cls.defs.body:
            if not isinstance(stmt, (AssignmentStmt, ClassDef)):
                continue

            if isinstance(stmt, AssignmentStmt):
                lhs: NameExpr = stmt.lvalues[0]
                if not isinstance(lhs, NameExpr) or lhs.name != 'model_config':
                    continue

                if isinstance(stmt.rvalue, CallExpr):  # calls to `dict` or `ConfigDict`
                    for arg_name, arg in zip(stmt.rvalue.arg_names, stmt.rvalue.args):
                        if arg_name is None:
                            continue
                        config.update(self.get_config_update(arg_name, arg, lax_extra=True))
                elif isinstance(stmt.rvalue, DictExpr):  # dict literals
                    for key_expr, value_expr in stmt.rvalue.items:
                        if not isinstance(key_expr, StrExpr):
                            continue
                        config.update(self.get_config_update(key_expr.value, value_expr))

            elif isinstance(stmt, ClassDef):
                if stmt.name != 'Config':  # 'deprecated' Config-class
                    continue
                for substmt in stmt.defs.body:
                    if not isinstance(substmt, AssignmentStmt):
                        continue
                    lhs = substmt.lvalues[0]
                    if not isinstance(lhs, NameExpr):
                        continue
                    config.update(self.get_config_update(lhs.name, substmt.rvalue))

            if has_config_kwargs:
                self._api.fail(
                    'Specifying config in two places is ambiguous, use either Config attribute or class kwargs',
                    cls,
                )
                break

            has_config_from_namespace = True

        if has_config_kwargs or has_config_from_namespace:
            if (
                stmt
                and config.has_alias_generator
                and not config.populate_by_name
                and self.plugin_config.warn_required_dynamic_aliases
            ):
                error_required_dynamic_aliases(self._api, stmt)

        for info in cls.info.mro[1:]:  # 0 is the current class
            if METADATA_KEY not in info.metadata:
                continue

            # Each class depends on the set of fields in its ancestors
            self._api.add_plugin_dependency(make_wildcard_trigger(info.fullname))
            for name, value in info.metadata[METADATA_KEY]['config'].items():
                config.setdefault(name, value)
        return config

    def collect_fields_and_class_vars(
        self, model_config: ModelConfigData, is_root_model: bool
    ) -> Tuple[Optional[List[PydanticModelField]], Optional[List[PydanticModelClassVar]]]:
        """Collects the fields for the model, accounting for parent classes."""
        cls: ClassDef = self._cls

        # First, collect fields and ClassVars belonging to any class in the MRO, ignoring duplicates.
        #
        # We iterate through the MRO in reverse because attrs defined in the parent must appear
        # earlier in the attributes list than attrs defined in the child. See:
        # https://docs.python.org/3/library/dataclasses.html#inheritance
        #
        # However, we also want fields defined in the subtype to override ones defined
        # in the parent. We can implement this via a dict without disrupting the attr order
        # because dicts preserve insertion order in Python 3.7+.
        found_fields: Dict[str, PydanticModelField] = {}
        found_class_vars: Dict[str, PydanticModelClassVar] = {}
        for info in reversed(cls.info.mro[1:-1]):  # 0 is the current class, -2 is BaseModel, -1 is object
            # if BASEMODEL_METADATA_TAG_KEY in info.metadata and BASEMODEL_METADATA_KEY not in info.metadata:
            #     # We haven't processed the base class yet. Need another pass.
            #     return None, None
            if METADATA_KEY not in info.metadata:
                continue

            # Each class depends on the set of attributes in its dataclass ancestors.
            self._api.add_plugin_dependency(make_wildcard_trigger(info.fullname))

            for name, data in info.metadata[METADATA_KEY]['fields'].items():
                field: PydanticModelField = PydanticModelField.deserialize(info, data, self._api)
                # (The following comment comes directly from the dataclasses plugin)
                # TODO: We shouldn't be performing type operations during the main
                #       semantic analysis pass, since some TypeInfo attributes might
                #       still be in flux. This should be performed in a later phase.
                field.expand_typevar_from_subtype(cls.info, self._api)
                found_fields[name] = field

                sym_node: Optional[SymbolTableNode] = cls.info.names.get(name)
                if sym_node and sym_node.node and not isinstance(sym_node.node, Var):
                    self._api.fail(
                        'BaseModel field may only be overridden by another field',
                        sym_node.node,
                    )
            # Collect ClassVars
            for name, data in info.metadata[METADATA_KEY]['class_vars'].items():
                found_class_vars[name] = PydanticModelClassVar.deserialize(data)

        # Second, collect fields and ClassVars belonging to the current class.
        current_field_names: Set[str] = set()
        current_class_vars_names: Set[str] = set()
        for stmt in self._get_assignment_statements_from_block(cls.defs):
            maybe_field: Optional[Union[PydanticModelField, PydanticModelClassVar]] = self.collect_field_or_class_var_from_stmt(stmt, model_config, found_class_vars)
            if maybe_field is None:
                continue

            lhs: NameExpr = stmt.lvalues[0]
            assert isinstance(lhs, NameExpr)  # collect_field_or_class_var_from_stmt guarantees this
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

    def _get_assignment_statements_from_if_statement(self, stmt: IfStmt) -> Iterator[AssignmentStmt]:
        for body in stmt.body:
            if not body.is_unreachable:
                yield from self._get_assignment_statements_from_block(body)
        if stmt.else_body is not None and not stmt.else_body.is_unreachable:
            yield from self._get_assignment_statements_from_block(stmt.else_body)

    def _get_assignment_statements_from_block(self, block: Block) -> Iterator[AssignmentStmt]:
        for stmt in block.body:
            if isinstance(stmt, AssignmentStmt):
                yield stmt
            elif isinstance(stmt, IfStmt):
                yield from self._get_assignment_statements_from_if_statement(stmt)

    def collect_field_or_class_var_from_stmt(  # noqa C901
        self, stmt: AssignmentStmt, model_config: ModelConfigData, class_vars: Dict[str, PydanticModelClassVar]
    ) -> Optional[Union[PydanticModelField, PydanticModelClassVar]]:
        """Get pydantic model field from statement.

        Args:
            stmt: The statement.
            model_config: Configuration settings for the model.
            class_vars: ClassVars already known to be defined on the model.

        Returns:
            A pydantic model field if it could find the field in statement. Otherwise, `None`.
        """
        cls: ClassDef = self._cls

        lhs: NameExpr = stmt.lvalues[0]
        if not isinstance(lhs, NameExpr) or not _fields.is_valid_field_name(lhs.name) or lhs.name == 'model_config':
            return None

        if not stmt.new_syntax:
            if (
                isinstance(stmt.rvalue, CallExpr)
                and isinstance(stmt.rvalue.callee, CallExpr)
                and isinstance(stmt.rvalue.callee.callee, NameExpr)
                and stmt.rvalue.callee.callee.fullname in DECORATOR_FULLNAMES
            ):
                # This is a (possibly-reused) validator or serializer, not a field
                # In particular, it looks something like: my_validator = validator('my_field')(f)
                # Eventually, we may want to attempt to respect model_config['ignored_types']
                return None

            if lhs.name in class_vars:
                # Class vars are not fields and are not required to be annotated
                return None

            # The assignment does not have an annotation, and it's not anything else we recognize
            error_untyped_fields(self._api, stmt)
            return None

        lhs = stmt.lvalues[0]
        if not isinstance(lhs, NameExpr):
            return None

        if not _fields.is_valid_field_name(lhs.name) or lhs.name == 'model_config':
            return None

        sym: Optional[SymbolTableNode] = cls.info.names.get(lhs.name)
        if sym is None:  # pragma: no cover
            # This is likely due to a star import (see the dataclasses plugin for a more detailed explanation)
            # This is the same logic used in the dataclasses plugin
            return None

        node: Optional[SymbolNode] = sym.node
        if isinstance(node, PlaceholderNode):  # pragma: no cover
            # See the PlaceholderNode docstring for more detail about how this can occur
            # Basically, it is an edge case when dealing with complex import logic

            # The dataclasses plugin now asserts this cannot happen, but I'd rather not error if it does..
            return None

        if isinstance(node, TypeAlias):
            self._api.fail(
                'Type aliases inside BaseModel definitions are not supported at runtime',
                node,
            )
            # Skip processing this node. This doesn't match the runtime behaviour,
            # but the only alternative would be to modify the SymbolTable,
            # and it's a little hairy to do that in a plugin.
            return None

        if not isinstance(node, Var):  # pragma: no cover
            # Don't know if this edge case still happens with the `is_valid_field` check above
            # but better safe than sorry

            # The dataclasses plugin now asserts this cannot happen, but I'd rather not error if it does..
            return None

        # x: ClassVar[int] is not a field
        if node.is_classvar:
            return PydanticModelClassVar(lhs.name)

        # x: InitVar[int] is not supported in BaseModel
        node_type: Type = get_proper_type(node.type)
        if isinstance(node_type, Instance) and node_type.type.fullname == 'dataclasses.InitVar':
            self._api.fail(
                'InitVar is not supported in BaseModel',
                node,
            )

        has_default: bool = self.get_has_default(stmt)
        strict: Optional[bool] = self.get_strict(stmt)

        if sym.type is None and node.is_final and node.is_inferred:
            # This follows the logic from the dataclasses plugin. The following comment is taken verbatim:
            #
            # This is a special case, assignment like x: Final = 42 is classified
            # annotated above, but mypy strips the `Final` turning it into x = 42.
            # We do not support inferred types in dataclasses, so we can try inferring
            # type for simple literals, and otherwise require an explicit type
            # argument for Final[...].
            typ: Optional[Type] = self._api.analyze_simple_literal_type(stmt.rvalue, is_final=True)
            if typ:
                node.type = typ
            else:
                self._api.fail(
                    'Need type argument for Final[...] with non-literal default in BaseModel',
                    stmt,
                )
                node.type = AnyType(TypeOfAny.from_error)

        if node.is_final and has_default:
            # TODO this path should be removed (see https://github.com/pydantic/pydantic/issues/11119)
            return PydanticModelClassVar(lhs.name)

        alias: Optional[str]
        has_dynamic_alias: bool
        alias, has_dynamic_alias = self.get_alias_info(stmt)
        if has_dynamic_alias and not model_config.populate_by_name and self.plugin_config.warn_required_dynamic_aliases:
            error_required_dynamic_aliases(self._api, stmt)
        is_frozen: bool = self.is_field_frozen(stmt)

        init_type: Optional[Type] = self._infer_dataclass_attr_init_type(sym, lhs.name, stmt)
        return PydanticModelField(
            name=lhs.name,
            has_dynamic_alias=has_dynamic_alias,
            has_default=has_default,
            strict=strict,
            alias=alias,
            is_frozen=is_frozen,
            line=stmt.line,
            column=stmt.column,
            type=init_type,
            info=cls.info,
        )

    def _infer_dataclass_attr_init_type(self, sym: SymbolTableNode, name: str, context: Context) -> Optional[Type]:
        """Infer __init__ argument type for an attribute.

        In particular, possibly use the signature of __set__.
        """
        default: Optional[Type] = sym.type
       