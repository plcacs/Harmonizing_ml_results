from typing import Any, Dict, List, Optional, Tuple, Type, Union, Callable, Iterable, Mapping, Protocol, TypeVar
from abc import ABCMeta, abstractmethod
import os
from copy import deepcopy
from dbt.adapters.base.column import Column
from dbt.adapters.base.relation import EventTimeFilter
from dbt.adapters.contracts.connection import AdapterResponse
from dbt.adapters.exceptions import MissingConfigError
from dbt.adapters.factory import get_adapter, get_adapter_package_names, get_adapter_type_names
from dbt.artifacts.resources import NodeConfig, NodeVersion, RefArgs, SeedConfig, SourceConfig
from dbt.clients.jinja import MacroGenerator, MacroStack, UnitTestMacroGenerator, get_rendered
from dbt.clients.jinja_static import statically_parse_unrendered_config
from dbt.config import IsFQNResource, Project, RuntimeConfig
from dbt.constants import DEFAULT_ENV_PLACEHOLDER
from dbt.context.base import Var, contextmember, contextproperty
from dbt.context.configured import FQNLookup
from dbt.context.context_config import ContextConfig
from dbt.context.exceptions_jinja import wrapped_exports
from dbt.context.macro_resolver import MacroResolver, TestMacroNamespace
from dbt.context.macros import MacroNamespace, MacroNamespaceBuilder
from dbt.context.manifest import ManifestContext
from dbt.contracts.graph.manifest import Disabled, Manifest
from dbt.contracts.graph.metrics import MetricReference, ResolvedMetricReference
from dbt.contracts.graph.nodes import AccessType, Exposure, Macro, ManifestNode, ModelNode, Resource, SeedNode, SemanticModel, SourceDefinition, UnitTestNode
from dbt.exceptions import CompilationError, ConflictingConfigKeysError, DbtReferenceError, EnvVarMissingError, InlineModelConfigError, LoadAgateTableNotSeedError, LoadAgateTableValueError, MacroDispatchArgError, MacroResultAlreadyLoadedError, MetricArgsError, NumberSourceArgsError, OperationsCannotRefEphemeralNodesError, ParsingError, PersistDocsValueTypeError, RefArgsError, RefBadContextError, SecretEnvVarLocationError, TargetNotFoundError
from dbt.flags import get_flags
from dbt.materializations.incremental.microbatch import MicrobatchBuilder
from dbt.node_types import ModelLanguage, NodeType
from dbt.utils import MultiDict, args_to_dict
from dbt_common.clients.jinja import MacroProtocol
from dbt_common.constants import SECRET_ENV_PREFIX
from dbt_common.context import get_invocation_context
from dbt_common.events.functions import get_metadata_vars
from dbt_common.exceptions import DbtInternalError, DbtRuntimeError, DbtValidationError, MacrosSourcesUnWriteableError
from dbt_common.utils import AttrDict, cast_to_str, merge

if TYPE_CHECKING:
    import agate

_MISSING = object()

class RelationProxy:
    def __init__(self, adapter: Any) -> None:
        self._quoting_config = adapter.config.quoting
        self._relation_type = adapter.Relation

    def __getattr__(self, key: str) -> Any:
        return getattr(self._relation_type, key)

    def create(self, *args: Any, **kwargs: Any) -> Any:
        kwargs["quote_policy"] = merge(self._quoting_config, kwargs.pop("quote_policy", {}))
        return self._relation_type.create(*args, **kwargs)

class BaseDatabaseWrapper:
    def __init__(self, adapter: Any, namespace: MacroNamespace) -> None:
        self._adapter = adapter
        self.Relation = RelationProxy(adapter)
        self._namespace = namespace

    def __getattr__(self, name: str) -> Any:
        raise NotImplementedError("subclasses need to implement this")

    @property
    def config(self) -> Any:
        return self._adapter.config

    def type(self) -> str:
        return self._adapter.type()

    def commit(self) -> None:
        return self._adapter.commit_if_has_connection()

    def _get_adapter_macro_prefixes(self) -> List[str]:
        search_prefixes = get_adapter_type_names(self._adapter.type()) + ["default"]
        return search_prefixes

    def _get_search_packages(self, namespace: Optional[str] = None) -> List[Optional[str]]:
        search_packages: List[Optional[str]] = [None]
        if namespace is None:
            search_packages = [None]
        elif isinstance(namespace, str):
            macro_search_order = self._adapter.config.get_macro_search_order(namespace)
            if macro_search_order:
                search_packages = macro_search_order
            elif not macro_search_order and namespace in self._adapter.config.dependencies:
                search_packages = [self.config.project_name, namespace]
        else:
            raise CompilationError(
                f"In adapter.dispatch, got a {type(namespace)} macro_namespace argument "
                f'("{namespace}"), but macro_namespace should be None or a string.'
            )
        return search_packages

    def dispatch(
        self,
        macro_name: str,
        macro_namespace: Optional[str] = None,
        packages: Optional[List[str]] = None,
    ) -> MacroGenerator:
        search_packages: List[Optional[str]]
        if "." in macro_name:
            suggest_macro_namespace, suggest_macro_name = macro_name.split(".", 1)
            msg = (
                f'In adapter.dispatch, got a macro name of "{macro_name}", '
                f'but "." is not a valid macro name component. Did you mean '
                f'`adapter.dispatch("{suggest_macro_name}", '
                f'macro_namespace="{suggest_macro_namespace}")`?'
            )
            raise CompilationError(msg)
        if packages is not None:
            raise MacroDispatchArgError(macro_name)
        search_packages = self._get_search_packages(macro_namespace)
        attempts = []
        for package_name in search_packages:
            for prefix in self._get_adapter_macro_prefixes():
                search_name = f"{prefix}__{macro_name}"
                try:
                    macro = self._namespace.get_from_package(package_name, search_name)
                except CompilationError:
                    macro = None
                if package_name is None:
                    attempts.append(search_name)
                else:
                    attempts.append(f"{package_name}.{search_name}")
                if macro is not None:
                    return macro
        searched = ", ".join(repr(a) for a in attempts)
        msg = f"In dispatch: No macro named '{macro_name}' found within namespace: '{macro_namespace}'\n    Searched for: {searched}"
        raise CompilationError(msg)

class BaseResolver(metaclass=ABCMeta):
    def __init__(self, db_wrapper: BaseDatabaseWrapper, model: Any, config: RuntimeConfig, manifest: Manifest) -> None:
        self.db_wrapper = db_wrapper
        self.model = model
        self.config = config
        self.manifest = manifest

    @property
    def current_project(self) -> str:
        return self.config.project_name

    @property
    def Relation(self) -> RelationProxy:
        return self.db_wrapper.Relation

    @property
    def resolve_limit(self) -> Optional[int]:
        return 0 if getattr(self.config.args, "EMPTY", False) else None

    def resolve_event_time_filter(self, target: ManifestNode) -> Optional[EventTimeFilter]:
        event_time_filter = None
        sample_mode = bool(
            os.environ.get("DBT_EXPERIMENTAL_SAMPLE_MODE")
            and getattr(self.config.args, "sample", None)
        )
        if (
            (
                isinstance(target.config, NodeConfig)
                or isinstance(target.config, SourceConfig)
                or isinstance(target.config, SeedConfig)
            )
            and target.config.event_time
            and isinstance(self.model, ModelNode)
        ):
            if (
                self.model.config.materialized == "incremental"
                and self.model.config.incremental_strategy == "microbatch"
                and self.manifest.use_microbatch_batches(project_name=self.config.project_name)
                and self.model.batch is not None
            ):
                if sample_mode:
                    start = (
                        self.config.args.sample.start
                        if self.config.args.sample.start > self.model.batch.event_time_start
                        else self.model.batch.event_time_start
                    )
                    end = (
                        self.config.args.sample.end
                        if self.config.args.sample.end < self.model.batch.event_time_end
                        else self.model.batch.event_time_end
                    )
                    event_time_filter = EventTimeFilter(
                        field_name=target.config.event_time,
                        start=start,
                        end=end,
                    )
                else:
                    event_time_filter = EventTimeFilter(
                        field_name=target.config.event_time,
                        start=self.model.batch.event_time_start,
                        end=self.model.batch.event_time_end,
                    )
            elif sample_mode:
                event_time_filter = EventTimeFilter(
                    field_name=target.config.event_time,
                    start=self.config.args.sample.start,
                    end=self.config.args.sample.end,
                )
        return event_time_filter

    @abstractmethod
    def __call__(self, *args: str) -> Union[str, RelationProxy, MetricReference]:
        pass

class BaseRefResolver(BaseResolver):
    @abstractmethod
    def resolve(
        self, name: str, package: Optional[str] = None, version: Optional[NodeVersion] = None
    ) -> RelationProxy: ...

    def _repack_args(
        self, name: str, package: Optional[str], version: Optional[NodeVersion]
    ) -> RefArgs:
        return RefArgs(package=package, name=name, version=version)

    def validate_args(self, name: str, package: Optional[str], version: Optional[NodeVersion]):
        if not isinstance(name, str):
            raise CompilationError(
                f"The name argument to ref() must be a string, got {type(name)}"
            )
        if package is not None and not isinstance(package, str):
            raise CompilationError(
                f"The package argument to ref() must be a string or None, got {type(package)}"
            )
        if version is not None and not isinstance(version, (str, int, float)):
            raise CompilationError(
                f"The version argument to ref() must be a string, int, float, or None - got {type(version)}"
            )

    def __call__(self, *args: str, **kwargs) -> RelationProxy:
        name: str
        package: Optional[str] = None
        version: Optional[NodeVersion] = None
        if len(args) == 1:
            name = args[0]
        elif len(args) == 2:
            package, name = args
        else:
            raise RefArgsError(node=self.model, args=args)
        version = kwargs.get("version") or kwargs.get("v")
        self.validate_args(name, package, version)
        return self.resolve(name, package, version)

class BaseSourceResolver(BaseResolver):
    @abstractmethod
    def resolve(self, source_name: str, table_name: str) -> Any:
        pass

    def validate_args(self, source_name: str, table_name: str):
        if not isinstance(source_name, str):
            raise CompilationError(
                f"The source name (first) argument to source() must be a "
                f"string, got {type(source_name)}"
            )
        if not isinstance(table_name, str):
            raise CompilationError(
                f"The table name (second) argument to source() must be a "
                f"string, got {type(table_name)}"
            )

    def __call__(self, *args: str) -> RelationProxy:
        if len(args) != 2:
            raise NumberSourceArgsError(args, node=self.model)
        self.validate_args(args[0], args[1])
        return self.resolve(args[0], args[1])

class BaseMetricResolver(BaseResolver):
    @abstractmethod
    def resolve(self, name: str, package: Optional[str] = None) -> MetricReference: ...

    def _repack_args(self, name: str, package: Optional[str]) -> List[str]:
        if package is None:
            return [name]
        else:
            return [package, name]

    def validate_args(self, name: str, package: Optional[str]):
        if not isinstance(name, str):
            raise CompilationError(
                f"The name argument to metric() must be a string, got {type(name)}"
            )
        if package is not None and not isinstance(package, str):
            raise CompilationError(
                f"The package argument to metric() must be a string or None, got {type(package)}"
            )

    def __call__(self, *args: str) -> MetricReference:
        name: str
        package: Optional[str] = None
        if len(args) == 1:
            name = args[0]
        elif len(args) == 2:
            package, name = args
        else:
            raise MetricArgsError(node=self.model, args=args)
        self.validate_args(name, package)
        return self.resolve(name, package)

class Config(Protocol):
    def __init__(self, model: Any, context_config: Optional[ContextConfig]): ...

class ParseConfigObject(Config):
    def __init__(self, model: Any, context_config: Optional[ContextConfig]):
        self.model = model
        self.context_config = context_config

    def _transform_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        for oldkey in ("pre_hook", "post_hook"):
            if oldkey in config:
                newkey = oldkey.replace("_", "-")
                if newkey in config:
                    raise ConflictingConfigKeysError(oldkey, newkey, node=self.model)
                config[newkey] = config.pop(oldkey)
        return config

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        if len(args) == 1 and len(kwargs) == 0:
            opts = args[0]
        elif len(args) == 0 and len(kwargs) > 0:
            opts = kwargs
        else:
            raise InlineModelConfigError(node=self.model)
        opts = self._transform_config(opts)
        if self.context_config is None:
            raise DbtRuntimeError("At parse time, did not receive a context config")
        if get_flags().state_modified_compare_more_unrendered_values:
            unrendered_config = statically_parse_unrendered_config(self.model.raw_code)
            if unrendered_config:
                self.context_config.add_unrendered_config_call(unrendered_config)
        self.context_config.add_config_call(opts)
        return ""

    def set(self, name: str, value: Any) -> str:
        return self.__call__({name: value})

    def require(self, name: str, validator: Optional[Callable] = None) -> str:
        return ""

    def get(self, name: str, default: Optional[Any] = None, validator: Optional[Callable] = None) -> str:
        return ""

    def persist_relation_docs(self) -> bool:
        return False

    def persist_column_docs(self) -> bool:
        return False

class RuntimeConfigObject(Config):
    def __init__(self, model: Any, context_config: Optional[ContextConfig] = None):
        self.model = model

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        return ""

    def set(self, name: str, value: Any) -> str:
        return self.__call__({name: value})

    def _validate(self, validator: Callable, value: Any) -> None:
        validator(value)

    def _lookup(self, name: str, default: Any = _MISSING) -> Any:
        if not hasattr(self.model, "config"):
            result = default
        else:
            result = self.model.config.get(name, default)
        if result is _MISSING:
            raise MissingConfigError(unique_id=self.model.unique_id, name=name)
        return result

    def require(self, name: str, validator: Optional[Callable] = None) -> Any:
        to_return = self._lookup(name)
        if validator is not None:
            self._validate(validator, to_return)
        return to_return

    def get(self, name: str, default: Optional[Any] = None, validator: Optional[Callable] = None) -> Any:
        to_return = self._lookup(name, default)
        if validator is not None and default is not None:
            self._validate(validator, to_return)
        return to_return

    def persist_relation_docs(self) -> bool:
        persist_docs = self.get("persist_docs", default={})
        if not isinstance(persist_docs, dict):
            raise PersistDocsValueTypeError(persist_docs)
        return persist_docs.get("relation", False)

    def persist_column_docs(self) -> bool:
        persist_docs = self.get("persist_docs", default={})
        if not isinstance(persist_docs, dict):
            raise PersistDocsValueTypeError(persist_docs)
        return persist_docs.get("columns", False)

class ParseDatabaseWrapper(BaseDatabaseWrapper):
    def __getattr__(self, name: str) -> Any:
        override = name in self._adapter._available_ and name in self._adapter._parse_replacements_
        if override:
            return self._adapter._parse_replacements_[name]
        elif name in self._adapter._available_:
            return getattr(self._adapter, name)
        else:
            raise AttributeError(
                "'{}' object has no attribute '{}'".format(self.__class__.__name__, name)
            )

class RuntimeDatabaseWrapper(BaseDatabaseWrapper):
    def __getattr__(self, name: str) -> Any:
        if name in self._adapter._available_:
            return getattr(self._adapter, name)
        else:
            raise AttributeError(
                "'{}' object has no attribute '{}'".format(self.__class__.__name__, name)
            )

class ParseRefResolver(BaseRefResolver):
    def resolve(
        self, name: str, package: Optional[str] = None, version: Optional[NodeVersion] = None
    ) -> RelationProxy:
        self.model.refs.append(self._repack_args(name, package, version))
        return self.Relation.create_from(self.config, self.model)

ResolveRef = Union[Disabled, ManifestNode]

class RuntimeRefResolver(BaseRefResolver):
    def resolve(
        self,
        target_name: str,
        target_package: Optional[str] = None,
        target_version: Optional[NodeVersion] = None,
    ) -> RelationProxy:
        target_model = self.manifest.resolve_ref(
            self.model,
            target_name,
            target_package,
            target_version,
            self.current_project,
            self.model.package_name,
        )
        if target_model is None or isinstance(target_model, Disabled):
            raise TargetNotFoundError(
                node=self.model,
                target_name=target_name,
                target_kind="node",
                target_package=target_package,
                target_version=target_version,
                disabled=isinstance(target_model, Disabled),
            )
        elif self.manifest.is_invalid_private_ref(
            self.model, target_model, self.config.dependencies
        ):
            raise DbtReferenceError(
                unique_id=self.model.unique_id,
                ref_unique_id=target_model.unique_id,
                access=AccessType.Private,
                scope=cast_to_str(target_model.group),
            )
        elif self.manifest.is_invalid_protected_ref(
            self.model, target_model, self.config.dependencies
        ):
            raise DbtReferenceError(
                unique_id=self.model.unique_id,
                ref_unique_id=target_model.unique_id,
                access=AccessType.Protected,
                scope=target_model.package_name,
            )
        self.validate(target_model, target_name, target_package, target_version)
        return self.create_relation(target_model)

    def create_relation(self, target_model: ManifestNode) -> RelationProxy:
        if target_model.is_ephemeral_model:
            self.model.set_cte(target_model.unique_id, None)
            return self.Relation.create_ephemeral_from(
                target_model,
                limit=self.resolve_limit,
                event_time_filter=self.resolve_event_time_filter(target_model),
            )
        elif (
            hasattr(target_model, "defer_relation")
            and target_model.defer_relation
            and self.config.args.defer
            and (
                (
                    self.config.args.favor_state
                    and target_model.unique_id not in selected_resources.SELECTED_RESOURCES
                )
                or not get_adapter(self.config).get_relation(
                    target_model.database, target_model.schema, target_model.identifier
                )
            )
        ):
            return self.Relation.create_from(
                self.config,
                target_model.defer_relation,
                limit=self.resolve_limit,
                event_time_filter=self.resolve_event_time_filter(target_model),
            )
        else:
            return self.Relation.create_from(
                self.config,
                target_model,
                limit=self.resolve_limit,
                event_time_filter=self.resolve_event_time_filter(target_model),
            )

    def validate(
        self,
        resolved: ManifestNode,
        target_name: str,
        target_package: Optional[str],
        target_version: Optional[NodeVersion],
    ) -> None:
        if resolved.unique_id not in self.model.depends_on.nodes:
            args = self._repack_args(target_name, target_package, target_version)
            raise RefBadContextError(node=self.model, args=args)

class OperationRefResolver(RuntimeRefResolver):
    def validate(
        self,
        resolved: ManifestNode,
        target_name: str,
        target_package: Optional[str],
        target_version: Optional[NodeVersion],
    ) -> None:
        pass

    def create_relation(self, target_model: ManifestNode) -> RelationProxy:
        if target_model.is_ephemeral_model:
            raise OperationsCannotRefEphemeralNodesError(target_model.name, node=self.model)
        else:
            return super().create_relation(target_model)

class RuntimeUnitTestRefResolver(RuntimeRefResolver):
    @property
    def resolve_limit(self) -> Optional[int]:
        return None

    def resolve(
        self,
        target_name: str,
        target_package: Optional[str] = None,
        target_version: Optional[NodeVersion] = None,
    ) -> RelationProxy:
        return super().resolve(target_name, target_package, target_version)

class ParseSourceResolver(BaseSourceResolver):
    def resolve(self, source_name: str, table_name: str) -> RelationProxy:
        self.model.sources.append([source_name, table_name])
        return self.Relation.create_from(self.config, self.model)

class RuntimeSourceResolver(BaseSourceResolver):
    def resolve(self, source_name: str, table_name: str) -> RelationProxy:
        target_source = self.manifest.resolve_source(
            source_name,
            table_name,
            self.current_project,
            self.model.package_name,
        )
        if target_source is None or isinstance(target_source, Disabled):
            raise TargetNotFoundError(
                node=self.model,
                target_name=f"{source_name}.{table_name}",
                target_kind="source",
                disabled=(isinstance(target_source, Disabled)),
            )
        class SourceQuotingBaseConfig:
            quoting: Dict[str, Any] = {}
        return self.Relation.create_from(
            SourceQuotingBaseConfig(),
            target_source,
            limit=self.resolve_limit,
            event_time_filter=self.resolve_event_time_filter(target_source),
        )

class RuntimeUnitTestSourceResolver(BaseSourceResolver):
    @property
    def resolve_limit(self) -> Optional[int]:
        return None

    def resolve(self, source_name: str, table_name: str) -> RelationProxy:
        target_source = self.manifest.resolve_source(
            source_name,
            table_name,
            self.current_project,
            self.model.package_name,
        )
        if target_source is None or isinstance(target_source, Disabled):
            raise TargetNotFoundError(
                node=self.model,
                target_name=f"{source_name}.{table_name}",
                target_kind="source",
                disabled=(isinstance(target_source, Disabled)),
            )
        self.model.set_cte(target_source.unique_id, None)
        return self.Relation.create_ephemeral_from(target_source)

class ParseMetricResolver(BaseMetricResolver):
    def resolve(self, name: str, package: Optional[str] = None) -> MetricReference:
        self.model.metrics.append(self._repack_args(name, package))
        return MetricReference(name, package)

class RuntimeMetricResolver(BaseMetricResolver):
    def resolve(self, target_name: str, target_package: Optional[str] = None) -> MetricReference:
        target_metric = self.manifest.resolve_metric(
            target_name,
            target_package,
            self.current_project,
            self.model.package_name,
        )
        if target_metric is None or isinstance(target_metric, Disabled):
            raise TargetNotFoundError(
                node=self.model,
                target_name=target_name,
                target_kind="metric",
                target_package=target_package,
            )
        return ResolvedMetricReference(target_metric, self.manifest)

class ModelConfiguredVar(Var):
    def __init__(
        self,
        context: Dict[str, Any],
        config: RuntimeConfig,
        node: Resource,
    ) -> None:
        self._node: Resource
        self._config: RuntimeConfig = config
        super().__init__(context, config.cli_vars, node=node)

    def packages_for_node(self) -> Iterable[Project]:
        dependencies = self._config.load_dependencies()
        package_name = self._node.package_name
        if package_name != self._config.project_name:
            if package_name in dependencies:
                yield dependencies[package_name]
        yield self._config

    def _generate_merged(self) -> Mapping[str, Any]:
        search_node: IsFQNResource
        if isinstance(self._node, IsFQNResource):
            search_node = self._node
        else:
            search_node = FQNLookup(self._node.package_name)
        adapter_type = self._config.credentials.type
        merged = MultiDict()
        for project in self.packages_for_node():
            merged.add(project.vars.vars_for(search_node, adapter_type))
        merged.add(self._cli_vars)
        return merged

class ParseVar(ModelConfiguredVar):
    def get_missing_var(self, var_name: str) -> None:
        return None

class RuntimeVar(ModelConfiguredVar):
    pass

class UnitTestVar(RuntimeVar):
    def __init__(
        self,
        context: Dict[str, Any],
        config: RuntimeConfig,
        node: Resource,
    ) -> None:
        config_copy = None
        assert isinstance(node, UnitTestNode)
        if node.overrides and node.overrides.vars:
            config_copy = deepcopy(config)
            config_copy.cli_vars.update(node.overrides.vars)
        super().__init__(context, config_copy or config, node=node)

class Provider(Protocol):
    execute: bool
    Config: Type[Config]
    DatabaseWrapper: Type[BaseDatabaseWrapper]
    Var: Type[ModelConfiguredVar]
    ref: Type[BaseRefResolver]
    source: Type[BaseSourceResolver]
    metric: Type[BaseMetricResolver]

class ParseProvider(Provider):
    execute = False
    Config = ParseConfigObject
    DatabaseWrapper = ParseDatabaseWrapper
    Var = ParseVar
    ref = ParseRefResolver
    source = ParseSourceResolver
    metric = ParseMetricResolver

class GenerateNameProvider(Provider):
    execute = False
    Config = RuntimeConfigObject
    DatabaseWrapper = ParseDatabaseWrapper
    Var = RuntimeVar
    ref = ParseRefResolver
    source = ParseSourceResolver
    metric = ParseMetricResolver

class RuntimeProvider(Provider):
    execute = True
    Config = RuntimeConfigObject
    DatabaseWrapper = RuntimeDatabaseWrapper
    Var = RuntimeVar
    ref = RuntimeRefResolver
    source = RuntimeSourceResolver
    metric = RuntimeMetricResolver

class RuntimeUnitTestProvider(Provider):
    execute = True
    Config = RuntimeConfigObject
    DatabaseWrapper = RuntimeDatabaseWrapper
    Var = UnitTestVar
    ref = RuntimeUnitTestRefResolver
    source = RuntimeUnitTestSourceResolver
    metric = RuntimeMetricResolver

class OperationProvider(RuntimeProvider):
    ref = OperationRefResolver

T = TypeVar("T")

class ProviderContext(ManifestContext):
    def __init__(
        self,
        model: Any,
        config: RuntimeConfig,
        manifest: Manifest,
        provider: Provider,
        context_config: Optional[ContextConfig],
    ) -> None:
        if provider is None:
            raise DbtInternalError(f"Invalid provider given to context: {provider}")
        self.config: RuntimeConfig
        self.model: Union[Macro, ManifestNode, SourceDefinition] = model
        super().__init__(config, manifest, model.package_name)
        self.sql_results: Dict[str, Optional[AttrDict]] = {}
        self.context_config: Optional[ContextConfig] = context_config
        self.provider: Provider = provider
        self.adapter = get_adapter(self.config)
        self.db_wrapper = self.provider.DatabaseWrapper(self.adapter, self.namespace)

    def _get_namespace_builder(self) -> MacroNamespaceBuilder:
        internal_packages = get_adapter_package_names(self.config.credentials.type)
        return MacroNamespaceBuilder(
            self.config.project_name,
            self.search_package,
            self.macro_stack,
            internal_packages,
            self.model,
        )

    @contextproperty()
    def dbt_metadata_envs(self) -> Dict[str, str]:
        return get_metadata_vars()

    @contextproperty()
    def invocation_args_dict(self) -> Dict[str, Any]:
        return args_to_dict(self.config.args)

    @contextproperty()
    def _sql_results(self) -> Dict[str, Optional[AttrDict]]:
        return self.sql_results

    @contextmember()
    def load_result(self, name: str) -> Optional[AttrDict]:
        if name in self.sql_results:
            if name == "main":
                return self.sql_results["main"]
            elif self.sql_results[name] is None:
                raise MacroResultAlreadyLoadedError(name)
            else:
                ret_val = self.sql_results[name]
                self.sql_results[name] = None
                return ret_val
        else:
            return None

    @contextmember()
    def store_result(
        self, name: str, response: Any, agate_table: Optional["agate.Table"] = None
    ) -> str:
        from dbt_common.clients import agate_helper
        if agate_table is None:
            agate_table = agate_helper.empty_table()
        self.sql_results[name] = AttrDict(
            {
                "response": response,
                "data": agate_helper.as_matrix(agate_table),
                "table": agate_table,
            }
        )
        return ""

    @contextmember()
    def store_raw_result(
        self,
        name: str,
        message: Optional[str] = None,
        code: Optional[str] = None,
        rows_affected: Optional[str] = None,
        agate_table: Optional["agate.Table"] = None,
    ) -> str:
        response = AdapterResponse(_message=message, code=code, rows_affected=rows_affected)
        return self.store_result(name, response, agate_table)

    @contextproperty()
    def validation(self) -> Dict[str, Callable[[T], None]]:
        def validate_any(*args: Any) -> Callable[[T], None]:
            def inner(value: T) -> None:
                for arg in args:
                    if isinstance(arg, type) and isinstance(value, arg):
                        return
                    elif value == arg:
                        return
                raise DbtValidationError(
                    'Expected value "{}" to be one of {}'.format(value, ",".join(map(str, args)))
                )
            return inner
        return AttrDict(
            {
                "any": validate_any,
            }
        )

    @contextmember()
    def write(self, payload: str) -> str:
        if isinstance(self.model, (Macro, SourceDefinition)):
            raise MacrosSourcesUnWriteableError(node=self.model)
        split_suffix = None
        if (
            isinstance(self.model, ModelNode)
            and self.model.config.get("incremental_strategy") == "microbatch"
        ):
            split_suffix = MicrobatchBuilder.format_batch_start(
                self.model.config.get("__dbt_internal_microbatch_event_time_start"),
                self.model.config.batch_size,
            )
        self.model.build_path = self.model.get_target_write_path(
            self.config.target_path, "run", split_suffix=split_suffix
        )
        self.model.write_node(self.config.project_root, self.model.build_path, payload)
        return ""

    @contextmember()
    def render(self, string: str) -> str:
        return get_rendered(string, self._ctx, self.model)

    @contextmember()
    def try_or_compiler_error(
        self, message_if_exception: str, func: Callable, *args: Any, **kwargs: Any
    ) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception:
            raise CompilationError(message_if_exception, self.model)

    @contextmember()
    def load_agate_table(self) -> "agate.Table":
        from dbt_common.clients import agate_helper
        if not isinstance(self.model, SeedNode):
            raise LoadAgateTableNotSeedError(self.model.resource_type, node=self.model)
        package_path = (
            os.path.join(self.config.packages_install_path, self.model.package_name)
            if self.model.package_name != self.config.project_name
            else "."
        )
        path = os.path.join(self.config.project_root, package_path, self.model.original_file_path)
        if not os.path.exists(path):
            assert self.model.root_path
            path = os.path.join(self.model.root_path, self.model.original_file_path)
        column_types = self.model.config.column_types
        delimiter = self.model.config.delimiter
        try:
            table = agate_helper.from_csv(path, text_columns=column_types, delimiter=delimiter)
        except ValueError as e:
            raise LoadAgateTableValueError(e, node=self.model)
        table.original_abspath = os.path.abspath(path)  # type: ignore
        return table

    @contextproperty()
    def ref(self) -> Callable:
        return self.provider.ref(self.db_wrapper, self.model, self.config, self.manifest)

    @contextproperty()
    def source(self) -> Callable:
        return self.provider.source(self.db_wrapper, self.model, self.config, self.manifest)

    @contextproperty()
    def metric(self) -> Callable:
        return self.provider.metric(self.db_wrapper, self.model, self.config, self.manifest)

    @contextproperty("config")
    def ctx_config(self) -> Config:
        return self.provider.Config(self.model, self.context_config)

    @contextproperty()
    def execute(self) -> bool:
        return self.provider.execute

    @contextproperty()
    def exceptions(self) -> Dict[str, Any]:
        return wrapped_exports(self.model)

    @contextproperty()
    def database(self) -> str:
        return self.config.credentials.database

    @contextproperty()
    def schema(self) -> str:
        return self.config.credentials.schema

    @contextproperty()
    def var(self) -> ModelConfiguredVar:
        return self.provider.Var(
            context=self._ctx,
            config=self.config,
            node=self.model,
        )

    @contextproperty("adapter")
    def ctx_adapter(self) -> BaseDatabaseWrapper:
        return self.db_wrapper

    @contextproperty()
    def api(self) -> Dict[str, Any]:
        return {
            "Relation": self.db_wrapper.Relation,
            "Column": self.adapter.Column,
        }

    @contextproperty()
    def column(self) -> Type[Column]:
        return self.adapter.Column

    @contextproperty()
    def env(self) -> Dict[str, Any]:
        return self.target

    @contextproperty()
    def graph(self) -> Dict[str, Any]:
       