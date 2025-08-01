import abc
import os
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Type, TypeVar, Union
from typing_extensions import Protocol

from dbt import selected_resources
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
from dbt.exceptions import (
    CompilationError,
    ConflictingConfigKeysError,
    DbtReferenceError,
    EnvVarMissingError,
    InlineModelConfigError,
    LoadAgateTableNotSeedError,
    LoadAgateTableValueError,
    MacroDispatchArgError,
    MacroResultAlreadyLoadedError,
    MetricArgsError,
    NumberSourceArgsError,
    OperationsCannotRefEphemeralNodesError,
    ParsingError,
    PersistDocsValueTypeError,
    RefArgsError,
    RefBadContextError,
    SecretEnvVarLocationError,
    TargetNotFoundError,
)
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

_MISSING: Any = object()


class RelationProxy:
    def __init__(self, adapter: Any) -> None:
        self._quoting_config = adapter.config.quoting
        self._relation_type = adapter.Relation

    def __getattr__(self, key: str) -> Any:
        return getattr(self._relation_type, key)

    def create(self, *args: Any, **kwargs: Any) -> Any:
        kwargs['quote_policy'] = merge(self._quoting_config, kwargs.pop('quote_policy', {}))
        return self._relation_type.create(*args, **kwargs)


class BaseDatabaseWrapper:
    """
    Wrapper for runtime database interaction. Applies the runtime quote policy
    via a relation proxy.
    """

    def __init__(self, adapter: Any, namespace: Any) -> None:
        self._adapter = adapter
        self.Relation = RelationProxy(adapter)
        self._namespace = namespace

    def __getattr__(self, name: str) -> Any:
        raise NotImplementedError('subclasses need to implement this')

    @property
    def config(self) -> Any:
        return self._adapter.config

    def type(self) -> Any:
        return self._adapter.type()

    def commit(self) -> Any:
        return self._adapter.commit_if_has_connection()

    def _get_adapter_macro_prefixes(self) -> List[str]:
        search_prefixes = get_adapter_type_names(self._adapter.type()) + ['default']
        return search_prefixes

    def _get_search_packages(self, namespace: Optional[Any] = None) -> List[Optional[str]]:
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
                f'In adapter.dispatch, got a {type(namespace)} macro_namespace argument ("{namespace}"), but macro_namespace should be None or a string.'
            )
        return search_packages

    def dispatch(self, macro_name: str, macro_namespace: Optional[str] = None, packages: Optional[Any] = None) -> Any:
        if '.' in macro_name:
            suggest_macro_namespace, suggest_macro_name = macro_name.split('.', 1)
            msg = (
                f'In adapter.dispatch, got a macro name of "{macro_name}", but "." is not a valid macro name component. '
                f'Did you mean `adapter.dispatch("{suggest_macro_name}", macro_namespace="{suggest_macro_namespace}")`?'
            )
            raise CompilationError(msg)
        if packages is not None:
            raise MacroDispatchArgError(macro_name)
        search_packages = self._get_search_packages(macro_namespace)
        attempts: List[str] = []
        for package_name in search_packages:
            for prefix in self._get_adapter_macro_prefixes():
                search_name = f'{prefix}__{macro_name}'
                try:
                    macro = self._namespace.get_from_package(package_name, search_name)
                except CompilationError:
                    macro = None
                if package_name is None:
                    attempts.append(search_name)
                else:
                    attempts.append(f'{package_name}.{search_name}')
                if macro is not None:
                    return macro
        searched = ', '.join((repr(a) for a in attempts))
        msg = f"In dispatch: No macro named '{macro_name}' found within namespace: '{macro_namespace}'\n    Searched for: {searched}"
        raise CompilationError(msg)


class BaseResolver(metaclass=abc.ABCMeta):
    def __init__(self, db_wrapper: BaseDatabaseWrapper, model: Any, config: Any, manifest: Manifest) -> None:
        self.db_wrapper = db_wrapper
        self.model = model
        self.config = config
        self.manifest = manifest

    @property
    def current_project(self) -> str:
        return self.config.project_name

    @property
    def Relation(self) -> Any:
        return self.db_wrapper.Relation

    @property
    def resolve_limit(self) -> Optional[int]:
        return 0 if getattr(self.config.args, 'EMPTY', False) else None

    def resolve_event_time_filter(self, target: Any) -> Optional[EventTimeFilter]:
        event_time_filter: Optional[EventTimeFilter] = None
        sample_mode = bool(os.environ.get('DBT_EXPERIMENTAL_SAMPLE_MODE') and getattr(self.config.args, 'sample', None))
        if (
            isinstance(target.config, NodeConfig)
            or isinstance(target.config, SourceConfig)
            or isinstance(target.config, SeedConfig)
        ) and target.config.event_time and isinstance(self.model, ModelNode):
            if (
                self.model.config.materialized == 'incremental'
                and self.model.config.incremental_strategy == 'microbatch'
                and self.manifest.use_microbatch_batches(project_name=self.config.project_name)
                and (self.model.batch is not None)
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
                    event_time_filter = EventTimeFilter(field_name=target.config.event_time, start=start, end=end)
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

    @abc.abstractmethod
    def __call__(self, *args: Any) -> Any:
        pass


class BaseRefResolver(BaseResolver):
    @abc.abstractmethod
    def resolve(
        self, name: str, package: Optional[str] = None, version: Optional[Union[str, int, float]] = None
    ) -> Any:
        ...

    def _repack_args(
        self, name: str, package: Optional[str], version: Optional[Union[str, int, float]]
    ) -> RefArgs:
        return RefArgs(package=package, name=name, version=version)

    def validate_args(
        self, name: str, package: Optional[str], version: Optional[Union[str, int, float]]
    ) -> None:
        if not isinstance(name, str):
            raise CompilationError(f'The name argument to ref() must be a string, got {type(name)}')
        if package is not None and (not isinstance(package, str)):
            raise CompilationError(f'The package argument to ref() must be a string or None, got {type(package)}')
        if version is not None and (not isinstance(version, (str, int, float))):
            raise CompilationError(
                f'The version argument to ref() must be a string, int, float, or None - got {type(version)}'
            )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        package: Optional[str] = None
        version: Optional[Union[str, int, float]] = None
        if len(args) == 1:
            name = args[0]
        elif len(args) == 2:
            package, name = args
        else:
            raise RefArgsError(node=self.model, args=args)
        version = kwargs.get('version') or kwargs.get('v')
        self.validate_args(name, package, version)
        return self.resolve(name, package, version)


class BaseSourceResolver(BaseResolver):
    @abc.abstractmethod
    def resolve(self, source_name: str, table_name: str) -> Any:
        pass

    def validate_args(self, source_name: str, table_name: str) -> None:
        if not isinstance(source_name, str):
            raise CompilationError(
                f'The source name (first) argument to source() must be a string, got {type(source_name)}'
            )
        if not isinstance(table_name, str):
            raise CompilationError(
                f'The table name (second) argument to source() must be a string, got {type(table_name)}'
            )

    def __call__(self, *args: Any) -> Any:
        if len(args) != 2:
            raise NumberSourceArgsError(args, node=self.model)
        self.validate_args(args[0], args[1])
        return self.resolve(args[0], args[1])


class BaseMetricResolver(BaseResolver):
    @abc.abstractmethod
    def resolve(self, name: str, package: Optional[str] = None) -> Any:
        ...

    def _repack_args(self, name: str, package: Optional[str]) -> List[Any]:
        if package is None:
            return [name]
        else:
            return [package, name]

    def validate_args(self, name: str, package: Optional[str]) -> None:
        if not isinstance(name, str):
            raise CompilationError(f'The name argument to metric() must be a string, got {type(name)}')
        if package is not None and (not isinstance(package, str)):
            raise CompilationError(
                f'The package argument to metric() must be a string or None, got {type(package)}'
            )

    def __call__(self, *args: Any) -> Any:
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
    def __init__(self, model: Any, context_config: Any) -> None:
        ...


class ParseConfigObject(Config):
    def __init__(self, model: Any, context_config: Any) -> None:
        self.model = model
        self.context_config = context_config

    def _transform_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        for oldkey in ('pre_hook', 'post_hook'):
            if oldkey in config:
                newkey = oldkey.replace('_', '-')
                if newkey in config:
                    raise ConflictingConfigKeysError(oldkey, newkey, node=self.model)
                config[newkey] = config.pop(oldkey)
        return config

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        if len(args) == 1 and len(kwargs) == 0:
            opts: Dict[str, Any] = args[0]
        elif len(args) == 0 and len(kwargs) > 0:
            opts = kwargs
        else:
            raise InlineModelConfigError(node=self.model)
        opts = self._transform_config(opts)
        if self.context_config is None:
            raise DbtRuntimeError('At parse time, did not receive a context config')
        if get_flags().state_modified_compare_more_unrendered_values:
            unrendered_config = statically_parse_unrendered_config(self.model.raw_code)
            if unrendered_config:
                self.context_config.add_unrendered_config_call(unrendered_config)
        self.context_config.add_config_call(opts)
        return ''

    def set(self, name: str, value: Any) -> str:
        return self.__call__({name: value})

    def require(self, name: str, validator: Optional[Callable[[Any], None]] = None) -> str:
        return ''

    def get(self, name: str, default: Any = None, validator: Optional[Callable[[Any], None]] = None) -> str:
        return ''

    def persist_relation_docs(self) -> bool:
        return False

    def persist_column_docs(self) -> bool:
        return False


class RuntimeConfigObject(Config):
    def __init__(self, model: Any, context_config: Optional[Any] = None) -> None:
        self.model = model

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        return ''

    def set(self, name: str, value: Any) -> str:
        return self.__call__({name: value})

    def _validate(self, validator: Callable[[Any], None], value: Any) -> None:
        validator(value)

    def _lookup(self, name: str, default: Any = _MISSING) -> Any:
        if not hasattr(self.model, 'config'):
            result = default
        else:
            result = self.model.config.get(name, default)
        if result is _MISSING:
            raise MissingConfigError(unique_id=self.model.unique_id, name=name)
        return result

    def require(self, name: str, validator: Optional[Callable[[Any], None]] = None) -> Any:
        to_return = self._lookup(name)
        if validator is not None:
            self._validate(validator, to_return)
        return to_return

    def get(self, name: str, default: Any = None, validator: Optional[Callable[[Any], None]] = None) -> Any:
        to_return = self._lookup(name, default)
        if validator is not None and default is not None:
            self._validate(validator, to_return)
        return to_return

    def persist_relation_docs(self) -> bool:
        persist_docs = self.get('persist_docs', default={})
        if not isinstance(persist_docs, dict):
            raise PersistDocsValueTypeError(persist_docs)
        return persist_docs.get('relation', False)

    def persist_column_docs(self) -> bool:
        persist_docs = self.get('persist_docs', default={})
        if not isinstance(persist_docs, dict):
            raise PersistDocsValueTypeError(persist_docs)
        return persist_docs.get('columns', False)


class ParseDatabaseWrapper(BaseDatabaseWrapper):
    """The parser subclass of the database wrapper applies any explicit
    parse-time overrides.
    """
    def __getattr__(self, name: str) -> Any:
        override = name in self._adapter._available_ and name in self._adapter._parse_replacements_
        if override:
            return self._adapter._parse_replacements_[name]
        elif name in self._adapter._available_:
            return getattr(self._adapter, name)
        else:
            raise AttributeError("'{}' object has no attribute '{}'".format(self.__class__.__name__, name))


class RuntimeDatabaseWrapper(BaseDatabaseWrapper):
    """The runtime database wrapper exposes everything the adapter marks
    available.
    """
    def __getattr__(self, name: str) -> Any:
        if name in self._adapter._available_:
            return getattr(self._adapter, name)
        else:
            raise AttributeError("'{}' object has no attribute '{}'".format(self.__class__.__name__, name))


class ParseRefResolver(BaseRefResolver):
    def resolve(
        self, name: str, package: Optional[str] = None, version: Optional[Union[str, int, float]] = None
    ) -> Any:
        self.model.refs.append(self._repack_args(name, package, version))
        return self.Relation.create_from(self.config, self.model)


ResolveRef = Union[Disabled, ManifestNode]


class RuntimeRefResolver(BaseRefResolver):
    def resolve(
        self, target_name: str, target_package: Optional[str] = None, target_version: Optional[Union[str, int, float]] = None
    ) -> Any:
        target_model = self.manifest.resolve_ref(
            self.model, target_name, target_package, target_version, self.current_project, self.model.package_name
        )
        if target_model is None or isinstance(target_model, Disabled):
            raise TargetNotFoundError(
                node=self.model,
                target_name=target_name,
                target_kind='node',
                target_package=target_package,
                target_version=target_version,
                disabled=isinstance(target_model, Disabled),
            )
        elif self.manifest.is_invalid_private_ref(self.model, target_model, self.config.dependencies):
            raise DbtReferenceError(
                unique_id=self.model.unique_id,
                ref_unique_id=target_model.unique_id,
                access=AccessType.Private,
                scope=cast_to_str(target_model.group),
            )
        elif self.manifest.is_invalid_protected_ref(self.model, target_model, self.config.dependencies):
            raise DbtReferenceError(
                unique_id=self.model.unique_id,
                ref_unique_id=target_model.unique_id,
                access=AccessType.Protected,
                scope=target_model.package_name,
            )
        self.validate(target_model, target_name, target_package, target_version)
        return self.create_relation(target_model)

    def create_relation(self, target_model: Any) -> Any:
        if target_model.is_ephemeral_model:
            self.model.set_cte(target_model.unique_id, None)
            return self.Relation.create_ephemeral_from(
                target_model, limit=self.resolve_limit, event_time_filter=self.resolve_event_time_filter(target_model)
            )
        elif hasattr(target_model, 'defer_relation') and target_model.defer_relation and self.config.args.defer and (self.config.args.favor_state and target_model.unique_id not in selected_resources.SELECTED_RESOURCES or not get_adapter(self.config).get_relation(target_model.database, target_model.schema, target_model.identifier)):
            return self.Relation.create_from(
                self.config, target_model.defer_relation, limit=self.resolve_limit, event_time_filter=self.resolve_event_time_filter(target_model)
            )
        else:
            return self.Relation.create_from(
                self.config, target_model, limit=self.resolve_limit, event_time_filter=self.resolve_event_time_filter(target_model)
            )

    def validate(
        self,
        resolved: Any,
        target_name: str,
        target_package: Optional[str],
        target_version: Optional[Union[str, int, float]],
    ) -> None:
        if resolved.unique_id not in self.model.depends_on.nodes:
            args = self._repack_args(target_name, target_package, target_version)
            raise RefBadContextError(node=self.model, args=args)


class OperationRefResolver(RuntimeRefResolver):
    def validate(
        self,
        resolved: Any,
        target_name: str,
        target_package: Optional[str],
        target_version: Optional[Union[str, int, float]],
    ) -> None:
        pass

    def create_relation(self, target_model: Any) -> Any:
        if target_model.is_ephemeral_model:
            raise OperationsCannotRefEphemeralNodesError(target_model.name, node=self.model)
        else:
            return super().create_relation(target_model)


class RuntimeUnitTestRefResolver(RuntimeRefResolver):
    @property
    def resolve_limit(self) -> Optional[int]:
        return None

    def resolve(
        self, target_name: str, target_package: Optional[str] = None, target_version: Optional[Union[str, int, float]] = None
    ) -> Any:
        return super().resolve(target_name, target_package, target_version)


class ParseSourceResolver(BaseSourceResolver):
    def resolve(self, source_name: str, table_name: str) -> Any:
        self.model.sources.append([source_name, table_name])
        return self.Relation.create_from(self.config, self.model)


class RuntimeSourceResolver(BaseSourceResolver):
    def resolve(self, source_name: str, table_name: str) -> Any:
        target_source = self.manifest.resolve_source(source_name, table_name, self.current_project, self.model.package_name)
        if target_source is None or isinstance(target_source, Disabled):
            raise TargetNotFoundError(
                node=self.model, target_name=f'{source_name}.{table_name}', target_kind='source', disabled=isinstance(target_source, Disabled)
            )

        class SourceQuotingBaseConfig:
            quoting: Dict[str, Any] = {}

        return self.Relation.create_from(SourceQuotingBaseConfig(), target_source, limit=self.resolve_limit, event_time_filter=self.resolve_event_time_filter(target_source))


class RuntimeUnitTestSourceResolver(BaseSourceResolver):
    @property
    def resolve_limit(self) -> Optional[int]:
        return None

    def resolve(self, source_name: str, table_name: str) -> Any:
        target_source = self.manifest.resolve_source(source_name, table_name, self.current_project, self.model.package_name)
        if target_source is None or isinstance(target_source, Disabled):
            raise TargetNotFoundError(
                node=self.model, target_name=f'{source_name}.{table_name}', target_kind='source', disabled=isinstance(target_source, Disabled)
            )
        self.model.set_cte(target_source.unique_id, None)
        return self.Relation.create_ephemeral_from(target_source)


class ParseMetricResolver(BaseMetricResolver):
    def resolve(self, name: str, package: Optional[str] = None) -> MetricReference:
        self.model.metrics.append(self._repack_args(name, package))
        return MetricReference(name, package)


class RuntimeMetricResolver(BaseMetricResolver):
    def resolve(self, target_name: str, target_package: Optional[str] = None) -> ResolvedMetricReference:
        target_metric = self.manifest.resolve_metric(target_name, target_package, self.current_project, self.model.package_name)
        if target_metric is None or isinstance(target_metric, Disabled):
            raise TargetNotFoundError(
                node=self.model, target_name=target_name, target_kind='metric', target_package=target_package
            )
        return ResolvedMetricReference(target_metric, self.manifest)


class ModelConfiguredVar(Var):
    def __init__(self, context: Any, config: Any, node: Any) -> None:
        self._config = config
        super().__init__(context, config.cli_vars, node=node)

    def packages_for_node(self) -> Iterable[Any]:
        dependencies = self._config.load_dependencies()
        package_name = self._node.package_name
        if package_name != self._config.project_name:
            if package_name in dependencies:
                yield dependencies[package_name]
        yield self._config

    def _generate_merged(self) -> MultiDict:
        if isinstance(self._node, IsFQNResource):
            search_node = self._node
        else:
            search_node = FQNLookup(self._node.package_name)
        adapter_type = self._config.credentials.type
        merged: MultiDict = MultiDict()
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
    def __init__(self, context: Any, config: Any, node: UnitTestNode) -> None:
        config_copy: Optional[Any] = None
        assert isinstance(node, UnitTestNode)
        if node.overrides and node.overrides.vars:
            config_copy = deepcopy(config)
            config_copy.cli_vars.update(node.overrides.vars)
        super().__init__(context, config_copy or config, node=node)


class Provider(Protocol):
    pass


class ParseProvider(Provider):
    execute: bool = False
    Config = ParseConfigObject
    DatabaseWrapper = ParseDatabaseWrapper
    Var = ParseVar
    ref = ParseRefResolver
    source = ParseSourceResolver
    metric = ParseMetricResolver


class GenerateNameProvider(Provider):
    execute: bool = False
    Config = RuntimeConfigObject
    DatabaseWrapper = ParseDatabaseWrapper
    Var = RuntimeVar
    ref = ParseRefResolver
    source = ParseSourceResolver
    metric = ParseMetricResolver


class RuntimeProvider(Provider):
    execute: bool = True
    Config = RuntimeConfigObject
    DatabaseWrapper = RuntimeDatabaseWrapper
    Var = RuntimeVar
    ref = RuntimeRefResolver
    source = RuntimeSourceResolver
    metric = RuntimeMetricResolver


class RuntimeUnitTestProvider(Provider):
    execute: bool = True
    Config = RuntimeConfigObject
    DatabaseWrapper = RuntimeDatabaseWrapper
    Var = UnitTestVar
    ref = RuntimeUnitTestRefResolver
    source = RuntimeUnitTestSourceResolver
    metric = RuntimeMetricResolver


class OperationProvider(RuntimeProvider):
    ref = OperationRefResolver


T = TypeVar('T')


class ProviderContext(ManifestContext):
    def __init__(self, model: Any, config: Any, manifest: Manifest, provider: Provider, context_config: Any) -> None:
        if provider is None:
            raise DbtInternalError(f'Invalid provider given to context: {provider}')
        self.model = model
        super().__init__(config, manifest, model.package_name)
        self.sql_results: Dict[str, Any] = {}
        self.context_config = context_config
        self.provider = provider
        self.adapter = get_adapter(self.config)
        self.db_wrapper = self.provider.DatabaseWrapper(self.adapter, self.namespace)

    def _get_namespace_builder(self) -> MacroNamespaceBuilder:
        internal_packages = get_adapter_package_names(self.config.credentials.type)
        return MacroNamespaceBuilder(self.config.project_name, self.search_package, self.macro_stack, internal_packages, self.model)

    @contextproperty()
    def dbt_metadata_envs(self) -> Any:
        return get_metadata_vars()

    @contextproperty()
    def invocation_args_dict(self) -> Dict[str, Any]:
        return args_to_dict(self.config.args)

    @contextproperty()
    def _sql_results(self) -> Dict[str, Any]:
        return self.sql_results

    @contextmember()
    def load_result(self, name: str) -> Any:
        if name in self.sql_results:
            if name == 'main':
                return self.sql_results['main']
            elif self.sql_results[name] is None:
                raise MacroResultAlreadyLoadedError(name)
            else:
                ret_val = self.sql_results[name]
                self.sql_results[name] = None
                return ret_val
        else:
            return None

    @contextmember()
    def store_result(self, name: str, response: Any, agate_table: Optional[Any] = None) -> str:
        from dbt_common.clients import agate_helper
        if agate_table is None:
            agate_table = agate_helper.empty_table()
        self.sql_results[name] = AttrDict({
            'response': response,
            'data': agate_helper.as_matrix(agate_table),
            'table': agate_table,
        })
        return ''

    @contextmember()
    def store_raw_result(
        self,
        name: str,
        message: Optional[str] = None,
        code: Optional[str] = None,
        rows_affected: Optional[str] = None,
        agate_table: Optional[Any] = None,
    ) -> str:
        response = AdapterResponse(_message=message, code=code, rows_affected=rows_affected)
        return self.store_result(name, response, agate_table)

    @contextproperty()
    def validation(self) -> AttrDict:
        def validate_any(*args: Any) -> Callable[[Any], None]:
            def inner(value: Any) -> None:
                for arg in args:
                    if isinstance(arg, type) and isinstance(value, arg):
                        return
                    elif value == arg:
                        return
                raise DbtValidationError('Expected value "{}" to be one of {}'.format(value, ','.join(map(str, args))))
            return inner
        return AttrDict({'any': validate_any})

    @contextmember()
    def write(self, payload: Any) -> str:
        if isinstance(self.model, (Macro, SourceDefinition)):
            raise MacrosSourcesUnWriteableError(node=self.model)
        split_suffix: Optional[str] = None
        if isinstance(self.model, ModelNode) and self.model.config.get('incremental_strategy') == 'microbatch':
            split_suffix = MicrobatchBuilder.format_batch_start(
                self.model.config.get('__dbt_internal_microbatch_event_time_start'), self.model.config.batch_size
            )
        self.model.build_path = self.model.get_target_write_path(self.config.target_path, 'run', split_suffix=split_suffix)
        self.model.write_node(self.config.project_root, self.model.build_path, payload)
        return ''

    @contextmember()
    def render(self, string: str) -> str:
        return get_rendered(string, self._ctx, self.model)

    @contextmember()
    def try_or_compiler_error(self, message_if_exception: str, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception:
            raise CompilationError(message_if_exception, self.model)

    @contextmember()
    def load_agate_table(self) -> Any:
        from dbt_common.clients import agate_helper
        if not isinstance(self.model, SeedNode):
            raise LoadAgateTableNotSeedError(self.model.resource_type, node=self.model)
        package_path: str = (
            os.path.join(self.config.packages_install_path, self.model.package_name)
            if self.model.package_name != self.config.project_name
            else '.'
        )
        path: str = os.path.join(self.config.project_root, package_path, self.model.original_file_path)
        if not os.path.exists(path):
            assert self.model.root_path
            path = os.path.join(self.model.root_path, self.model.original_file_path)
        column_types = self.model.config.column_types
        delimiter = self.model.config.delimiter
        try:
            table = agate_helper.from_csv(path, text_columns=column_types, delimiter=delimiter)
        except ValueError as e:
            raise LoadAgateTableValueError(e, node=self.model)
        table.original_abspath = os.path.abspath(path)
        return table

    @contextproperty()
    def ref(self) -> Any:
        return self.provider.ref(self.db_wrapper, self.model, self.config, self.manifest)

    @contextproperty()
    def source(self) -> Any:
        return self.provider.source(self.db_wrapper, self.model, self.config, self.manifest)

    @contextproperty()
    def metric(self) -> Any:
        return self.provider.metric(self.db_wrapper, self.model, self.config, self.manifest)

    @contextproperty('config')
    def ctx_config(self) -> Any:
        return self.provider.Config(self.model, self.context_config)

    @contextproperty()
    def execute(self) -> bool:
        return self.provider.execute

    @contextproperty()
    def exceptions(self) -> Any:
        return wrapped_exports(self.model)

    @contextproperty()
    def database(self) -> str:
        return self.config.credentials.database

    @contextproperty()
    def schema(self) -> str:
        return self.config.credentials.schema

    @contextproperty()
    def var(self) -> Any:
        return self.provider.Var(context=self._ctx, config=self.config, node=self.model)

    @contextproperty('adapter')
    def ctx_adapter(self) -> Any:
        return self.db_wrapper

    @contextproperty()
    def api(self) -> Dict[str, Any]:
        return {'Relation': self.db_wrapper.Relation, 'Column': self.adapter.Column}

    @contextproperty()
    def column(self) -> Any:
        return self.adapter.Column

    @contextproperty()
    def env(self) -> Any:
        return self.target

    @contextproperty()
    def graph(self) -> Dict[Any, Any]:
        return self.manifest.flat_graph

    @contextproperty('model')
    def ctx_model(self) -> Dict[str, Any]:
        model_dct = self.model.to_dict(omit_none=True)
        if 'compiled_code' in model_dct:
            model_dct['compiled_sql'] = model_dct['compiled_code']
        if hasattr(self.model, 'contract') and self.model.contract.alias_types is True and ('columns' in model_dct):
            for column in model_dct['columns'].values():
                if 'data_type' in column:
                    orig_data_type = column['data_type']
                    new_data_type = self.adapter.Column.translate_type(orig_data_type)
                    column['data_type'] = new_data_type
        return model_dct

    @contextproperty()
    def pre_hooks(self) -> Any:
        if self.model.resource_type in [NodeType.Source, NodeType.Test, NodeType.Unit]:
            return []
        return [h.to_dict(omit_none=True) for h in self.model.config.pre_hook]

    @contextproperty()
    def post_hooks(self) -> Any:
        if self.model.resource_type in [NodeType.Source, NodeType.Test, NodeType.Unit]:
            return []
        return [h.to_dict(omit_none=True) for h in self.model.config.post_hook]

    @contextproperty()
    def sql(self) -> Any:
        return None

    @contextproperty()
    def sql_now(self) -> str:
        return self.adapter.date_function()

    @contextmember()
    def adapter_macro(self, name: str, *args: Any, **kwargs: Any) -> Any:
        msg = (
            'The "adapter_macro" macro has been deprecated. Instead, use the `adapter.dispatch` method '
            'to find a macro and call the result.  For more information, see: https://docs.getdbt.com/reference/dbt-jinja-functions/dispatch) '
            f'adapter_macro was called for: {name}'
        )
        raise CompilationError(msg)

    @contextmember()
    def env_var(self, var: str, default: Optional[Any] = None) -> Any:
        return_value: Optional[Any] = None
        if var.startswith(SECRET_ENV_PREFIX):
            raise SecretEnvVarLocationError(var)
        env = get_invocation_context().env
        if var in env:
            return_value = env[var]
        elif default is not None:
            return_value = default
        if return_value is not None:
            compiling: bool = True if hasattr(self.model, 'compiled') and getattr(self.model, 'compiled', False) is True else False
            if self.model and (not compiling):
                self.manifest.env_vars[var] = return_value if var in env else DEFAULT_ENV_PLACEHOLDER
                if self.model.file_id in self.manifest.files:
                    source_file = self.manifest.files[self.model.file_id]
                    if source_file.parse_file_type != 'schema':
                        source_file.env_vars.append(var)
            return return_value
        else:
            raise EnvVarMissingError(var)

    @contextproperty()
    def selected_resources(self) -> Any:
        return selected_resources.SELECTED_RESOURCES

    @contextmember()
    def submit_python_job(self, parsed_model: Any, compiled_code: str) -> Any:
        if not (self.context_macro_stack.depth == 2 and self.context_macro_stack.call_stack[1] == 'macro.dbt.statement' and ('materialization' in self.context_macro_stack.call_stack[0])):
            raise DbtRuntimeError(
                f'submit_python_job is not intended to be called here, at model {parsed_model["alias"]}, with macro call_stack {self.context_macro_stack.call_stack}.'
            )
        return self.adapter.submit_python_job(parsed_model, compiled_code)


class MacroContext(ProviderContext):
    def __init__(self, model: Any, config: Any, manifest: Manifest, provider: Provider, search_package: Optional[str]) -> None:
        super().__init__(model, config, manifest, provider, None)
        if search_package is None:
            self._search_package = config.project_name
        else:
            self._search_package = search_package


class SourceContext(ProviderContext):
    @contextproperty()
    def this(self) -> Any:
        return self.db_wrapper.Relation.create_from(self.config, self.model)

    @contextproperty()
    def source_node(self) -> Any:
        return self.model


class ModelContext(ProviderContext):
    @contextproperty()
    def pre_hooks(self) -> List[Any]:
        if self.model.resource_type in [NodeType.Source, NodeType.Test, NodeType.Unit]:
            return []
        return [h.to_dict(omit_none=True) for h in self.model.config.pre_hook]

    @contextproperty()
    def post_hooks(self) -> List[Any]:
        if self.model.resource_type in [NodeType.Source, NodeType.Test, NodeType.Unit]:
            return []
        return [h.to_dict(omit_none=True) for h in self.model.config.post_hook]

    @contextproperty()
    def compiled_code(self) -> Optional[str]:
        if getattr(self.model, 'defer_relation', None) and self.config.args.which == 'clone':
            return f'select * from {self.model.defer_relation.relation_name or str(self.defer_relation)}'
        elif getattr(self.model, 'extra_ctes_injected', None):
            return self.model.compiled_code
        else:
            return None

    @contextproperty()
    def sql(self) -> Optional[str]:
        if self.model.language == ModelLanguage.sql:
            return self.compiled_code
        else:
            return None

    @contextproperty()
    def database(self) -> str:
        return getattr(self.model, 'database', self.config.credentials.database)

    @contextproperty()
    def schema(self) -> str:
        return getattr(self.model, 'schema', self.config.credentials.schema)

    @contextproperty()
    def this(self) -> Any:
        if self.model.resource_type == NodeType.Operation:
            return None
        return self.db_wrapper.Relation.create_from(self.config, self.model)

    @contextproperty()
    def defer_relation(self) -> Any:
        if getattr(self.model, 'defer_relation', None):
            return self.db_wrapper.Relation.create_from(self.config, self.model.defer_relation)
        else:
            return None


class UnitTestContext(ModelContext):
    @contextmember()
    def env_var(self, var: str, default: Optional[Any] = None) -> Any:
        if self.model.overrides and self.model.overrides.env_vars and var in self.model.overrides.env_vars:
            return self.model.overrides.env_vars[var]
        else:
            return super().env_var(var, default)

    @contextproperty()
    def this(self) -> Any:
        if self.model.this_input_node_unique_id:
            this_node = self.manifest.expect(self.model.this_input_node_unique_id)
            self.model.set_cte(this_node.unique_id, None)
            return self.adapter.Relation.add_ephemeral_prefix(this_node.identifier)
        return None


def generate_parser_model_context(model: Any, config: Any, manifest: Manifest, context_config: Any) -> Dict[str, Any]:
    ctx = ModelContext(model, config, manifest, ParseProvider(), context_config)
    return ctx.to_dict()


def generate_generate_name_macro_context(macro: Any, config: Any, manifest: Manifest) -> Dict[str, Any]:
    ctx = MacroContext(macro, config, manifest, GenerateNameProvider(), None)
    return ctx.to_dict()


def generate_runtime_model_context(model: Any, config: Any, manifest: Manifest) -> Dict[str, Any]:
    ctx = ModelContext(model, config, manifest, RuntimeProvider(), None)
    return ctx.to_dict()


def generate_runtime_macro_context(macro: Any, config: Any, manifest: Manifest, package_name: str) -> Dict[str, Any]:
    ctx = MacroContext(macro, config, manifest, OperationProvider(), package_name)
    return ctx.to_dict()


def generate_runtime_unit_test_context(unit_test: UnitTestNode, config: Any, manifest: Manifest) -> Dict[str, Any]:
    ctx = UnitTestContext(unit_test, config, manifest, RuntimeUnitTestProvider(), None)
    ctx_dict: Dict[str, Any] = ctx.to_dict()
    if unit_test.overrides and unit_test.overrides.macros:
        global_macro_overrides: Dict[str, Any] = {}
        package_macro_overrides: Dict[Tuple[str, str], Any] = {}
        for macro_name, macro_value in unit_test.overrides.macros.items():
            macro_name_split = macro_name.split('.')
            macro_package: Optional[str] = macro_name_split[0] if len(macro_name_split) == 2 else None
            macro_name_only = macro_name_split[-1]
            if macro_package is None and macro_name_only in ctx_dict:
                original_context_value = ctx_dict[macro_name_only]
                if isinstance(original_context_value, MacroGenerator):
                    macro_value = UnitTestMacroGenerator(original_context_value, macro_value)
                global_macro_overrides[macro_name_only] = macro_value
            elif macro_package and macro_package in ctx_dict and (macro_name_only in ctx_dict[macro_package]):
                original_context_value = ctx_dict[macro_package][macro_name_only]
                if isinstance(original_context_value, MacroGenerator):
                    macro_value = UnitTestMacroGenerator(original_context_value, macro_value)
                package_macro_overrides[(macro_package, macro_name_only)] = macro_value
        for (macro_package, macro_name_only), macro_override_value in package_macro_overrides.items():
            ctx_dict[macro_package][macro_name_only] = macro_override_value
            if macro_package == 'dbt':
                ctx_dict[macro_name_only] = macro_override_value
        for macro_name_only, macro_override_value in global_macro_overrides.items():
            ctx_dict[macro_name_only] = macro_override_value
            if ctx_dict['dbt'].get(macro_name_only):
                ctx_dict['dbt'][macro_name_only] = macro_override_value
    return ctx_dict


class ExposureRefResolver(BaseResolver):
    def __call__(self, *args: Any, **kwargs: Any) -> str:
        package: Optional[str] = None
        if len(args) == 1:
            name = args[0]
        elif len(args) == 2:
            package, name = args
        else:
            raise RefArgsError(node=self.model, args=args)
        version = kwargs.get('version') or kwargs.get('v')
        self.model.refs.append(RefArgs(package=package, name=name, version=version))
        return ''


class ExposureSourceResolver(BaseResolver):
    def __call__(self, *args: Any) -> str:
        if len(args) != 2:
            raise NumberSourceArgsError(args, node=self.model)
        self.model.sources.append(list(args))
        return ''


class ExposureMetricResolver(BaseResolver):
    def __call__(self, *args: Any) -> str:
        if len(args) not in (1, 2):
            raise MetricArgsError(node=self.model, args=args)
        self.model.metrics.append(list(args))
        return ''


def generate_parse_exposure(exposure: Any, config: Any, manifest: Manifest, package_name: str) -> Dict[str, Any]:
    project = config.load_dependencies()[package_name]
    return {
        'ref': ExposureRefResolver(None, exposure, project, manifest),
        'source': ExposureSourceResolver(None, exposure, project, manifest),
        'metric': ExposureMetricResolver(None, exposure, project, manifest),
    }


class SemanticModelRefResolver(BaseResolver):
    def __call__(self, *args: Any, **kwargs: Any) -> str:
        package: Optional[str] = None
        if len(args) == 1:
            name = args[0]
        elif len(args) == 2:
            package, name = args
        else:
            raise RefArgsError(node=self.model, args=args)
        version = kwargs.get('version') or kwargs.get('v')
        self.validate_args(name, package, version)
        self.model.refs.append(RefArgs(package=package, name=name, version=version))
        return ''

    def validate_args(self, name: str, package: Optional[str], version: Optional[Any]) -> None:
        if not isinstance(name, str):
            raise ParsingError(
                f'In a semantic model or metrics section in {self.model.original_file_path} the name argument to ref() must be a string'
            )


def generate_parse_semantic_models(semantic_model: Any, config: Any, manifest: Manifest, package_name: str) -> Dict[str, Any]:
    project = config.load_dependencies()[package_name]
    return {'ref': SemanticModelRefResolver(None, semantic_model, project, manifest)}


class TestContext(ProviderContext):
    def __init__(self, model: Any, config: Any, manifest: Manifest, provider: Provider, context_config: Any, macro_resolver: MacroResolver) -> None:
        self.macro_resolver = macro_resolver
        self.thread_ctx = MacroStack()
        super().__init__(model, config, manifest, provider, context_config)
        self._build_test_namespace()
        self.db_wrapper = self.provider.DatabaseWrapper(self.adapter, self.namespace)

    def _build_namespace(self) -> Dict[str, Any]:
        return {}

    def _build_test_namespace(self) -> None:
        depends_on_macros: List[str] = []
        get_where_subquery = self.macro_resolver.macros_by_name.get('get_where_subquery')
        if get_where_subquery:
            depends_on_macros.append(get_where_subquery.unique_id)
        if self.model.depends_on and self.model.depends_on.macros:
            depends_on_macros.extend(self.model.depends_on.macros)
        lookup_macros = depends_on_macros.copy()
        for macro_unique_id in lookup_macros:
            lookup_macro = self.macro_resolver.macros.get(macro_unique_id)
            if lookup_macro:
                depends_on_macros.extend(lookup_macro.depends_on.macros)
        macro_namespace = TestMacroNamespace(self.macro_resolver, self._ctx, self.model, self.thread_ctx, depends_on_macros)
        self.namespace = macro_namespace

    @contextmember()
    def env_var(self, var: str, default: Optional[Any] = None) -> Any:
        return_value: Optional[Any] = None
        if var.startswith(SECRET_ENV_PREFIX):
            raise SecretEnvVarLocationError(var)
        env = get_invocation_context().env
        if var in env:
            return_value = env[var]
        elif default is not None:
            return_value = default
        if return_value is not None:
            if self.model:
                self.manifest.env_vars[var] = return_value if var in env else DEFAULT_ENV_PLACEHOLDER
                if self.model.resource_type == NodeType.Test and self.model.file_key_name:
                    source_file = self.manifest.files[self.model.file_id]
                    yaml_key, name = self.model.file_key_name.split('.')
                    source_file.add_env_var(var, yaml_key, name)
            return return_value
        else:
            raise EnvVarMissingError(var)


def generate_test_context(model: Any, config: Any, manifest: Manifest, context_config: Any, macro_resolver: MacroResolver) -> Dict[str, Any]:
    ctx = TestContext(model, config, manifest, ParseProvider(), context_config, macro_resolver)
    return ctx.to_dict()