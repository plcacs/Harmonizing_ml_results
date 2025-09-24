import abc
import os
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Type, TypeVar, Union, cast
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
_MISSING: Any = object()

class RelationProxy:

    def __init__(self, adapter: Any) -> None:
        self._quoting_config: Dict[str, Any] = adapter.config.quoting
        self._relation_type: Any = adapter.Relation

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
        self._adapter: Any = adapter
        self.Relation: RelationProxy = RelationProxy(adapter)
        self._namespace: Any = namespace

    def __getattr__(self, name: str) -> Any:
        raise NotImplementedError('subclasses need to implement this')

    @property
    def config(self) -> Any:
        return self._adapter.config

    def type(self) -> str:
        return self._adapter.type()

    def commit(self) -> None:
        return self._adapter.commit_if_has_connection()

    def _get_adapter_macro_prefixes(self) -> List[str]:
        search_prefixes: List[str] = get_adapter_type_names(self._adapter.type()) + ['default']
        return search_prefixes

    def _get_search_packages(self, namespace: Optional[str] = None) -> List[Optional[str]]:
        search_packages: List[Optional[str]] = [None]
        if namespace is None:
            search_packages = [None]
        elif isinstance(namespace, str):
            macro_search_order: Optional[List[str]] = self._adapter.config.get_macro_search_order(namespace)
            if macro_search_order:
                search_packages = macro_search_order
            elif not macro_search_order and namespace in self._adapter.config.dependencies:
                search_packages = [self.config.project_name, namespace]
        else:
            raise CompilationError(f'In adapter.dispatch, got a {type(namespace)} macro_namespace argument ("{namespace}"), but macro_namespace should be None or a string.')
        return search_packages

    def dispatch(self, macro_name: str, macro_namespace: Optional[str] = None, packages: Optional[Any] = None) -> Any:
        search_packages: List[Optional[str]]
        if '.' in macro_name:
            (suggest_macro_namespace, suggest_macro_name) = macro_name.split('.', 1)
            msg = f'In adapter.dispatch, got a macro name of "{macro_name}", but "." is not a valid macro name component. Did you mean `adapter.dispatch("{suggest_macro_name}", macro_namespace="{suggest_macro_namespace}")`?'
            raise CompilationError(msg)
        if packages is not None:
            raise MacroDispatchArgError(macro_name)
        search_packages = self._get_search_packages(macro_namespace)
        attempts: List[str] = []
        for package_name in search_packages:
            for prefix in self._get_adapter_macro_prefixes():
                search_name: str = f'{prefix}__{macro_name}'
                try:
                    macro: Optional[Any] = self._namespace.get_from_package(package_name, search_name)
                except CompilationError:
                    macro = None
                if package_name is None:
                    attempts.append(search_name)
                else:
                    attempts.append(f'{package_name}.{search_name}')
                if macro is not None:
                    return macro
        searched: str = ', '.join((repr(a) for a in attempts))
        msg = f"In dispatch: No macro named '{macro_name}' found within namespace: '{macro_namespace}'\n    Searched for: {searched}"
        raise CompilationError(msg)

class BaseResolver(metaclass=abc.ABCMeta):

    def __init__(self, db_wrapper: BaseDatabaseWrapper, model: Any, config: RuntimeConfig, manifest: Manifest) -> None:
        self.db_wrapper: BaseDatabaseWrapper = db_wrapper
        self.model: Any = model
        self.config: RuntimeConfig = config
        self.manifest: Manifest = manifest

    @property
    def current_project(self) -> str:
        return self.config.project_name

    @property
    def Relation(self) -> RelationProxy:
        return self.db_wrapper.Relation

    @property
    def resolve_limit(self) -> Optional[int]:
        return 0 if getattr(self.config.args, 'EMPTY', False) else None

    def resolve_event_time_filter(self, target: Any) -> Optional[EventTimeFilter]:
        event_time_filter: Optional[EventTimeFilter] = None
        sample_mode: bool = bool(os.environ.get('DBT_EXPERIMENTAL_SAMPLE_MODE') and getattr(self.config.args, 'sample', None))
        if (isinstance(target.config, NodeConfig) or isinstance(target.config, SourceConfig) or isinstance(target.config, SeedConfig)) and target.config.event_time and isinstance(self.model, ModelNode):
            if self.model.config.materialized == 'incremental' and self.model.config.incremental_strategy == 'microbatch' and self.manifest.use_microbatch_batches(project_name=self.config.project_name) and (self.model.batch is not None):
                if sample_mode:
                    start: Any = self.config.args.sample.start if self.config.args.sample.start > self.model.batch.event_time_start else self.model.batch.event_time_start
                    end: Any = self.config.args.sample.end if self.config.args.sample.end < self.model.batch.event_time_end else self.model.batch.event_time_end
                    event_time_filter = EventTimeFilter(field_name=target.config.event_time, start=start, end=end)
                else:
                    event_time_filter = EventTimeFilter(field_name=target.config.event_time, start=self.model.batch.event_time_start, end=self.model.batch.event_time_end)
            elif sample_mode:
                event_time_filter = EventTimeFilter(field_name=target.config.event_time, start=self.config.args.sample.start, end=self.config.args.sample.end)
        return event_time_filter

    @abc.abstractmethod
    def __call__(self, *args: str) -> Any:
        pass

class BaseRefResolver(BaseResolver):

    @abc.abstractmethod
    def resolve(self, name: str, package: Optional[str] = None, version: Optional[NodeVersion] = None) -> Any:
        ...

    def _repack_args(self, name: str, package: Optional[str], version: Optional[NodeVersion]) -> RefArgs:
        return RefArgs(package=package, name=name, version=version)

    def validate_args(self, name: str, package: Optional[str], version: Optional[NodeVersion]) -> None:
        if not isinstance(name, str):
            raise CompilationError(f'The name argument to ref() must be a string, got {type(name)}')
        if package is not None and (not isinstance(package, str)):
            raise CompilationError(f'The package argument to ref() must be a string or None, got {type(package)}')
        if version is not None and (not isinstance(version, (str, int, float))):
            raise CompilationError(f'The version argument to ref() must be a string, int, float, or None - got {type(version)}')

    def __call__(self, *args: str, **kwargs: Any) -> Any:
        name: str
        package: Optional[str] = None
        version: Optional[NodeVersion] = None
        if len(args) == 1:
            name = args[0]
        elif len(args) == 2:
            (package, name) = args
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
            raise CompilationError(f'The source name (first) argument to source() must be a string, got {type(source_name)}')
        if not isinstance(table_name, str):
            raise CompilationError(f'The table name (second) argument to source() must be a string, got {type(table_name)}')

    def __call__(self, *args: str) -> RelationProxy:
        if len(args) != 2:
            raise NumberSourceArgsError(args, node=self.model)
        self.validate_args(args[0], args[1])
        return self.resolve(args[0], args[1])

class BaseMetricResolver(BaseResolver):

    @abc.abstractmethod
    def resolve(self, name: str, package: Optional[str] = None) -> MetricReference:
        ...

    def _repack_args(self, name: str, package: Optional[str]) -> List[str]:
        if package is None:
            return [name]
        else:
            return [package, name]

    def validate_args(self, name: str, package: Optional[str]) -> None:
        if not isinstance(name, str):
            raise CompilationError(f'The name argument to metric() must be a string, got {type(name)}')
        if package is not None and (not isinstance(package, str)):
            raise CompilationError(f'The package argument to metric() must be a string or None, got {type(package)}')

    def __call__(self, *args: str) -> MetricReference:
        name: str
        package: Optional[str] = None
        if len(args) == 1:
            name = args[0]
        elif len(args) == 2:
            (package, name) = args
        else:
            raise MetricArgsError(node=self.model, args=args)
        self.validate_args(name, package)
        return self.resolve(name, package)

class Config(Protocol):

    def __init__(self, model: Any, context_config: Optional[ContextConfig]) -> None:
        ...

class ParseConfigObject(Config):

    def __init__(self, model: Any, context_config: Optional[ContextConfig]) -> None:
        self.model: Any = model
        self.context_config: Optional[ContextConfig] = context_config

    def _transform_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        for oldkey in ('pre_hook', 'post_hook'):
            if oldkey in config:
                newkey: str = oldkey.replace('_', '-')
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
            unrendered_config: Optional[Dict[str, Any]] = statically_parse_unrendered_config(self.model.raw_code)
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

    def __init__(self, model: Any, context_config: Optional[ContextConfig] = None) -> None:
        self.model: Any = model

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        return ''

    def set(self, name: str, value: Any) -> str:
        return self.__call__({name: value})

    def _validate(self, validator: Optional[Callable[[Any], None]], value: Any) -> None:
        if validator is not None:
            validator(value)

    def _lookup(self, name: str, default: Any = _MISSING) -> Any:
        if not hasattr(self.model, 'config'):
            result: Any = default
        else:
            result = self.model.config.get(name, default)
        if result is _MISSING:
            raise MissingConfigError(unique_id=self.model.unique_id, name=name)
        return result

    def require(self, name: str, validator: Optional[Callable[[Any], None]] = None) -> Any:
        to_return: Any = self._lookup(name)
        if validator is not None:
            self._validate(validator, to_return)
        return to_return

    def get(self, name: str, default: Any = None, validator: Optional[Callable[[Any], None]] = None) -> Any:
        to_return: Any = self._lookup(name, default)
        if validator is not None and default is not None:
            self._validate(validator, to_return)
        return to_return

    def persist_relation_docs(self) -> bool:
        persist_docs: Dict[str, Any] = self.get('persist_docs', default={})
        if not isinstance(persist_docs, dict):
            raise PersistDocsValueTypeError(persist_docs)
        return persist_docs.get('relation', False)

    def persist_column_docs(self) -> bool:
        persist_docs: Dict[str, Any] = self.get('persist_docs', default={})
        if not isinstance(persist_docs, dict):
            raise PersistDocsValueTypeError(persist_docs)
        return persist_docs.get('columns', False)

class ParseDatabaseWrapper(BaseDatabaseWrapper):
    """The parser subclass of the database wrapper applies any explicit
    parse-time overrides.
    """

    def __getattr__(self, name: str) -> Any:
        override: bool = name in self._adapter._available_ and name in self._adapter._parse_replacements_
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
        if name in self._adapter._available_