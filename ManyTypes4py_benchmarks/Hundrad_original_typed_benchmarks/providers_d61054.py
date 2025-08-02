import abc
import os
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Type, TypeVar, Union
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
_MISSING = object()

class RelationProxy:

    def __init__(self, adapter):
        self._quoting_config = adapter.config.quoting
        self._relation_type = adapter.Relation

    def __getattr__(self, key):
        return getattr(self._relation_type, key)

    def create(self, *args, **kwargs):
        kwargs['quote_policy'] = merge(self._quoting_config, kwargs.pop('quote_policy', {}))
        return self._relation_type.create(*args, **kwargs)

class BaseDatabaseWrapper:
    """
    Wrapper for runtime database interaction. Applies the runtime quote policy
    via a relation proxy.
    """

    def __init__(self, adapter, namespace):
        self._adapter = adapter
        self.Relation = RelationProxy(adapter)
        self._namespace = namespace

    def __getattr__(self, name):
        raise NotImplementedError('subclasses need to implement this')

    @property
    def config(self):
        return self._adapter.config

    def type(self):
        return self._adapter.type()

    def commit(self):
        return self._adapter.commit_if_has_connection()

    def _get_adapter_macro_prefixes(self):
        search_prefixes = get_adapter_type_names(self._adapter.type()) + ['default']
        return search_prefixes

    def _get_search_packages(self, namespace=None):
        search_packages = [None]
        if namespace is None:
            search_packages = [None]
        elif isinstance(namespace, str):
            macro_search_order = self._adapter.config.get_macro_search_order(namespace)
            if macro_search_order:
                search_packages = macro_search_order
            elif not macro_search_order and namespace in self._adapter.config.dependencies:
                search_packages = [self.config.project_name, namespace]
        else:
            raise CompilationError(f'In adapter.dispatch, got a {type(namespace)} macro_namespace argument ("{namespace}"), but macro_namespace should be None or a string.')
        return search_packages

    def dispatch(self, macro_name, macro_namespace=None, packages=None):
        if '.' in macro_name:
            suggest_macro_namespace, suggest_macro_name = macro_name.split('.', 1)
            msg = f'In adapter.dispatch, got a macro name of "{macro_name}", but "." is not a valid macro name component. Did you mean `adapter.dispatch("{suggest_macro_name}", macro_namespace="{suggest_macro_namespace}")`?'
            raise CompilationError(msg)
        if packages is not None:
            raise MacroDispatchArgError(macro_name)
        search_packages = self._get_search_packages(macro_namespace)
        attempts = []
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

    def __init__(self, db_wrapper, model, config, manifest):
        self.db_wrapper = db_wrapper
        self.model = model
        self.config = config
        self.manifest = manifest

    @property
    def current_project(self):
        return self.config.project_name

    @property
    def Relation(self):
        return self.db_wrapper.Relation

    @property
    def resolve_limit(self):
        return 0 if getattr(self.config.args, 'EMPTY', False) else None

    def resolve_event_time_filter(self, target):
        event_time_filter = None
        sample_mode = bool(os.environ.get('DBT_EXPERIMENTAL_SAMPLE_MODE') and getattr(self.config.args, 'sample', None))
        if (isinstance(target.config, NodeConfig) or isinstance(target.config, SourceConfig) or isinstance(target.config, SeedConfig)) and target.config.event_time and isinstance(self.model, ModelNode):
            if self.model.config.materialized == 'incremental' and self.model.config.incremental_strategy == 'microbatch' and self.manifest.use_microbatch_batches(project_name=self.config.project_name) and (self.model.batch is not None):
                if sample_mode:
                    start = self.config.args.sample.start if self.config.args.sample.start > self.model.batch.event_time_start else self.model.batch.event_time_start
                    end = self.config.args.sample.end if self.config.args.sample.end < self.model.batch.event_time_end else self.model.batch.event_time_end
                    event_time_filter = EventTimeFilter(field_name=target.config.event_time, start=start, end=end)
                else:
                    event_time_filter = EventTimeFilter(field_name=target.config.event_time, start=self.model.batch.event_time_start, end=self.model.batch.event_time_end)
            elif sample_mode:
                event_time_filter = EventTimeFilter(field_name=target.config.event_time, start=self.config.args.sample.start, end=self.config.args.sample.end)
        return event_time_filter

    @abc.abstractmethod
    def __call__(self, *args):
        pass

class BaseRefResolver(BaseResolver):

    @abc.abstractmethod
    def resolve(self, name, package=None, version=None):
        ...

    def _repack_args(self, name, package, version):
        return RefArgs(package=package, name=name, version=version)

    def validate_args(self, name, package, version):
        if not isinstance(name, str):
            raise CompilationError(f'The name argument to ref() must be a string, got {type(name)}')
        if package is not None and (not isinstance(package, str)):
            raise CompilationError(f'The package argument to ref() must be a string or None, got {type(package)}')
        if version is not None and (not isinstance(version, (str, int, float))):
            raise CompilationError(f'The version argument to ref() must be a string, int, float, or None - got {type(version)}')

    def __call__(self, *args, **kwargs):
        package = None
        version = None
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
    def resolve(self, source_name, table_name):
        pass

    def validate_args(self, source_name, table_name):
        if not isinstance(source_name, str):
            raise CompilationError(f'The source name (first) argument to source() must be a string, got {type(source_name)}')
        if not isinstance(table_name, str):
            raise CompilationError(f'The table name (second) argument to source() must be a string, got {type(table_name)}')

    def __call__(self, *args):
        if len(args) != 2:
            raise NumberSourceArgsError(args, node=self.model)
        self.validate_args(args[0], args[1])
        return self.resolve(args[0], args[1])

class BaseMetricResolver(BaseResolver):

    @abc.abstractmethod
    def resolve(self, name, package=None):
        ...

    def _repack_args(self, name, package):
        if package is None:
            return [name]
        else:
            return [package, name]

    def validate_args(self, name, package):
        if not isinstance(name, str):
            raise CompilationError(f'The name argument to metric() must be a string, got {type(name)}')
        if package is not None and (not isinstance(package, str)):
            raise CompilationError(f'The package argument to metric() must be a string or None, got {type(package)}')

    def __call__(self, *args):
        package = None
        if len(args) == 1:
            name = args[0]
        elif len(args) == 2:
            package, name = args
        else:
            raise MetricArgsError(node=self.model, args=args)
        self.validate_args(name, package)
        return self.resolve(name, package)

class Config(Protocol):

    def __init__(self, model, context_config):
        ...

class ParseConfigObject(Config):

    def __init__(self, model, context_config):
        self.model = model
        self.context_config = context_config

    def _transform_config(self, config):
        for oldkey in ('pre_hook', 'post_hook'):
            if oldkey in config:
                newkey = oldkey.replace('_', '-')
                if newkey in config:
                    raise ConflictingConfigKeysError(oldkey, newkey, node=self.model)
                config[newkey] = config.pop(oldkey)
        return config

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0:
            opts = args[0]
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

    def set(self, name, value):
        return self.__call__({name: value})

    def require(self, name, validator=None):
        return ''

    def get(self, name, default=None, validator=None):
        return ''

    def persist_relation_docs(self):
        return False

    def persist_column_docs(self):
        return False

class RuntimeConfigObject(Config):

    def __init__(self, model, context_config=None):
        self.model = model

    def __call__(self, *args, **kwargs):
        return ''

    def set(self, name, value):
        return self.__call__({name: value})

    def _validate(self, validator, value):
        validator(value)

    def _lookup(self, name, default=_MISSING):
        if not hasattr(self.model, 'config'):
            result = default
        else:
            result = self.model.config.get(name, default)
        if result is _MISSING:
            raise MissingConfigError(unique_id=self.model.unique_id, name=name)
        return result

    def require(self, name, validator=None):
        to_return = self._lookup(name)
        if validator is not None:
            self._validate(validator, to_return)
        return to_return

    def get(self, name, default=None, validator=None):
        to_return = self._lookup(name, default)
        if validator is not None and default is not None:
            self._validate(validator, to_return)
        return to_return

    def persist_relation_docs(self):
        persist_docs = self.get('persist_docs', default={})
        if not isinstance(persist_docs, dict):
            raise PersistDocsValueTypeError(persist_docs)
        return persist_docs.get('relation', False)

    def persist_column_docs(self):
        persist_docs = self.get('persist_docs', default={})
        if not isinstance(persist_docs, dict):
            raise PersistDocsValueTypeError(persist_docs)
        return persist_docs.get('columns', False)

class ParseDatabaseWrapper(BaseDatabaseWrapper):
    """The parser subclass of the database wrapper applies any explicit
    parse-time overrides.
    """

    def __getattr__(self, name):
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

    def __getattr__(self, name):
        if name in self._adapter._available_:
            return getattr(self._adapter, name)
        else:
            raise AttributeError("'{}' object has no attribute '{}'".format(self.__class__.__name__, name))

class ParseRefResolver(BaseRefResolver):

    def resolve(self, name, package=None, version=None):
        self.model.refs.append(self._repack_args(name, package, version))
        return self.Relation.create_from(self.config, self.model)
ResolveRef = Union[Disabled, ManifestNode]

class RuntimeRefResolver(BaseRefResolver):

    def resolve(self, target_name, target_package=None, target_version=None):
        target_model = self.manifest.resolve_ref(self.model, target_name, target_package, target_version, self.current_project, self.model.package_name)
        if target_model is None or isinstance(target_model, Disabled):
            raise TargetNotFoundError(node=self.model, target_name=target_name, target_kind='node', target_package=target_package, target_version=target_version, disabled=isinstance(target_model, Disabled))
        elif self.manifest.is_invalid_private_ref(self.model, target_model, self.config.dependencies):
            raise DbtReferenceError(unique_id=self.model.unique_id, ref_unique_id=target_model.unique_id, access=AccessType.Private, scope=cast_to_str(target_model.group))
        elif self.manifest.is_invalid_protected_ref(self.model, target_model, self.config.dependencies):
            raise DbtReferenceError(unique_id=self.model.unique_id, ref_unique_id=target_model.unique_id, access=AccessType.Protected, scope=target_model.package_name)
        self.validate(target_model, target_name, target_package, target_version)
        return self.create_relation(target_model)

    def create_relation(self, target_model):
        if target_model.is_ephemeral_model:
            self.model.set_cte(target_model.unique_id, None)
            return self.Relation.create_ephemeral_from(target_model, limit=self.resolve_limit, event_time_filter=self.resolve_event_time_filter(target_model))
        elif hasattr(target_model, 'defer_relation') and target_model.defer_relation and self.config.args.defer and (self.config.args.favor_state and target_model.unique_id not in selected_resources.SELECTED_RESOURCES or not get_adapter(self.config).get_relation(target_model.database, target_model.schema, target_model.identifier)):
            return self.Relation.create_from(self.config, target_model.defer_relation, limit=self.resolve_limit, event_time_filter=self.resolve_event_time_filter(target_model))
        else:
            return self.Relation.create_from(self.config, target_model, limit=self.resolve_limit, event_time_filter=self.resolve_event_time_filter(target_model))

    def validate(self, resolved, target_name, target_package, target_version):
        if resolved.unique_id not in self.model.depends_on.nodes:
            args = self._repack_args(target_name, target_package, target_version)
            raise RefBadContextError(node=self.model, args=args)

class OperationRefResolver(RuntimeRefResolver):

    def validate(self, resolved, target_name, target_package, target_version):
        pass

    def create_relation(self, target_model):
        if target_model.is_ephemeral_model:
            raise OperationsCannotRefEphemeralNodesError(target_model.name, node=self.model)
        else:
            return super().create_relation(target_model)

class RuntimeUnitTestRefResolver(RuntimeRefResolver):

    @property
    def resolve_limit(self):
        return None

    def resolve(self, target_name, target_package=None, target_version=None):
        return super().resolve(target_name, target_package, target_version)

class ParseSourceResolver(BaseSourceResolver):

    def resolve(self, source_name, table_name):
        self.model.sources.append([source_name, table_name])
        return self.Relation.create_from(self.config, self.model)

class RuntimeSourceResolver(BaseSourceResolver):

    def resolve(self, source_name, table_name):
        target_source = self.manifest.resolve_source(source_name, table_name, self.current_project, self.model.package_name)
        if target_source is None or isinstance(target_source, Disabled):
            raise TargetNotFoundError(node=self.model, target_name=f'{source_name}.{table_name}', target_kind='source', disabled=isinstance(target_source, Disabled))

        class SourceQuotingBaseConfig:
            quoting = {}
        return self.Relation.create_from(SourceQuotingBaseConfig(), target_source, limit=self.resolve_limit, event_time_filter=self.resolve_event_time_filter(target_source))

class RuntimeUnitTestSourceResolver(BaseSourceResolver):

    @property
    def resolve_limit(self):
        return None

    def resolve(self, source_name, table_name):
        target_source = self.manifest.resolve_source(source_name, table_name, self.current_project, self.model.package_name)
        if target_source is None or isinstance(target_source, Disabled):
            raise TargetNotFoundError(node=self.model, target_name=f'{source_name}.{table_name}', target_kind='source', disabled=isinstance(target_source, Disabled))
        self.model.set_cte(target_source.unique_id, None)
        return self.Relation.create_ephemeral_from(target_source)

class ParseMetricResolver(BaseMetricResolver):

    def resolve(self, name, package=None):
        self.model.metrics.append(self._repack_args(name, package))
        return MetricReference(name, package)

class RuntimeMetricResolver(BaseMetricResolver):

    def resolve(self, target_name, target_package=None):
        target_metric = self.manifest.resolve_metric(target_name, target_package, self.current_project, self.model.package_name)
        if target_metric is None or isinstance(target_metric, Disabled):
            raise TargetNotFoundError(node=self.model, target_name=target_name, target_kind='metric', target_package=target_package)
        return ResolvedMetricReference(target_metric, self.manifest)

class ModelConfiguredVar(Var):

    def __init__(self, context, config, node):
        self._config = config
        super().__init__(context, config.cli_vars, node=node)

    def packages_for_node(self):
        dependencies = self._config.load_dependencies()
        package_name = self._node.package_name
        if package_name != self._config.project_name:
            if package_name in dependencies:
                yield dependencies[package_name]
        yield self._config

    def _generate_merged(self):
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

    def get_missing_var(self, var_name):
        return None

class RuntimeVar(ModelConfiguredVar):
    pass

class UnitTestVar(RuntimeVar):

    def __init__(self, context, config, node):
        config_copy = None
        assert isinstance(node, UnitTestNode)
        if node.overrides and node.overrides.vars:
            config_copy = deepcopy(config)
            config_copy.cli_vars.update(node.overrides.vars)
        super().__init__(context, config_copy or config, node=node)

class Provider(Protocol):
    pass

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
T = TypeVar('T')

class ProviderContext(ManifestContext):

    def __init__(self, model, config, manifest, provider, context_config):
        if provider is None:
            raise DbtInternalError(f'Invalid provider given to context: {provider}')
        self.model = model
        super().__init__(config, manifest, model.package_name)
        self.sql_results = {}
        self.context_config = context_config
        self.provider = provider
        self.adapter = get_adapter(self.config)
        self.db_wrapper = self.provider.DatabaseWrapper(self.adapter, self.namespace)

    def _get_namespace_builder(self):
        internal_packages = get_adapter_package_names(self.config.credentials.type)
        return MacroNamespaceBuilder(self.config.project_name, self.search_package, self.macro_stack, internal_packages, self.model)

    @contextproperty()
    def dbt_metadata_envs(self):
        return get_metadata_vars()

    @contextproperty()
    def invocation_args_dict(self):
        return args_to_dict(self.config.args)

    @contextproperty()
    def _sql_results(self):
        return self.sql_results

    @contextmember()
    def load_result(self, name):
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
    def store_result(self, name, response, agate_table=None):
        from dbt_common.clients import agate_helper
        if agate_table is None:
            agate_table = agate_helper.empty_table()
        self.sql_results[name] = AttrDict({'response': response, 'data': agate_helper.as_matrix(agate_table), 'table': agate_table})
        return ''

    @contextmember()
    def store_raw_result(self, name, message=Optional[str], code=Optional[str], rows_affected=Optional[str], agate_table=None):
        response = AdapterResponse(_message=message, code=code, rows_affected=rows_affected)
        return self.store_result(name, response, agate_table)

    @contextproperty()
    def validation(self):

        def validate_any(*args):

            def inner(value):
                for arg in args:
                    if isinstance(arg, type) and isinstance(value, arg):
                        return
                    elif value == arg:
                        return
                raise DbtValidationError('Expected value "{}" to be one of {}'.format(value, ','.join(map(str, args))))
            return inner
        return AttrDict({'any': validate_any})

    @contextmember()
    def write(self, payload):
        if isinstance(self.model, (Macro, SourceDefinition)):
            raise MacrosSourcesUnWriteableError(node=self.model)
        split_suffix = None
        if isinstance(self.model, ModelNode) and self.model.config.get('incremental_strategy') == 'microbatch':
            split_suffix = MicrobatchBuilder.format_batch_start(self.model.config.get('__dbt_internal_microbatch_event_time_start'), self.model.config.batch_size)
        self.model.build_path = self.model.get_target_write_path(self.config.target_path, 'run', split_suffix=split_suffix)
        self.model.write_node(self.config.project_root, self.model.build_path, payload)
        return ''

    @contextmember()
    def render(self, string):
        return get_rendered(string, self._ctx, self.model)

    @contextmember()
    def try_or_compiler_error(self, message_if_exception, func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            raise CompilationError(message_if_exception, self.model)

    @contextmember()
    def load_agate_table(self):
        from dbt_common.clients import agate_helper
        if not isinstance(self.model, SeedNode):
            raise LoadAgateTableNotSeedError(self.model.resource_type, node=self.model)
        package_path = os.path.join(self.config.packages_install_path, self.model.package_name) if self.model.package_name != self.config.project_name else '.'
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
        table.original_abspath = os.path.abspath(path)
        return table

    @contextproperty()
    def ref(self):
        """The most important function in dbt is `ref()`; it's impossible to
        build even moderately complex models without it. `ref()` is how you
        reference one model within another. This is a very common behavior, as
        typically models are built to be "stacked" on top of one another. Here
        is how this looks in practice:

        > model_a.sql:

            select *
            from public.raw_data

        > model_b.sql:

            select *
            from {{ref('model_a')}}


        `ref()` is, under the hood, actually doing two important things. First,
        it is interpolating the schema into your model file to allow you to
        change your deployment schema via configuration. Second, it is using
        these references between models to automatically build the dependency
        graph. This will enable dbt to deploy models in the correct order when
        using dbt run.

        The `ref` function returns a Relation object.

        ## Advanced ref usage

        There is also a two-argument variant of the `ref` function. With this
        variant, you can pass both a package name and model name to `ref` to
        avoid ambiguity. This functionality is not commonly required for
        typical dbt usage.

        > model.sql:

            select * from {{ ref('package_name', 'model_name') }}"
        """
        return self.provider.ref(self.db_wrapper, self.model, self.config, self.manifest)

    @contextproperty()
    def source(self):
        return self.provider.source(self.db_wrapper, self.model, self.config, self.manifest)

    @contextproperty()
    def metric(self):
        return self.provider.metric(self.db_wrapper, self.model, self.config, self.manifest)

    @contextproperty('config')
    def ctx_config(self):
        """The `config` variable exists to handle end-user configuration for
        custom materializations. Configs like `unique_key` can be implemented
        using the `config` variable in your own materializations.

        For example, code in the `incremental` materialization like this:

            {% materialization incremental, default -%}
            {%- set unique_key = config.get('unique_key') -%}
            ...

        is responsible for handling model code that looks like this:

            {{
              config(
                materialized='incremental',
                unique_key='id'
              )
            }}


        ## config.get

        name: The name of the configuration variable (required)
        default: The default value to use if this configuration is not provided
            (optional)

        The `config.get` function is used to get configurations for a model
        from the end-user. Configs defined in this way are optional, and a
        default value can be provided.

        Example usage:

            {% materialization incremental, default -%}
              -- Example w/ no default. unique_key will be None if the user does not provide this configuration
              {%- set unique_key = config.get('unique_key') -%}
              -- Example w/ default value. Default to 'id' if 'unique_key' not provided
              {%- set unique_key = config.get('unique_key', default='id') -%}
              ...

        ## config.require

        name: The name of the configuration variable (required)

        The `config.require` function is used to get configurations for a model
        from the end-user. Configs defined using this function are required,
        and failure to provide them will result in a compilation error.

        Example usage:

            {% materialization incremental, default -%}
              {%- set unique_key = config.require('unique_key') -%}
              ...
        """
        return self.provider.Config(self.model, self.context_config)

    @contextproperty()
    def execute(self):
        """`execute` is a Jinja variable that returns True when dbt is in
        "execute" mode.

        When you execute a dbt compile or dbt run command, dbt:

        - Reads all of the files in your project and generates a "manifest"
            comprised of models, tests, and other graph nodes present in your
            project. During this phase, dbt uses the `ref` statements it finds
            to generate the DAG for your project. *No SQL is run during this
            phase*, and `execute == False`.
        - Compiles (and runs) each node (eg. building models, or running
            tests). SQL is run during this phase, and `execute == True`.

        Any Jinja that relies on a result being returned from the database will
        error during the parse phase. For example, this SQL will return an
        error:

        > models/order_payment_methods.sql:

            {% set payment_method_query %}
            select distinct
            payment_method
            from {{ ref('raw_payments') }}
            order by 1
            {% endset %}
            {% set results = run_query(relation_query) %}
            {# Return the first column #}
            {% set payment_methods = results.columns[0].values() %}

        The error returned by dbt will look as follows:

            Encountered an error:
                Compilation Error in model order_payment_methods (models/order_payment_methods.sql)
            'None' has no attribute 'table'

        This is because Line #11 assumes that a table has been returned, when,
        during the parse phase, this query hasn't been run.

        To work around this, wrap any problematic Jinja in an
        `{% if execute %}` statement:

        > models/order_payment_methods.sql:

            {% set payment_method_query %}
            select distinct
            payment_method
            from {{ ref('raw_payments') }}
            order by 1
            {% endset %}
            {% set results = run_query(relation_query) %}
            {% if execute %}
            {# Return the first column #}
            {% set payment_methods = results.columns[0].values() %}
            {% else %}
            {% set payment_methods = [] %}
            {% endif %}
        """
        return self.provider.execute

    @contextproperty()
    def exceptions(self):
        """The exceptions namespace can be used to raise warnings and errors in
        dbt userspace.


        ## raise_compiler_error

        The `exceptions.raise_compiler_error` method will raise a compiler
        error with the provided message. This is typically only useful in
        macros or materializations when invalid arguments are provided by the
        calling model. Note that throwing an exception will cause a model to
        fail, so please use this variable with care!

        Example usage:

        > exceptions.sql:

            {% if number < 0 or number > 100 %}
              {{ exceptions.raise_compiler_error("Invalid `number`. Got: " ~ number) }}
            {% endif %}

        ## warn

        The `exceptions.warn` method will raise a compiler warning with the
        provided message. If the `--warn-error` flag is provided to dbt, then
        this warning will be elevated to an exception, which is raised.

        Example usage:

        > warn.sql:

            {% if number < 0 or number > 100 %}
              {% do exceptions.warn("Invalid `number`. Got: " ~ number) %}
            {% endif %}
        """
        return wrapped_exports(self.model)

    @contextproperty()
    def database(self):
        return self.config.credentials.database

    @contextproperty()
    def schema(self):
        return self.config.credentials.schema

    @contextproperty()
    def var(self):
        return self.provider.Var(context=self._ctx, config=self.config, node=self.model)

    @contextproperty('adapter')
    def ctx_adapter(self):
        """`adapter` is a wrapper around the internal database adapter used by
        dbt. It allows users to make calls to the database in their dbt models.
        The adapter methods will be translated into specific SQL statements
        depending on the type of adapter your project is using.
        """
        return self.db_wrapper

    @contextproperty()
    def api(self):
        return {'Relation': self.db_wrapper.Relation, 'Column': self.adapter.Column}

    @contextproperty()
    def column(self):
        return self.adapter.Column

    @contextproperty()
    def env(self):
        return self.target

    @contextproperty()
    def graph(self):
        """The `graph` context variable contains information about the nodes in
        your dbt project. Models, sources, tests, and snapshots are all
        examples of nodes in dbt projects.

        ## The graph context variable

        The graph context variable is a dictionary which maps node ids onto dictionary representations of those nodes. A simplified example might look like:

            {
              "model.project_name.model_name": {
                "config": {"materialzed": "table", "sort": "id"},
                "tags": ["abc", "123"],
                "path": "models/path/to/model_name.sql",
                ...
              },
              "source.project_name.source_name": {
                "path": "models/path/to/schema.yml",
                "columns": {
                  "id": { .... },
                  "first_name": { .... },
                },
                ...
              }
            }

        The exact contract for these model and source nodes is not currently
        documented, but that will change in the future.

        ## Accessing models

        The `model` entries in the `graph` dictionary will be incomplete or
        incorrect during parsing. If accessing the models in your project via
        the `graph` variable, be sure to use the `execute` flag to ensure that
        this code only executes at run-time and not at parse-time. Do not use
        the `graph` variable to build you DAG, as the resulting dbt behavior
        will be undefined and likely incorrect.

        Example usage:

        > graph-usage.sql:

            /*
              Print information about all of the models in the Snowplow package
            */
            {% if execute %}
              {% for node in graph.nodes.values()
                 | selectattr("resource_type", "equalto", "model")
                 | selectattr("package_name", "equalto", "snowplow") %}

                {% do log(node.unique_id ~ ", materialized: " ~ node.config.materialized, info=true) %}

              {% endfor %}
            {% endif %}
            /*
              Example output
            ---------------------------------------------------------------
            model.snowplow.snowplow_id_map, materialized: incremental
            model.snowplow.snowplow_page_views, materialized: incremental
            model.snowplow.snowplow_web_events, materialized: incremental
            model.snowplow.snowplow_web_page_context, materialized: table
            model.snowplow.snowplow_web_events_scroll_depth, materialized: incremental
            model.snowplow.snowplow_web_events_time, materialized: incremental
            model.snowplow.snowplow_web_events_internal_fixed, materialized: ephemeral
            model.snowplow.snowplow_base_web_page_context, materialized: ephemeral
            model.snowplow.snowplow_base_events, materialized: ephemeral
            model.snowplow.snowplow_sessions_tmp, materialized: incremental
            model.snowplow.snowplow_sessions, materialized: table
            */

        ## Accessing sources

        To access the sources in your dbt project programatically, use the "sources" attribute.

        Example usage:

        > models/events_unioned.sql

            /*
              Union all of the Snowplow sources defined in the project
              which begin with the string "event_"
            */
            {% set sources = [] -%}
            {% for node in graph.sources.values() -%}
              {%- if node.name.startswith('event_') and node.source_name == 'snowplow' -%}
                {%- do sources.append(source(node.source_name, node.name)) -%}
              {%- endif -%}
            {%- endfor %}
            select * from (
              {%- for source in sources %}
                {{ source }} {% if not loop.last %} union all {% endif %}
              {% endfor %}
            )
            /*
              Example compiled SQL
            ---------------------------------------------------------------
            select * from (
              select * from raw.snowplow.event_add_to_cart union all
              select * from raw.snowplow.event_remove_from_cart union all
              select * from raw.snowplow.event_checkout
            )
            */

        """
        return self.manifest.flat_graph

    @contextproperty('model')
    def ctx_model(self):
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
    def pre_hooks(self):
        return None

    @contextproperty()
    def post_hooks(self):
        return None

    @contextproperty()
    def sql(self):
        return None

    @contextproperty()
    def sql_now(self):
        return self.adapter.date_function()

    @contextmember()
    def adapter_macro(self, name, *args, **kwargs):
        """This was deprecated in v0.18 in favor of adapter.dispatch"""
        msg = 'The "adapter_macro" macro has been deprecated. Instead, use the `adapter.dispatch` method to find a macro and call the result.  For more information, see: https://docs.getdbt.com/reference/dbt-jinja-functions/dispatch) adapter_macro was called for: {macro_name}'.format(macro_name=name)
        raise CompilationError(msg)

    @contextmember()
    def env_var(self, var, default=None):
        """The env_var() function. Return the environment variable named 'var'.
        If there is no such environment variable set, return the default.

        If the default is None, raise an exception for an undefined variable.
        """
        return_value = None
        if var.startswith(SECRET_ENV_PREFIX):
            raise SecretEnvVarLocationError(var)
        env = get_invocation_context().env
        if var in env:
            return_value = env[var]
        elif default is not None:
            return_value = default
        if return_value is not None:
            compiling = True if hasattr(self.model, 'compiled') and getattr(self.model, 'compiled', False) is True else False
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
    def selected_resources(self):
        """The `selected_resources` variable contains a list of the resources
        selected based on the parameters provided to the dbt command.
        Currently, is not populated for the command `run-operation` that
        doesn't support `--select`.
        """
        return selected_resources.SELECTED_RESOURCES

    @contextmember()
    def submit_python_job(self, parsed_model, compiled_code):
        if not (self.context_macro_stack.depth == 2 and self.context_macro_stack.call_stack[1] == 'macro.dbt.statement' and ('materialization' in self.context_macro_stack.call_stack[0])):
            raise DbtRuntimeError(f'submit_python_job is not intended to be called here, at model {parsed_model['alias']}, with macro call_stack {self.context_macro_stack.call_stack}.')
        return self.adapter.submit_python_job(parsed_model, compiled_code)

class MacroContext(ProviderContext):
    """Internally, macros can be executed like nodes, with some restrictions:

    - they don't have all values available that nodes do:
       - 'this', 'pre_hooks', 'post_hooks', and 'sql' are missing
       - 'schema' does not use any 'model' information
    - they can't be configured with config() directives
    """

    def __init__(self, model, config, manifest, provider, search_package):
        super().__init__(model, config, manifest, provider, None)
        if search_package is None:
            self._search_package = config.project_name
        else:
            self._search_package = search_package

class SourceContext(ProviderContext):

    @contextproperty()
    def this(self):
        return self.db_wrapper.Relation.create_from(self.config, self.model)

    @contextproperty()
    def source_node(self):
        return self.model

class ModelContext(ProviderContext):

    @contextproperty()
    def pre_hooks(self):
        if self.model.resource_type in [NodeType.Source, NodeType.Test, NodeType.Unit]:
            return []
        return [h.to_dict(omit_none=True) for h in self.model.config.pre_hook]

    @contextproperty()
    def post_hooks(self):
        if self.model.resource_type in [NodeType.Source, NodeType.Test, NodeType.Unit]:
            return []
        return [h.to_dict(omit_none=True) for h in self.model.config.post_hook]

    @contextproperty()
    def compiled_code(self):
        if getattr(self.model, 'defer_relation', None) and self.config.args.which == 'clone':
            return f'select * from {self.model.defer_relation.relation_name or str(self.defer_relation)}'
        elif getattr(self.model, 'extra_ctes_injected', None):
            return self.model.compiled_code
        else:
            return None

    @contextproperty()
    def sql(self):
        if self.model.language == ModelLanguage.sql:
            return self.compiled_code
        else:
            return None

    @contextproperty()
    def database(self):
        return getattr(self.model, 'database', self.config.credentials.database)

    @contextproperty()
    def schema(self):
        return getattr(self.model, 'schema', self.config.credentials.schema)

    @contextproperty()
    def this(self):
        """`this` makes available schema information about the currently
        executing model. It's is useful in any context in which you need to
        write code that references the current model, for example when defining
        a `sql_where` clause for an incremental model and for writing pre- and
        post-model hooks that operate on the model in some way. Developers have
        options for how to use `this`:

            |------------------|------------------|
            | dbt Model Syntax | Output           |
            |------------------|------------------|
            |     {{this}}     | "schema"."table" |
            |------------------|------------------|
            |  {{this.schema}} | schema           |
            |------------------|------------------|
            |  {{this.table}}  | table            |
            |------------------|------------------|
            |  {{this.name}}   | table            |
            |------------------|------------------|

        Here's an example of how to use `this` in `dbt_project.yml` to grant
        select rights on a table to a different db user.

        > example.yml:

            models:
              project-name:
                post-hook:
                  - "grant select on {{ this }} to db_reader"
        """
        if self.model.resource_type == NodeType.Operation:
            return None
        return self.db_wrapper.Relation.create_from(self.config, self.model)

    @contextproperty()
    def defer_relation(self):
        """
        For commands which add information about this node's corresponding
        production version (via a --state artifact), access the Relation
        object for that stateful other
        """
        if getattr(self.model, 'defer_relation', None):
            return self.db_wrapper.Relation.create_from(self.config, self.model.defer_relation)
        else:
            return None

class UnitTestContext(ModelContext):

    @contextmember()
    def env_var(self, var, default=None):
        """The env_var() function. Return the overriden unit test environment variable named 'var'.

        If there is no unit test override, return the environment variable named 'var'.

        If there is no such environment variable set, return the default.

        If the default is None, raise an exception for an undefined variable.
        """
        if self.model.overrides and var in self.model.overrides.env_vars:
            return self.model.overrides.env_vars[var]
        else:
            return super().env_var(var, default)

    @contextproperty()
    def this(self):
        if self.model.this_input_node_unique_id:
            this_node = self.manifest.expect(self.model.this_input_node_unique_id)
            self.model.set_cte(this_node.unique_id, None)
            return self.adapter.Relation.add_ephemeral_prefix(this_node.identifier)
        return None

def generate_parser_model_context(model, config, manifest, context_config):
    ctx = ModelContext(model, config, manifest, ParseProvider(), context_config)
    return ctx.to_dict()

def generate_generate_name_macro_context(macro, config, manifest):
    ctx = MacroContext(macro, config, manifest, GenerateNameProvider(), None)
    return ctx.to_dict()

def generate_runtime_model_context(model, config, manifest):
    ctx = ModelContext(model, config, manifest, RuntimeProvider(), None)
    return ctx.to_dict()

def generate_runtime_macro_context(macro, config, manifest, package_name):
    ctx = MacroContext(macro, config, manifest, OperationProvider(), package_name)
    return ctx.to_dict()

def generate_runtime_unit_test_context(unit_test, config, manifest):
    ctx = UnitTestContext(unit_test, config, manifest, RuntimeUnitTestProvider(), None)
    ctx_dict = ctx.to_dict()
    if unit_test.overrides and unit_test.overrides.macros:
        global_macro_overrides = {}
        package_macro_overrides = {}
        for macro_name, macro_value in unit_test.overrides.macros.items():
            macro_name_split = macro_name.split('.')
            macro_package = macro_name_split[0] if len(macro_name_split) == 2 else None
            macro_name = macro_name_split[-1]
            if macro_package is None and macro_name in ctx_dict:
                original_context_value = ctx_dict[macro_name]
                if isinstance(original_context_value, MacroGenerator):
                    macro_value = UnitTestMacroGenerator(original_context_value, macro_value)
                global_macro_overrides[macro_name] = macro_value
            elif macro_package and macro_package in ctx_dict and (macro_name in ctx_dict[macro_package]):
                original_context_value = ctx_dict[macro_package][macro_name]
                if isinstance(original_context_value, MacroGenerator):
                    macro_value = UnitTestMacroGenerator(original_context_value, macro_value)
                package_macro_overrides[macro_package, macro_name] = macro_value
        for (macro_package, macro_name), macro_override_value in package_macro_overrides.items():
            ctx_dict[macro_package][macro_name] = macro_override_value
            if macro_package == 'dbt':
                ctx_dict[macro_name] = macro_value
        for macro_name, macro_override_value in global_macro_overrides.items():
            ctx_dict[macro_name] = macro_override_value
            if ctx_dict['dbt'].get(macro_name):
                ctx_dict['dbt'][macro_name] = macro_override_value
    return ctx_dict

class ExposureRefResolver(BaseResolver):

    def __call__(self, *args, **kwargs):
        package = None
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

    def __call__(self, *args):
        if len(args) != 2:
            raise NumberSourceArgsError(args, node=self.model)
        self.model.sources.append(list(args))
        return ''

class ExposureMetricResolver(BaseResolver):

    def __call__(self, *args):
        if len(args) not in (1, 2):
            raise MetricArgsError(node=self.model, args=args)
        self.model.metrics.append(list(args))
        return ''

def generate_parse_exposure(exposure, config, manifest, package_name):
    project = config.load_dependencies()[package_name]
    return {'ref': ExposureRefResolver(None, exposure, project, manifest), 'source': ExposureSourceResolver(None, exposure, project, manifest), 'metric': ExposureMetricResolver(None, exposure, project, manifest)}

class SemanticModelRefResolver(BaseResolver):

    def __call__(self, *args, **kwargs):
        package = None
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

    def validate_args(self, name, package, version):
        if not isinstance(name, str):
            raise ParsingError(f'In a semantic model or metrics section in {self.model.original_file_path} the name argument to ref() must be a string')

def generate_parse_semantic_models(semantic_model, config, manifest, package_name):
    project = config.load_dependencies()[package_name]
    return {'ref': SemanticModelRefResolver(None, semantic_model, project, manifest)}

class TestContext(ProviderContext):

    def __init__(self, model, config, manifest, provider, context_config, macro_resolver):
        self.macro_resolver = macro_resolver
        self.thread_ctx = MacroStack()
        super().__init__(model, config, manifest, provider, context_config)
        self._build_test_namespace()
        self.db_wrapper = self.provider.DatabaseWrapper(self.adapter, self.namespace)

    def _build_namespace(self):
        return {}

    def _build_test_namespace(self):
        depends_on_macros = []
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
    def env_var(self, var, default=None):
        return_value = None
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

def generate_test_context(model, config, manifest, context_config, macro_resolver):
    ctx = TestContext(model, config, manifest, ParseProvider(), context_config, macro_resolver)
    return ctx.to_dict()