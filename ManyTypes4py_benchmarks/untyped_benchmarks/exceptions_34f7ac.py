import io
import json
import re
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Union
from dbt.node_types import REFABLE_NODE_TYPES, AccessType, NodeType
from dbt_common.constants import SECRET_ENV_PREFIX
from dbt_common.dataclass_schema import ValidationError
from dbt_common.exceptions import CommandResultError, CompilationError, DbtConfigError, DbtInternalError, DbtRuntimeError, DbtValidationError, env_secrets, scrub_secrets
if TYPE_CHECKING:
    import agate

class ContractBreakingChangeError(DbtRuntimeError):
    CODE = 10016
    MESSAGE = 'Breaking Change to Contract'

    def __init__(self, breaking_changes, node=None):
        self.breaking_changes = breaking_changes
        super().__init__(self.message(), node)

    @property
    def type(self):
        return 'Breaking change to contract'

    def message(self):
        reasons = '\n  - '.join(self.breaking_changes)
        return f'While comparing to previous project state, dbt detected a breaking change to an enforced contract.\n  - {reasons}\nConsider making an additive (non-breaking) change instead, if possible.\nOtherwise, create a new model version: https://docs.getdbt.com/docs/collaborate/govern/model-versions'

class ParsingError(DbtRuntimeError):
    CODE = 10015
    MESSAGE = 'Parsing Error'

    @property
    def type(self):
        return 'Parsing'

class dbtPluginError(DbtRuntimeError):
    CODE = 10020
    MESSAGE = 'Plugin Error'

class JSONValidationError(DbtValidationError):

    def __init__(self, typename, errors):
        self.typename = typename
        self.errors = errors
        self.errors_message = ', '.join(errors)
        msg = f'Invalid arguments passed to "{self.typename}" instance: {self.errors_message}'
        super().__init__(msg)

    def __reduce__(self):
        return (JSONValidationError, (self.typename, self.errors))

class AliasError(DbtValidationError):
    pass

class DependencyError(Exception):
    CODE = 10006
    MESSAGE = 'Dependency Error'

class FailFastError(DbtRuntimeError):
    CODE = 10013
    MESSAGE = 'FailFast Error'

    def __init__(self, msg, result=None, node=None):
        super().__init__(msg=msg, node=node)
        self.result = result

    @property
    def type(self):
        return 'FailFast'

class DbtProjectError(DbtConfigError):
    pass

class DbtSelectorsError(DbtConfigError):
    pass

class DbtProfileError(DbtConfigError):
    pass

class DbtExclusivePropertyUseError(DbtConfigError):
    pass

class InvalidSelectorError(DbtRuntimeError):

    def __init__(self, name):
        self.name = name
        super().__init__(name)

class DuplicateYamlKeyError(CompilationError):
    pass

class GraphDependencyNotFoundError(CompilationError):

    def __init__(self, node, dependency):
        self.node = node
        self.dependency = dependency
        super().__init__(msg=self.get_message())

    def get_message(self):
        msg = f"'{self.node.unique_id}' depends on '{self.dependency}' which is not in the graph!"
        return msg

class ForeignKeyConstraintToSyntaxError(CompilationError):

    def __init__(self, node, expression):
        self.expression = expression
        self.node = node
        super().__init__(msg=self.get_message())

    def get_message(self):
        msg = f"'{self.node.unique_id}' defines a foreign key constraint 'to' expression which is not valid 'ref' or 'source' syntax: {self.expression}."
        return msg

class NoSupportedLanguagesFoundError(CompilationError):

    def __init__(self, node):
        self.node = node
        self.msg = f'No supported_languages found in materialization macro {self.node.name}'
        super().__init__(msg=self.msg)

class MaterializtionMacroNotUsedError(CompilationError):

    def __init__(self, node):
        self.node = node
        self.msg = 'Only materialization macros can be used with this function'
        super().__init__(msg=self.msg)

class MacroNamespaceNotStringError(CompilationError):

    def __init__(self, kwarg_type):
        self.kwarg_type = kwarg_type
        super().__init__(msg=self.get_message())

    def get_message(self):
        msg = f'The macro_namespace parameter to adapter.dispatch is a {self.kwarg_type}, not a string'
        return msg

class UnknownGitCloningProblemError(DbtRuntimeError):

    def __init__(self, repo):
        self.repo = scrub_secrets(repo, env_secrets())
        super().__init__(msg=self.get_message())

    def get_message(self):
        msg = f'        Something went wrong while cloning {self.repo}\n        Check the debug logs for more information\n        '
        return msg

class NoAdaptersAvailableError(DbtRuntimeError):

    def __init__(self):
        super().__init__(msg=self.get_message())

    def get_message(self):
        msg = 'No adapters available. Learn how to install an adapter by going to https://docs.getdbt.com/docs/connect-adapters#install-using-the-cli'
        return msg

class BadSpecError(DbtInternalError):

    def __init__(self, repo, revision, error):
        self.repo = repo
        self.revision = revision
        self.stderr = scrub_secrets(error.stderr.strip(), env_secrets())
        super().__init__(msg=self.get_message())

    def get_message(self):
        msg = f"Error checking out spec='{self.revision}' for repo {self.repo}\n{self.stderr}"
        return msg

class GitCloningError(DbtInternalError):

    def __init__(self, repo, revision, error):
        self.repo = repo
        self.revision = revision
        self.error = error
        super().__init__(msg=self.get_message())

    def get_message(self):
        stderr = self.error.stderr.strip()
        if 'usage: git' in stderr:
            stderr = stderr.split('\nusage: git')[0]
        if re.match("fatal: destination path '(.+)' already exists", stderr):
            self.error.cmd = list(scrub_secrets(str(self.error.cmd), env_secrets()))
            raise self.error
        msg = f"Error checking out spec='{self.revision}' for repo {self.repo}\n{stderr}"
        return scrub_secrets(msg, env_secrets())

class GitCheckoutError(BadSpecError):
    pass

class OperationError(CompilationError):

    def __init__(self, operation_name):
        self.operation_name = operation_name
        super().__init__(msg=self.get_message())

    def get_message(self):
        msg = f'dbt encountered an error when attempting to create a {self.operation_name}. If this error persists, please create an issue at: \n\nhttps://github.com/dbt-labs/dbt-core'
        return msg

class ZipStrictWrongTypeError(CompilationError):

    def __init__(self, exc):
        self.exc = exc
        msg = str(self.exc)
        super().__init__(msg=msg)

class SetStrictWrongTypeError(CompilationError):

    def __init__(self, exc):
        self.exc = exc
        msg = str(self.exc)
        super().__init__(msg=msg)

class LoadAgateTableValueError(CompilationError):

    def __init__(self, exc, node):
        self.exc = exc
        self.node = node
        msg = str(self.exc)
        super().__init__(msg=msg)

class LoadAgateTableNotSeedError(CompilationError):

    def __init__(self, resource_type, node):
        self.resource_type = resource_type
        self.node = node
        msg = f'can only load_agate_table for seeds (got a {self.resource_type})'
        super().__init__(msg=msg)

class PackageNotInDepsError(CompilationError):

    def __init__(self, package_name, node):
        self.package_name = package_name
        self.node = node
        msg = f'Node package named {self.package_name} not found!'
        super().__init__(msg=msg)

class OperationsCannotRefEphemeralNodesError(CompilationError):

    def __init__(self, target_name, node):
        self.target_name = target_name
        self.node = node
        msg = f'Operations can not ref() ephemeral nodes, but {target_name} is ephemeral'
        super().__init__(msg=msg)

class PersistDocsValueTypeError(CompilationError):

    def __init__(self, persist_docs):
        self.persist_docs = persist_docs
        msg = f"Invalid value provided for 'persist_docs'. Expected dict but received {type(self.persist_docs)}"
        super().__init__(msg=msg)

class InlineModelConfigError(CompilationError):

    def __init__(self, node):
        self.node = node
        msg = 'Invalid inline model config'
        super().__init__(msg=msg)

class ConflictingConfigKeysError(CompilationError):

    def __init__(self, oldkey, newkey, node):
        self.oldkey = oldkey
        self.newkey = newkey
        self.node = node
        msg = f'Invalid config, has conflicting keys "{self.oldkey}" and "{self.newkey}"'
        super().__init__(msg=msg)

class NumberSourceArgsError(CompilationError):

    def __init__(self, args, node):
        self.args = args
        self.node = node
        msg = f'source() takes exactly two arguments ({len(self.args)} given)'
        super().__init__(msg=msg)

class RequiredVarNotFoundError(CompilationError):

    def __init__(self, var_name, merged, node):
        self.var_name = var_name
        self.merged = merged
        self.node = node
        super().__init__(msg=self.get_message())

    def get_message(self):
        if self.node is not None:
            node_name = self.node.name
        else:
            node_name = '<Configuration>'
        dct = {k: self.merged[k] for k in self.merged}
        pretty_vars = json.dumps(dct, sort_keys=True, indent=4)
        msg = f"Required var '{self.var_name}' not found in config:\nVars supplied to {node_name} = {pretty_vars}"
        return scrub_secrets(msg, self.var_secrets())

    def var_secrets(self):
        return [v for k, v in self.merged.items() if k.startswith(SECRET_ENV_PREFIX) and v.strip()]

class PackageNotFoundForMacroError(CompilationError):

    def __init__(self, package_name):
        self.package_name = package_name
        msg = f"Could not find package '{self.package_name}'"
        super().__init__(msg=msg)

class SecretEnvVarLocationError(ParsingError):

    def __init__(self, env_var_name):
        self.env_var_name = env_var_name
        super().__init__(msg=self.get_message())

    def get_message(self):
        msg = f"Secret env vars are allowed only in profiles.yml or packages.yml. Found '{self.env_var_name}' referenced elsewhere."
        return msg

class BooleanError(CompilationError):

    def __init__(self, return_value, macro_name):
        self.return_value = return_value
        self.macro_name = macro_name
        super().__init__(msg=self.get_message())

    def get_message(self):
        msg = f"Macro '{self.macro_name}' returns '{self.return_value}'.  It is not type 'bool' and cannot not be converted reliably to a bool."
        return msg

class RefArgsError(CompilationError):

    def __init__(self, node, args):
        self.node = node
        self.args = args
        super().__init__(msg=self.get_message())

    def get_message(self):
        msg = f'ref() takes at most two arguments ({len(self.args)} given)'
        return msg

class MetricArgsError(CompilationError):

    def __init__(self, node, args):
        self.node = node
        self.args = args
        super().__init__(msg=self.get_message())

    def get_message(self):
        msg = f'metric() takes at most two arguments ({len(self.args)} given)'
        return msg

class RefBadContextError(CompilationError):

    def __init__(self, node, args):
        self.node = node
        self.args = args.positional_args
        self.kwargs = args.keyword_args
        super().__init__(msg=self.get_message())

    def get_message(self):
        if isinstance(self.node, dict):
            model_name = self.node['name']
        else:
            model_name = self.node.name
        ref_args = ', '.join(("'{}'".format(a) for a in self.args))
        keyword_args = ''
        if self.kwargs:
            keyword_args = ', '.join(("{}='{}'".format(k, v) for k, v in self.kwargs.items()))
            keyword_args = ',' + keyword_args
        ref_string = f'{{{{ ref({ref_args}{keyword_args}) }}}}'
        msg = f'dbt was unable to infer all dependencies for the model "{model_name}".\nThis typically happens when ref() is placed within a conditional block.\n\nTo fix this, add the following hint to the top of the model "{model_name}":\n\n-- depends_on: {ref_string}'
        return msg

class DocArgsError(CompilationError):

    def __init__(self, node, args):
        self.node = node
        self.args = args
        super().__init__(msg=self.get_message())

    def get_message(self):
        msg = f'doc() takes at most two arguments ({len(self.args)} given)'
        return msg

class DocTargetNotFoundError(CompilationError):

    def __init__(self, node, target_doc_name, target_doc_package=None):
        self.node = node
        self.target_doc_name = target_doc_name
        self.target_doc_package = target_doc_package
        super().__init__(msg=self.get_message())

    def get_message(self):
        target_package_string = ''
        if self.target_doc_package is not None:
            target_package_string = f"in package '{self.target_doc_package}' "
        msg = f"Documentation for '{self.node.unique_id}' depends on doc '{self.target_doc_name}' {target_package_string} which was not found"
        return msg

class MacroDispatchArgError(CompilationError):

    def __init__(self, macro_name):
        self.macro_name = macro_name
        super().__init__(msg=self.get_message())

    def get_message(self):
        msg = f'        The "packages" argument of adapter.dispatch() has been deprecated.\n        Use the "macro_namespace" argument instead.\n\n        Raised during dispatch for: {self.macro_name}\n\n        For more information, see:\n\n        https://docs.getdbt.com/reference/dbt-jinja-functions/dispatch\n        '
        return msg

class DuplicateMacroNameError(CompilationError):

    def __init__(self, node_1, node_2, namespace):
        self.node_1 = node_1
        self.node_2 = node_2
        self.namespace = namespace
        super().__init__(msg=self.get_message())

    def get_message(self):
        duped_name = self.node_1.name
        if self.node_1.package_name != self.node_2.package_name:
            extra = f' ("{self.node_1.package_name}" and "{self.node_2.package_name}" are both in the "{self.namespace}" namespace)'
        else:
            extra = ''
        msg = f'dbt found two macros with the name "{duped_name}" in the namespace "{self.namespace}"{extra}. Since these macros have the same name and exist in the same namespace, dbt will be unable to decide which to call. To fix this, change the name of one of these macros:\n- {self.node_1.unique_id} ({self.node_1.original_file_path})\n- {self.node_2.unique_id} ({self.node_2.original_file_path})'
        return msg

class MacroResultAlreadyLoadedError(CompilationError):

    def __init__(self, result_name):
        self.result_name = result_name
        super().__init__(msg=self.get_message())

    def get_message(self):
        msg = f"The 'statement' result named '{self.result_name}' has already been loaded into a variable"
        return msg

class DictParseError(ParsingError):

    def __init__(self, exc, node):
        self.exc = exc
        self.node = node
        msg = self.validator_error_message(exc)
        super().__init__(msg=msg)

class ConfigUpdateError(ParsingError):

    def __init__(self, exc, node):
        self.exc = exc
        self.node = node
        msg = self.validator_error_message(exc)
        super().__init__(msg=msg)

class PythonParsingError(ParsingError):

    def __init__(self, exc, node):
        self.exc = exc
        self.node = node
        super().__init__(msg=self.get_message())

    def get_message(self):
        validated_exc = self.validator_error_message(self.exc)
        msg = f'{validated_exc}\n{self.exc.text}'
        return msg

class PythonLiteralEvalError(ParsingError):

    def __init__(self, exc, node):
        self.exc = exc
        self.node = node
        super().__init__(msg=self.get_message())

    def get_message(self):
        msg = f'Error when trying to literal_eval an arg to dbt.ref(), dbt.source(), dbt.config() or dbt.config.get() \n{self.exc}\nhttps://docs.python.org/3/library/ast.html#ast.literal_eval\nIn dbt python model, `dbt.ref`, `dbt.source`, `dbt.config`, `dbt.config.get` function args only support Python literal structures'
        return msg

class ModelConfigError(ParsingError):

    def __init__(self, exc, node):
        self.msg = self.validator_error_message(exc)
        self.node = node
        super().__init__(msg=self.msg)

class YamlParseListError(ParsingError):

    def __init__(self, path, key, yaml_data, cause):
        self.path = path
        self.key = key
        self.yaml_data = yaml_data
        self.cause = cause
        super().__init__(msg=self.get_message())

    def get_message(self):
        if isinstance(self.cause, str):
            reason = self.cause
        elif isinstance(self.cause, ValidationError):
            reason = self.validator_error_message(self.cause)
        else:
            reason = self.cause.msg
        msg = f'Invalid {self.key} config given in {self.path} @ {self.key}: {self.yaml_data} - {reason}'
        return msg

class YamlParseDictError(ParsingError):

    def __init__(self, path, key, yaml_data, cause):
        self.path = path
        self.key = key
        self.yaml_data = yaml_data
        self.cause = cause
        super().__init__(msg=self.get_message())

    def get_message(self):
        if isinstance(self.cause, str):
            reason = self.cause
        elif isinstance(self.cause, ValidationError):
            reason = self.validator_error_message(self.cause)
        else:
            reason = self.cause.msg
        msg = f'Invalid {self.key} config given in {self.path} @ {self.key}: {self.yaml_data} - {reason}'
        return msg

class YamlLoadError(ParsingError):

    def __init__(self, path, exc, project_name=None):
        self.project_name = project_name
        self.path = path
        self.exc = exc
        super().__init__(msg=self.get_message())

    def get_message(self):
        reason = self.validator_error_message(self.exc)
        msg = f'Error reading {self.project_name}: {self.path} - {reason}'
        return msg

class TestConfigError(ParsingError):

    def __init__(self, exc, node):
        self.msg = self.validator_error_message(exc)
        self.node = node
        super().__init__(msg=self.msg)

class SchemaConfigError(ParsingError):

    def __init__(self, exc, node):
        self.msg = self.validator_error_message(exc)
        self.node = node
        super().__init__(msg=self.msg)

class SnapshopConfigError(ParsingError):

    def __init__(self, exc, node):
        self.msg = self.validator_error_message(exc)
        self.node = node
        super().__init__(msg=self.msg)

class DbtReferenceError(ParsingError):

    def __init__(self, unique_id, ref_unique_id, access, scope):
        self.unique_id = unique_id
        self.ref_unique_id = ref_unique_id
        self.access = access
        self.scope = scope
        self.scope_type = 'group' if self.access == AccessType.Private else 'package'
        super().__init__(msg=self.get_message())

    def get_message(self):
        return f"Node {self.unique_id} attempted to reference node {self.ref_unique_id}, which is not allowed because the referenced node is {self.access} to the '{self.scope}' {self.scope_type}."

class InvalidAccessTypeError(ParsingError):

    def __init__(self, unique_id, field_value, materialization=None):
        self.unique_id = unique_id
        self.field_value = field_value
        self.materialization = materialization
        with_materialization = f"with '{self.materialization}' materialization " if self.materialization else ''
        msg = f'Node {self.unique_id} {with_materialization}has an invalid value ({self.field_value}) for the access field'
        super().__init__(msg=msg)

class InvalidUnitTestGivenInput(ParsingError):

    def __init__(self, input):
        msg = f"Unit test given inputs must be either a 'ref', 'source' or 'this' call. Got: '{input}'."
        super().__init__(msg=msg)

class SameKeyNestedError(CompilationError):

    def __init__(self):
        msg = 'Test cannot have the same key at the top-level and in config'
        super().__init__(msg=msg)

class TestArgIncludesModelError(CompilationError):

    def __init__(self):
        msg = 'Test arguments include "model", which is a reserved argument'
        super().__init__(msg=msg)

class UnexpectedTestNamePatternError(CompilationError):

    def __init__(self, test_name):
        self.test_name = test_name
        msg = f'Test name string did not match expected pattern: {self.test_name}'
        super().__init__(msg=msg)

class CustomMacroPopulatingConfigValueError(CompilationError):

    def __init__(self, target_name, name, key, err_msg, column_name=None):
        self.target_name = target_name
        self.column_name = column_name
        self.name = name
        self.key = key
        self.err_msg = err_msg
        super().__init__(msg=self.get_message())

    def get_message(self):
        msg = f'''The {self.target_name}.{self.column_name} column's "{self.name}" test references an undefined macro in its {self.key} configuration argument. The macro {self.err_msg}.\nPlease note that the generic test configuration parser currently does not support using custom macros to populate configuration values'''
        return msg

class TagsNotListOfStringsError(CompilationError):

    def __init__(self, tags):
        self.tags = tags
        msg = f'got {self.tags} ({type(self.tags)}) for tags, expected a list of strings'
        super().__init__(msg=msg)

class TagNotStringError(CompilationError):

    def __init__(self, tag):
        self.tag = tag
        msg = f'got {self.tag} ({type(self.tag)}) for tag, expected a str'
        super().__init__(msg=msg)

class TestNameNotStringError(ParsingError):

    def __init__(self, test_name):
        self.test_name = test_name
        super().__init__(msg=self.get_message())

    def get_message(self):
        msg = f'test name must be a str, got {type(self.test_name)} (value {self.test_name})'
        return msg

class TestArgsNotDictError(ParsingError):

    def __init__(self, test_args):
        self.test_args = test_args
        super().__init__(msg=self.get_message())

    def get_message(self):
        msg = f'test arguments must be a dict, got {type(self.test_args)} (value {self.test_args})'
        return msg

class TestDefinitionDictLengthError(ParsingError):

    def __init__(self, test):
        self.test = test
        super().__init__(msg=self.get_message())

    def get_message(self):
        msg = f'test definition dictionary must have exactly one key, got {self.test} instead ({len(self.test)} keys)'
        return msg

class TestTypeError(ParsingError):

    def __init__(self, test):
        self.test = test
        super().__init__(msg=self.get_message())

    def get_message(self):
        msg = f'test must be dict or str, got {type(self.test)} (value {self.test})'
        return msg

class EnvVarMissingError(ParsingError):

    def __init__(self, var):
        self.var = var
        super().__init__(msg=self.get_message())

    def get_message(self):
        msg = f"Env var required but not provided: '{self.var}'"
        return msg

class TargetNotFoundError(CompilationError):

    def __init__(self, node, target_name, target_kind, target_package=None, target_version=None, disabled=None):
        self.node = node
        self.target_name = target_name
        self.target_kind = target_kind
        self.target_package = target_package
        self.target_version = target_version
        self.disabled = disabled
        super().__init__(msg=self.get_message())

    def get_message(self):
        original_file_path = self.node.original_file_path
        unique_id = self.node.unique_id
        resource_type_title = self.node.resource_type.title()
        if self.disabled is None:
            reason = 'was not found or is disabled'
        elif self.disabled is True:
            reason = 'is disabled'
        else:
            reason = 'was not found'
        target_version_string = ''
        if self.target_version is not None:
            target_version_string = f"with version '{self.target_version}' "
        target_package_string = ''
        if self.target_package is not None:
            target_package_string = f"in package or project '{self.target_package}' "
        msg = f"{resource_type_title} '{unique_id}' ({original_file_path}) depends on a {self.target_kind} named '{self.target_name}' {target_version_string}{target_package_string}which {reason}"
        return msg

class DuplicateSourcePatchNameError(CompilationError):

    def __init__(self, patch_1, patch_2):
        self.patch_1 = patch_1
        self.patch_2 = patch_2
        super().__init__(msg=self.get_message())

    def get_message(self):
        name = f'{self.patch_1.overrides}.{self.patch_1.name}'
        fix = self._fix_dupe_msg(self.patch_1.path, self.patch_2.path, name, 'sources')
        msg = f'dbt found two schema.yml entries for the same source named {self.patch_1.name} in package {self.patch_1.overrides}. Sources may only be overridden a single time. To fix this, {fix}'
        return msg

class DuplicateMacroPatchNameError(CompilationError):

    def __init__(self, patch_1, existing_patch_path):
        self.patch_1 = patch_1
        self.existing_patch_path = existing_patch_path
        super().__init__(msg=self.get_message())

    def get_message(self):
        package_name = self.patch_1.package_name
        name = self.patch_1.name
        fix = self._fix_dupe_msg(self.patch_1.original_file_path, self.existing_patch_path, name, 'macros')
        msg = f'dbt found two schema.yml entries for the same macro in package {package_name} named {name}. Macros may only be described a single time. To fix this, {fix}'
        return msg

class DuplicateAliasError(AliasError):

    def __init__(self, kwargs, aliases, canonical_key):
        self.kwargs = kwargs
        self.aliases = aliases
        self.canonical_key = canonical_key
        super().__init__(msg=self.get_message())

    def get_message(self):
        key_names = ', '.join(('{}'.format(k) for k in self.kwargs if self.aliases.get(k) == self.canonical_key))
        msg = f'Got duplicate keys: ({key_names}) all map to "{self.canonical_key}"'
        return msg

class MultipleVersionGitDepsError(DependencyError):

    def __init__(self, git, requested):
        self.git = git
        self.requested = requested
        msg = f'git dependencies should contain exactly one version. {self.git} contains: {self.requested}'
        super().__init__(msg)

class DuplicateProjectDependencyError(DependencyError):

    def __init__(self, project_name):
        self.project_name = project_name
        msg = f'Found duplicate project "{self.project_name}". This occurs when a dependency has the same project name as some other dependency.'
        super().__init__(msg)

class DuplicateDependencyToRootError(DependencyError):

    def __init__(self, project_name):
        self.project_name = project_name
        msg = f'Found a dependency with the same name as the root project "{self.project_name}". Package names must be unique in a project. Please rename one of these packages.'
        super().__init__(msg)

class MismatchedDependencyTypeError(DependencyError):

    def __init__(self, new, old):
        self.new = new
        self.old = old
        msg = f'Cannot incorporate {self.new} ({self.new.__class__.__name__}) in {self.old} ({self.old.__class__.__name__}): mismatched types'
        super().__init__(msg)

class PackageVersionNotFoundError(DependencyError):

    def __init__(self, package_name, version_range, available_versions, should_version_check):
        self.package_name = package_name
        self.version_range = version_range
        self.available_versions = available_versions
        self.should_version_check = should_version_check
        super().__init__(self.get_message())

    def get_message(self):
        base_msg = 'Could not find a matching compatible version for package {}\n  Requested range: {}\n  Compatible versions: {}\n'
        addendum = "\n  Not shown: package versions incompatible with installed version of dbt-core\n  To include them, run 'dbt --no-version-check deps'" if self.should_version_check else ''
        msg = base_msg.format(self.package_name, self.version_range, self.available_versions) + addendum
        return msg

class PackageNotFoundError(DependencyError):

    def __init__(self, package_name):
        self.package_name = package_name
        msg = f'Package {self.package_name} was not found in the package index'
        super().__init__(msg)

class ProfileConfigError(DbtProfileError):

    def __init__(self, exc):
        self.exc = exc
        msg = self.validator_error_message(self.exc)
        super().__init__(msg=msg)

class ProjectContractError(DbtProjectError):

    def __init__(self, exc):
        self.exc = exc
        msg = self.validator_error_message(self.exc)
        super().__init__(msg=msg)

class ProjectContractBrokenError(DbtProjectError):

    def __init__(self, exc):
        self.exc = exc
        msg = self.validator_error_message(self.exc)
        super().__init__(msg=msg)

class ConfigContractBrokenError(DbtProjectError):

    def __init__(self, exc):
        self.exc = exc
        msg = self.validator_error_message(self.exc)
        super().__init__(msg=msg)

class NonUniquePackageNameError(CompilationError):

    def __init__(self, project_name):
        self.project_name = project_name
        super().__init__(msg=self.get_message())

    def get_message(self):
        msg = f'dbt found more than one package with the name "{self.project_name}" included in this project. Package names must be unique in a project. Please rename one of these packages.'
        return msg

class UninstalledPackagesFoundError(CompilationError):

    def __init__(self, count_packages_specified, count_packages_installed, packages_specified_path, packages_install_path):
        self.count_packages_specified = count_packages_specified
        self.count_packages_installed = count_packages_installed
        self.packages_specified_path = packages_specified_path
        self.packages_install_path = packages_install_path
        super().__init__(msg=self.get_message())

    def get_message(self):
        msg = f'dbt found {self.count_packages_specified} package(s) specified in {self.packages_specified_path}, but only {self.count_packages_installed} package(s) installed in {self.packages_install_path}. Run "dbt deps" to install package dependencies.'
        return msg

class OptionNotYamlDictError(CompilationError):

    def __init__(self, var_type, option_name):
        self.var_type = var_type
        self.option_name = option_name
        super().__init__(msg=self.get_message())

    def get_message(self):
        type_name = self.var_type.__name__
        msg = f"The --{self.option_name} argument must be a YAML dictionary, but was of type '{type_name}'"
        return msg

class UnrecognizedCredentialTypeError(CompilationError):

    def __init__(self, typename, supported_types):
        self.typename = typename
        self.supported_types = supported_types
        super().__init__(msg=self.get_message())

    def get_message(self):
        msg = 'Unrecognized credentials type "{}" - supported types are ({})'.format(self.typename, ', '.join(('"{}"'.format(t) for t in self.supported_types)))
        return msg

class PatchTargetNotFoundError(CompilationError):

    def __init__(self, patches):
        self.patches = patches
        super().__init__(msg=self.get_message())

    def get_message(self):
        patch_list = '\n\t'.join((f'model {p.name} (referenced in path {p.original_file_path})' for p in self.patches.values()))
        msg = f'dbt could not find models for the following patches:\n\t{patch_list}'
        return msg

class MissingRelationError(CompilationError):

    def __init__(self, relation, model=None):
        self.relation = relation
        self.model = model
        msg = f'Relation {self.relation} not found!'
        super().__init__(msg=msg)

class AmbiguousAliasError(CompilationError):

    def __init__(self, node_1, node_2, duped_name=None):
        self.node_1 = node_1
        self.node_2 = node_2
        if duped_name is None:
            self.duped_name = f'{self.node_1.database}.{self.node_1.schema}.{self.node_1.alias}'
        else:
            self.duped_name = duped_name
        super().__init__(msg=self.get_message())

    def get_message(self):
        msg = f'dbt found two resources with the database representation "{self.duped_name}".\ndbt cannot create two resources with identical database representations. To fix this,\nchange the configuration of one of these resources:\n- {self.node_1.unique_id} ({self.node_1.original_file_path})\n- {self.node_2.unique_id} ({self.node_2.original_file_path})'
        return msg

class AmbiguousResourceNameRefError(CompilationError):

    def __init__(self, duped_name, unique_ids, node=None):
        self.duped_name = duped_name
        self.unique_ids = unique_ids
        self.packages = [unique_id.split('.')[1] for unique_id in unique_ids]
        super().__init__(msg=self.get_message(), node=node)

    def get_message(self):
        formatted_unique_ids = "'{0}'".format("', '".join(self.unique_ids))
        formatted_packages = "'{0}'".format("' or '".join(self.packages))
        msg = f"When referencing '{self.duped_name}', dbt found nodes in multiple packages: {formatted_unique_ids}\nTo fix this, use two-argument 'ref', with the package name first: {formatted_packages}"
        return msg

class AmbiguousCatalogMatchError(CompilationError):

    def __init__(self, unique_id, match_1, match_2):
        self.unique_id = unique_id
        self.match_1 = match_1
        self.match_2 = match_2
        super().__init__(msg=self.get_message())

    def get_match_string(self, match):
        match_schema = match.get('metadata', {}).get('schema')
        match_name = match.get('metadata', {}).get('name')
        return f'{match_schema}.{match_name}'

    def get_message(self):
        msg = f'dbt found two relations in your warehouse with similar database identifiers. dbt\nis unable to determine which of these relations was created by the model "{self.unique_id}".\nIn order for dbt to correctly generate the catalog, one of the following relations must be deleted or renamed:\n\n - {self.get_match_string(self.match_1)}\n - {self.get_match_string(self.match_2)}'
        return msg

class DependencyNotFoundError(CompilationError):

    def __init__(self, node, node_description, required_pkg):
        self.node = node
        self.node_description = node_description
        self.required_pkg = required_pkg
        super().__init__(msg=self.get_message())

    def get_message(self):
        msg = f'Error while parsing {self.node_description}.\nThe required package "{self.required_pkg}" was not found. Is the package installed?\nHint: You may need to run `dbt deps`.'
        return msg

class DuplicatePatchPathError(CompilationError):

    def __init__(self, patch_1, existing_patch_path):
        self.patch_1 = patch_1
        self.existing_patch_path = existing_patch_path
        super().__init__(msg=self.get_message())

    def get_message(self):
        name = self.patch_1.name
        fix = self._fix_dupe_msg(self.patch_1.original_file_path, self.existing_patch_path, name, 'resource')
        msg = f'dbt found two schema.yml entries for the same resource named {name}. Resources and their associated columns may only be described a single time. To fix this, {fix}'
        return msg

class DuplicateResourceNameError(CompilationError):

    def __init__(self, node_1, node_2):
        self.node_1 = node_1
        self.node_2 = node_2
        super().__init__(msg=self.get_message())

    def get_message(self):
        duped_name = self.node_1.name
        node_type = NodeType(self.node_1.resource_type)
        pluralized = node_type.pluralize() if self.node_1.resource_type == self.node_2.resource_type else 'resources'
        action = 'looking for'
        if node_type in REFABLE_NODE_TYPES:
            formatted_name = f'ref("{duped_name}")'
        elif node_type == NodeType.Source:
            duped_name = self.node_1.get_full_source_name()
            formatted_name = self.node_1.get_source_representation()
        elif node_type == NodeType.Documentation:
            formatted_name = f'doc("{duped_name}")'
        elif node_type == NodeType.Test and hasattr(self.node_1, 'test_metadata'):
            column_name = f'column "{self.node_1.column_name}" in ' if self.node_1.column_name else ''
            model_name = self.node_1.file_key_name
            duped_name = f'{self.node_1.name}" defined on {column_name}"{model_name}'
            action = 'running'
            formatted_name = 'tests'
        else:
            formatted_name = duped_name
        msg = f'\ndbt found two {pluralized} with the name "{duped_name}".\n\nSince these resources have the same name, dbt will be unable to find the correct resource\nwhen {action} {formatted_name}.\n\nTo fix this, change the name of one of these resources:\n- {self.node_1.unique_id} ({self.node_1.original_file_path})\n- {self.node_2.unique_id} ({self.node_2.original_file_path})\n    '.strip()
        return msg

class DuplicateVersionedUnversionedError(ParsingError):

    def __init__(self, versioned_node, unversioned_node):
        self.versioned_node = versioned_node
        self.unversioned_node = unversioned_node
        super().__init__(msg=self.get_message())

    def get_message(self):
        msg = f"""\ndbt found versioned and unversioned models with the name "{self.versioned_node.name}".\n\nSince these resources have the same name, dbt will be unable to find the correct resource\nwhen looking for ref('{self.versioned_node.name}').\n\nTo fix this, change the name of the unversioned resource\n{self.unversioned_node.unique_id} ({self.unversioned_node.original_file_path})\nor add the unversioned model to the versions in {self.versioned_node.patch_path}\n    """.strip()
        return msg

class PropertyYMLError(CompilationError):

    def __init__(self, path, issue):
        self.path = path
        self.issue = issue
        super().__init__(msg=self.get_message())

    def get_message(self):
        msg = f'The yml property file at {self.path} is invalid because {self.issue}. Please consult the documentation for more information on yml property file syntax:\n\nhttps://docs.getdbt.com/reference/configs-and-properties'
        return msg

class ContractError(CompilationError):

    def __init__(self, yaml_columns, sql_columns):
        self.yaml_columns = yaml_columns
        self.sql_columns = sql_columns
        super().__init__(msg=self.get_message())

    def get_mismatches(self):
        from dbt_common.clients.agate_helper import table_from_data_flat
        column_names = ['column_name', 'definition_type', 'contract_type', 'mismatch_reason']
        mismatches = []
        sql_col_set = set()
        for sql_col in self.sql_columns:
            sql_col_set.add(sql_col['name'])
            for i, yaml_col in enumerate(self.yaml_columns):
                if sql_col['name'] == yaml_col['name']:
                    if sql_col['data_type'] == yaml_col['data_type']:
                        break
                    else:
                        row = [sql_col['name'], sql_col['data_type'], yaml_col['data_type'], 'data type mismatch']
                        mismatches += [dict(zip(column_names, row))]
                        break
                if i == len(self.yaml_columns) - 1:
                    row = [sql_col['name'], sql_col['data_type'], '', 'missing in contract']
                    mismatches += [dict(zip(column_names, row))]
        for yaml_col in self.yaml_columns:
            if yaml_col['name'] not in sql_col_set:
                row = [yaml_col['name'], '', yaml_col['data_type'], 'missing in definition']
                mismatches += [dict(zip(column_names, row))]
        mismatches_sorted = sorted(mismatches, key=lambda d: d['column_name'])
        return table_from_data_flat(mismatches_sorted, column_names)

    def get_message(self):
        if not self.yaml_columns:
            return "This model has an enforced contract, and its 'columns' specification is missing"
        table = self.get_mismatches()
        output = io.StringIO()
        table.print_table(output=output, max_rows=None, max_column_width=50)
        mismatches = output.getvalue()
        msg = f"This model has an enforced contract that failed.\nPlease ensure the name, data_type, and number of columns in your contract match the columns in your model's definition.\n\n{mismatches}"
        return msg

class UnknownAsyncIDException(Exception):
    CODE = 10012
    MESSAGE = 'RPC server got an unknown async ID'

    def __init__(self, task_id):
        self.task_id = task_id

    def __str__(self):
        return f'{self.MESSAGE}: {self.task_id}'

class RPCFailureResult(DbtRuntimeError):
    CODE = 10002
    MESSAGE = 'RPC execution error'

class RPCTimeoutException(DbtRuntimeError):
    CODE = 10008
    MESSAGE = 'RPC timeout error'

    def __init__(self, timeout=None):
        super().__init__(self.MESSAGE)
        self.timeout = timeout

    def data(self):
        result = super().data()
        result.update({'timeout': self.timeout, 'message': f'RPC timed out after {self.timeout}s'})
        return result

class RPCKilledException(DbtRuntimeError):
    CODE = 10009
    MESSAGE = 'RPC process killed'

    def __init__(self, signum):
        self.signum = signum
        self.msg = f'RPC process killed by signal {self.signum}'
        super().__init__(self.msg)

    def data(self):
        return {'signum': self.signum, 'message': self.msg}

class RPCCompiling(DbtRuntimeError):
    CODE = 10010
    MESSAGE = 'RPC server is compiling the project, call the "status" method for compile status'

    def __init__(self, msg=None, node=None):
        if msg is None:
            msg = 'compile in progress'
        super().__init__(msg, node)

class RPCLoadException(DbtRuntimeError):
    CODE = 10011
    MESSAGE = 'RPC server failed to compile project, call the "status" method for compile status'

    def __init__(self, cause):
        self.cause = cause
        self.msg = f'{self.MESSAGE}: {self.cause['message']}'
        super().__init__(self.msg)

    def data(self):
        return {'cause': self.cause, 'message': self.msg}