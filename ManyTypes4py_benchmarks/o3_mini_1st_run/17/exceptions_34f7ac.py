import io
import json
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from dbt.node_types import REFABLE_NODE_TYPES, AccessType, NodeType
from dbt_common.constants import SECRET_ENV_PREFIX
from dbt_common.dataclass_schema import ValidationError
from dbt_common.exceptions import (
    CommandResultError,
    CompilationError,
    DbtConfigError,
    DbtInternalError,
    DbtRuntimeError,
    DbtValidationError,
    env_secrets,
    scrub_secrets,
)
if TYPE_CHECKING:
    import agate


class ContractBreakingChangeError(DbtRuntimeError):
    CODE: int = 10016
    MESSAGE: str = 'Breaking Change to Contract'

    def __init__(self, breaking_changes: List[str], node: Optional[Any] = None) -> None:
        self.breaking_changes: List[str] = breaking_changes
        super().__init__(self.message(), node)

    @property
    def type(self) -> str:
        return 'Breaking change to contract'

    def message(self) -> str:
        reasons: str = '\n  - '.join(self.breaking_changes)
        return (
            f'While comparing to previous project state, dbt detected a breaking change to an enforced contract.\n'
            f'  - {reasons}\n'
            'Consider making an additive (non-breaking) change instead, if possible.\n'
            'Otherwise, create a new model version: https://docs.getdbt.com/docs/collaborate/govern/model-versions'
        )


class ParsingError(DbtRuntimeError):
    CODE: int = 10015
    MESSAGE: str = 'Parsing Error'

    @property
    def type(self) -> str:
        return 'Parsing'


class dbtPluginError(DbtRuntimeError):
    CODE: int = 10020
    MESSAGE: str = 'Plugin Error'


class JSONValidationError(DbtValidationError):
    def __init__(self, typename: str, errors: List[str]) -> None:
        self.typename: str = typename
        self.errors: List[str] = errors
        self.errors_message: str = ', '.join(errors)
        msg: str = f'Invalid arguments passed to "{self.typename}" instance: {self.errors_message}'
        super().__init__(msg)

    def __reduce__(self) -> Any:
        return (JSONValidationError, (self.typename, self.errors))


class AliasError(DbtValidationError):
    pass


class DependencyError(Exception):
    CODE: int = 10006
    MESSAGE: str = 'Dependency Error'


class FailFastError(DbtRuntimeError):
    CODE: int = 10013
    MESSAGE: str = 'FailFast Error'

    def __init__(self, msg: str, result: Optional[Any] = None, node: Optional[Any] = None) -> None:
        super().__init__(msg=msg, node=node)
        self.result: Optional[Any] = result

    @property
    def type(self) -> str:
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
    def __init__(self, name: str) -> None:
        self.name: str = name
        super().__init__(name)


class DuplicateYamlKeyError(CompilationError):
    pass


class GraphDependencyNotFoundError(CompilationError):
    def __init__(self, node: Any, dependency: str) -> None:
        self.node: Any = node
        self.dependency: str = dependency
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg: str = f"'{self.node.unique_id}' depends on '{self.dependency}' which is not in the graph!"
        return msg


class ForeignKeyConstraintToSyntaxError(CompilationError):
    def __init__(self, node: Any, expression: str) -> None:
        self.expression: str = expression
        self.node: Any = node
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg: str = (
            f"'{self.node.unique_id}' defines a foreign key constraint 'to' expression which is not valid "
            f"'ref' or 'source' syntax: {self.expression}."
        )
        return msg


class NoSupportedLanguagesFoundError(CompilationError):
    def __init__(self, node: Any) -> None:
        self.node: Any = node
        self.msg: str = f'No supported_languages found in materialization macro {self.node.name}'
        super().__init__(msg=self.msg)


class MaterializtionMacroNotUsedError(CompilationError):
    def __init__(self, node: Any) -> None:
        self.node: Any = node
        self.msg: str = 'Only materialization macros can be used with this function'
        super().__init__(msg=self.msg)


class MacroNamespaceNotStringError(CompilationError):
    def __init__(self, kwarg_type: Any) -> None:
        self.kwarg_type: Any = kwarg_type
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg: str = f'The macro_namespace parameter to adapter.dispatch is a {self.kwarg_type}, not a string'
        return msg


class UnknownGitCloningProblemError(DbtRuntimeError):
    def __init__(self, repo: str) -> None:
        self.repo: str = scrub_secrets(repo, env_secrets())
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg: str = (
            f'        Something went wrong while cloning {self.repo}\n'
            f'        Check the debug logs for more information\n        '
        )
        return msg


class NoAdaptersAvailableError(DbtRuntimeError):
    def __init__(self) -> None:
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg: str = (
            'No adapters available. Learn how to install an adapter by going to '
            'https://docs.getdbt.com/docs/connect-adapters#install-using-the-cli'
        )
        return msg


class BadSpecError(DbtInternalError):
    def __init__(self, repo: str, revision: str, error: Any) -> None:
        self.repo: str = repo
        self.revision: str = revision
        self.stderr: str = scrub_secrets(error.stderr.strip(), env_secrets())
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg: str = f"Error checking out spec='{self.revision}' for repo {self.repo}\n{self.stderr}"
        return msg


class GitCloningError(DbtInternalError):
    def __init__(self, repo: str, revision: str, error: Any) -> None:
        self.repo: str = repo
        self.revision: str = revision
        self.error: Any = error
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        stderr: str = self.error.stderr.strip()
        if 'usage: git' in stderr:
            stderr = stderr.split('\nusage: git')[0]
        if re.match("fatal: destination path '(.+)' already exists", stderr):
            self.error.cmd = list(scrub_secrets(str(self.error.cmd), env_secrets()))
            raise self.error
        msg: str = f"Error checking out spec='{self.revision}' for repo {self.repo}\n{stderr}"
        return scrub_secrets(msg, env_secrets())


class GitCheckoutError(BadSpecError):
    pass


class OperationError(CompilationError):
    def __init__(self, operation_name: str) -> None:
        self.operation_name: str = operation_name
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg: str = (
            f'dbt encountered an error when attempting to create a {self.operation_name}. '
            'If this error persists, please create an issue at: \n\nhttps://github.com/dbt-labs/dbt-core'
        )
        return msg


class ZipStrictWrongTypeError(CompilationError):
    def __init__(self, exc: Exception) -> None:
        self.exc: Exception = exc
        msg: str = str(self.exc)
        super().__init__(msg=msg)


class SetStrictWrongTypeError(CompilationError):
    def __init__(self, exc: Exception) -> None:
        self.exc: Exception = exc
        msg: str = str(self.exc)
        super().__init__(msg=msg)


class LoadAgateTableValueError(CompilationError):
    def __init__(self, exc: Exception, node: Any) -> None:
        self.exc: Exception = exc
        self.node: Any = node
        msg: str = str(self.exc)
        super().__init__(msg=msg)


class LoadAgateTableNotSeedError(CompilationError):
    def __init__(self, resource_type: str, node: Any) -> None:
        self.resource_type: str = resource_type
        self.node: Any = node
        msg: str = f'can only load_agate_table for seeds (got a {self.resource_type})'
        super().__init__(msg=msg)


class PackageNotInDepsError(CompilationError):
    def __init__(self, package_name: str, node: Any) -> None:
        self.package_name: str = package_name
        self.node: Any = node
        msg: str = f'Node package named {self.package_name} not found!'
        super().__init__(msg=msg)


class OperationsCannotRefEphemeralNodesError(CompilationError):
    def __init__(self, target_name: str, node: Any) -> None:
        self.target_name: str = target_name
        self.node: Any = node
        msg: str = f'Operations can not ref() ephemeral nodes, but {target_name} is ephemeral'
        super().__init__(msg=msg)


class PersistDocsValueTypeError(CompilationError):
    def __init__(self, persist_docs: Any) -> None:
        self.persist_docs: Any = persist_docs
        msg: str = f"Invalid value provided for 'persist_docs'. Expected dict but received {type(self.persist_docs)}"
        super().__init__(msg=msg)


class InlineModelConfigError(CompilationError):
    def __init__(self, node: Any) -> None:
        self.node: Any = node
        msg: str = 'Invalid inline model config'
        super().__init__(msg=msg)


class ConflictingConfigKeysError(CompilationError):
    def __init__(self, oldkey: str, newkey: str, node: Any) -> None:
        self.oldkey: str = oldkey
        self.newkey: str = newkey
        self.node: Any = node
        msg: str = f'Invalid config, has conflicting keys "{self.oldkey}" and "{self.newkey}"'
        super().__init__(msg=msg)


class NumberSourceArgsError(CompilationError):
    def __init__(self, args: List[Any], node: Any) -> None:
        self.args: List[Any] = args
        self.node: Any = node
        msg: str = f'source() takes exactly two arguments ({len(self.args)} given)'
        super().__init__(msg=msg)


class RequiredVarNotFoundError(CompilationError):
    def __init__(self, var_name: str, merged: Dict[str, str], node: Optional[Any]) -> None:
        self.var_name: str = var_name
        self.merged: Dict[str, str] = merged
        self.node: Optional[Any] = node
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        node_name: str = self.node.name if self.node is not None else '<Configuration>'
        dct: Dict[str, str] = {k: self.merged[k] for k in self.merged}
        pretty_vars: str = json.dumps(dct, sort_keys=True, indent=4)
        msg: str = f"Required var '{self.var_name}' not found in config:\nVars supplied to {node_name} = {pretty_vars}"
        return scrub_secrets(msg, self.var_secrets())

    def var_secrets(self) -> List[str]:
        return [v for k, v in self.merged.items() if k.startswith(SECRET_ENV_PREFIX) and v.strip()]


class PackageNotFoundForMacroError(CompilationError):
    def __init__(self, package_name: str) -> None:
        self.package_name: str = package_name
        msg: str = f"Could not find package '{self.package_name}'"
        super().__init__(msg=msg)


class SecretEnvVarLocationError(ParsingError):
    def __init__(self, env_var_name: str) -> None:
        self.env_var_name: str = env_var_name
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg: str = f"Secret env vars are allowed only in profiles.yml or packages.yml. Found '{self.env_var_name}' referenced elsewhere."
        return msg


class BooleanError(CompilationError):
    def __init__(self, return_value: Any, macro_name: str) -> None:
        self.return_value: Any = return_value
        self.macro_name: str = macro_name
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg: str = (
            f"Macro '{self.macro_name}' returns '{self.return_value}'.  It is not type 'bool' and cannot not be converted reliably to a bool."
        )
        return msg


class RefArgsError(CompilationError):
    def __init__(self, node: Any, args: List[Any]) -> None:
        self.node: Any = node
        self.args: List[Any] = args
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg: str = f'ref() takes at most two arguments ({len(self.args)} given)'
        return msg


class MetricArgsError(CompilationError):
    def __init__(self, node: Any, args: List[Any]) -> None:
        self.node: Any = node
        self.args: List[Any] = args
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg: str = f'metric() takes at most two arguments ({len(self.args)} given)'
        return msg


class RefBadContextError(CompilationError):
    def __init__(self, node: Any, args: Any) -> None:
        self.node: Any = node
        self.args = args.positional_args
        self.kwargs: Dict[str, Any] = args.keyword_args
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        if isinstance(self.node, dict):
            model_name: str = self.node['name']
        else:
            model_name: str = self.node.name
        ref_args: str = ', '.join(("'{}'".format(a) for a in self.args))
        keyword_args: str = ''
        if self.kwargs:
            keyword_args = ', '.join(("{}='{}'".format(k, v) for k, v in self.kwargs.items()))
            keyword_args = ',' + keyword_args
        ref_string: str = f'{{{{ ref({ref_args}{keyword_args}) }}}}'
        msg: str = (
            f'dbt was unable to infer all dependencies for the model "{model_name}".\n'
            f'This typically happens when ref() is placed within a conditional block.\n\n'
            f'To fix this, add the following hint to the top of the model "{model_name}":\n\n'
            f'-- depends_on: {ref_string}'
        )
        return msg


class DocArgsError(CompilationError):
    def __init__(self, node: Any, args: List[Any]) -> None:
        self.node: Any = node
        self.args: List[Any] = args
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg: str = f'doc() takes at most two arguments ({len(self.args)} given)'
        return msg


class DocTargetNotFoundError(CompilationError):
    def __init__(self, node: Any, target_doc_name: str, target_doc_package: Optional[str] = None) -> None:
        self.node: Any = node
        self.target_doc_name: str = target_doc_name
        self.target_doc_package: Optional[str] = target_doc_package
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        target_package_string: str = ''
        if self.target_doc_package is not None:
            target_package_string = f"in package '{self.target_doc_package}' "
        msg: str = (
            f"Documentation for '{self.node.unique_id}' depends on doc '{self.target_doc_name}' "
            f"{target_package_string}which was not found"
        )
        return msg


class MacroDispatchArgError(CompilationError):
    def __init__(self, macro_name: str) -> None:
        self.macro_name: str = macro_name
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg: str = (
            '        The "packages" argument of adapter.dispatch() has been deprecated.\n'
            '        Use the "macro_namespace" argument instead.\n\n'
            f'        Raised during dispatch for: {self.macro_name}\n\n'
            '        For more information, see:\n\n'
            '        https://docs.getdbt.com/reference/dbt-jinja-functions/dispatch\n        '
        )
        return msg


class DuplicateMacroNameError(CompilationError):
    def __init__(self, node_1: Any, node_2: Any, namespace: str) -> None:
        self.node_1: Any = node_1
        self.node_2: Any = node_2
        self.namespace: str = namespace
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        duped_name: str = self.node_1.name
        if self.node_1.package_name != self.node_2.package_name:
            extra: str = f' ("{self.node_1.package_name}" and "{self.node_2.package_name}" are both in the "{self.namespace}" namespace)'
        else:
            extra = ''
        msg: str = (
            f'dbt found two macros with the name "{duped_name}" in the namespace "{self.namespace}"{extra}. '
            f'Since these macros have the same name and exist in the same namespace, dbt will be unable to decide which to call. '
            f'To fix this, change the name of one of these macros:\n'
            f'- {self.node_1.unique_id} ({self.node_1.original_file_path})\n'
            f'- {self.node_2.unique_id} ({self.node_2.original_file_path})'
        )
        return msg


class MacroResultAlreadyLoadedError(CompilationError):
    def __init__(self, result_name: str) -> None:
        self.result_name: str = result_name
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg: str = f"The 'statement' result named '{self.result_name}' has already been loaded into a variable"
        return msg


class DictParseError(ParsingError):
    def __init__(self, exc: Exception, node: Any) -> None:
        self.exc: Exception = exc
        self.node: Any = node
        msg: str = self.validator_error_message(exc)
        super().__init__(msg=msg)


class ConfigUpdateError(ParsingError):
    def __init__(self, exc: Exception, node: Any) -> None:
        self.exc: Exception = exc
        self.node: Any = node
        msg: str = self.validator_error_message(exc)
        super().__init__(msg=msg)


class PythonParsingError(ParsingError):
    def __init__(self, exc: Exception, node: Any) -> None:
        self.exc: Exception = exc
        self.node: Any = node
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        validated_exc: str = self.validator_error_message(self.exc)
        msg: str = f'{validated_exc}\n{self.exc.text}'
        return msg


class PythonLiteralEvalError(ParsingError):
    def __init__(self, exc: Exception, node: Any) -> None:
        self.exc: Exception = exc
        self.node: Any = node
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg: str = (
            f'Error when trying to literal_eval an arg to dbt.ref(), dbt.source(), dbt.config() or dbt.config.get() \n'
            f'{self.exc}\n'
            'https://docs.python.org/3/library/ast.html#ast.literal_eval\n'
            'In dbt python model, `dbt.ref`, `dbt.source`, `dbt.config`, `dbt.config.get` function args only support Python literal structures'
        )
        return msg


class ModelConfigError(ParsingError):
    def __init__(self, exc: Exception, node: Any) -> None:
        self.msg: str = self.validator_error_message(exc)
        self.node: Any = node
        super().__init__(msg=self.msg)


class YamlParseListError(ParsingError):
    def __init__(self, path: str, key: str, yaml_data: Any, cause: Any) -> None:
        self.path: str = path
        self.key: str = key
        self.yaml_data: Any = yaml_data
        self.cause: Any = cause
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        if isinstance(self.cause, str):
            reason: str = self.cause
        elif isinstance(self.cause, ValidationError):
            reason = self.validator_error_message(self.cause)
        else:
            reason = self.cause.msg
        msg: str = f'Invalid {self.key} config given in {self.path} @ {self.key}: {self.yaml_data} - {reason}'
        return msg


class YamlParseDictError(ParsingError):
    def __init__(self, path: str, key: str, yaml_data: Any, cause: Any) -> None:
        self.path: str = path
        self.key: str = key
        self.yaml_data: Any = yaml_data
        self.cause: Any = cause
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        if isinstance(self.cause, str):
            reason: str = self.cause
        elif isinstance(self.cause, ValidationError):
            reason = self.validator_error_message(self.cause)
        else:
            reason = self.cause.msg
        msg: str = f'Invalid {self.key} config given in {self.path} @ {self.key}: {self.yaml_data} - {reason}'
        return msg


class YamlLoadError(ParsingError):
    def __init__(self, path: str, exc: Exception, project_name: Optional[str] = None) -> None:
        self.project_name: Optional[str] = project_name
        self.path: str = path
        self.exc: Exception = exc
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        reason: str = self.validator_error_message(self.exc)
        msg: str = f'Error reading {self.project_name}: {self.path} - {reason}'
        return msg


class TestConfigError(ParsingError):
    def __init__(self, exc: Exception, node: Any) -> None:
        self.msg: str = self.validator_error_message(exc)
        self.node: Any = node
        super().__init__(msg=self.msg)


class SchemaConfigError(ParsingError):
    def __init__(self, exc: Exception, node: Any) -> None:
        self.msg: str = self.validator_error_message(exc)
        self.node: Any = node
        super().__init__(msg=self.msg)


class SnapshopConfigError(ParsingError):
    def __init__(self, exc: Exception, node: Any) -> None:
        self.msg: str = self.validator_error_message(exc)
        self.node: Any = node
        super().__init__(msg=self.msg)


class DbtReferenceError(ParsingError):
    def __init__(self, unique_id: str, ref_unique_id: str, access: AccessType, scope: str) -> None:
        self.unique_id: str = unique_id
        self.ref_unique_id: str = ref_unique_id
        self.access: AccessType = access
        self.scope: str = scope
        self.scope_type: str = 'group' if self.access == AccessType.Private else 'package'
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        return (
            f"Node {self.unique_id} attempted to reference node {self.ref_unique_id}, which is not allowed because "
            f"the referenced node is {self.access} to the '{self.scope}' {self.scope_type}."
        )


class InvalidAccessTypeError(ParsingError):
    def __init__(self, unique_id: str, field_value: Any, materialization: Optional[str] = None) -> None:
        self.unique_id: str = unique_id
        self.field_value: Any = field_value
        self.materialization: Optional[str] = materialization
        with_materialization: str = f"with '{self.materialization}' materialization " if self.materialization else ''
        msg: str = f'Node {self.unique_id} {with_materialization}has an invalid value ({self.field_value}) for the access field'
        super().__init__(msg=msg)


class InvalidUnitTestGivenInput(ParsingError):
    def __init__(self, input: Any) -> None:
        msg: str = f"Unit test given inputs must be either a 'ref', 'source' or 'this' call. Got: '{input}'."
        super().__init__(msg=msg)


class SameKeyNestedError(CompilationError):
    def __init__(self) -> None:
        msg: str = 'Test cannot have the same key at the top-level and in config'
        super().__init__(msg=msg)


class TestArgIncludesModelError(CompilationError):
    def __init__(self) -> None:
        msg: str = 'Test arguments include "model", which is a reserved argument'
        super().__init__(msg=msg)


class UnexpectedTestNamePatternError(CompilationError):
    def __init__(self, test_name: str) -> None:
        self.test_name: str = test_name
        msg: str = f'Test name string did not match expected pattern: {self.test_name}'
        super().__init__(msg=msg)


class CustomMacroPopulatingConfigValueError(CompilationError):
    def __init__(self, target_name: str, name: str, key: str, err_msg: str, column_name: Optional[str] = None) -> None:
        self.target_name: str = target_name
        self.column_name: Optional[str] = column_name
        self.name: str = name
        self.key: str = key
        self.err_msg: str = err_msg
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg: str = (
            f'''The {self.target_name}.{self.column_name} column's "{self.name}" test references an undefined macro in its {self.key} configuration argument. '''
            f'''The macro {self.err_msg}.\nPlease note that the generic test configuration parser currently does not support using custom macros to populate configuration values'''
        )
        return msg


class TagsNotListOfStringsError(CompilationError):
    def __init__(self, tags: Any) -> None:
        self.tags: Any = tags
        msg: str = f'got {self.tags} ({type(self.tags)}) for tags, expected a list of strings'
        super().__init__(msg=msg)


class TagNotStringError(CompilationError):
    def __init__(self, tag: Any) -> None:
        self.tag: Any = tag
        msg: str = f'got {self.tag} ({type(self.tag)}) for tag, expected a str'
        super().__init__(msg=msg)


class TestNameNotStringError(ParsingError):
    def __init__(self, test_name: Any) -> None:
        self.test_name: Any = test_name
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg: str = f'test name must be a str, got {type(self.test_name)} (value {self.test_name})'
        return msg


class TestArgsNotDictError(ParsingError):
    def __init__(self, test_args: Any) -> None:
        self.test_args: Any = test_args
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg: str = f'test arguments must be a dict, got {type(self.test_args)} (value {self.test_args})'
        return msg


class TestDefinitionDictLengthError(ParsingError):
    def __init__(self, test: Dict[Any, Any]) -> None:
        self.test: Dict[Any, Any] = test
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg: str = f'test definition dictionary must have exactly one key, got {self.test} instead ({len(self.test)} keys)'
        return msg


class TestTypeError(ParsingError):
    def __init__(self, test: Any) -> None:
        self.test: Any = test
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg: str = f'test must be dict or str, got {type(self.test)} (value {self.test})'
        return msg


class EnvVarMissingError(ParsingError):
    def __init__(self, var: str) -> None:
        self.var: str = var
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg: str = f"Env var required but not provided: '{self.var}'"
        return msg


class TargetNotFoundError(CompilationError):
    def __init__(
        self,
        node: Any,
        target_name: str,
        target_kind: str,
        target_package: Optional[str] = None,
        target_version: Optional[str] = None,
        disabled: Optional[bool] = None,
    ) -> None:
        self.node: Any = node
        self.target_name: str = target_name
        self.target_kind: str = target_kind
        self.target_package: Optional[str] = target_package
        self.target_version: Optional[str] = target_version
        self.disabled: Optional[bool] = disabled
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        original_file_path: str = self.node.original_file_path
        unique_id: str = self.node.unique_id
        resource_type_title: str = self.node.resource_type.title()
        if self.disabled is None:
            reason: str = 'was not found or is disabled'
        elif self.disabled is True:
            reason = 'is disabled'
        else:
            reason = 'was not found'
        target_version_string: str = ''
        if self.target_version is not None:
            target_version_string = f"with version '{self.target_version}' "
        target_package_string: str = ''
        if self.target_package is not None:
            target_package_string = f"in package or project '{self.target_package}' "
        msg: str = (
            f"{resource_type_title} '{unique_id}' ({original_file_path}) depends on a {self.target_kind} named "
            f"'{self.target_name}' {target_version_string}{target_package_string}which {reason}"
        )
        return msg


class DuplicateSourcePatchNameError(CompilationError):
    def __init__(self, patch_1: Any, patch_2: Any) -> None:
        self.patch_1: Any = patch_1
        self.patch_2: Any = patch_2
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        name: str = f'{self.patch_1.overrides}.{self.patch_1.name}'
        fix: str = self._fix_dupe_msg(self.patch_1.path, self.patch_2.path, name, 'sources')
        msg: str = (
            f'dbt found two schema.yml entries for the same source named {self.patch_1.name} in package {self.patch_1.overrides}. '
            f'Sources may only be overridden a single time. To fix this, {fix}'
        )
        return msg


class DuplicateMacroPatchNameError(CompilationError):
    def __init__(self, patch_1: Any, existing_patch_path: str) -> None:
        self.patch_1: Any = patch_1
        self.existing_patch_path: str = existing_patch_path
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        package_name: str = self.patch_1.package_name
        name: str = self.patch_1.name
        fix: str = self._fix_dupe_msg(self.patch_1.original_file_path, self.existing_patch_path, name, 'macros')
        msg: str = (
            f'dbt found two schema.yml entries for the same macro in package {package_name} named {name}. '
            f'Macros may only be described a single time. To fix this, {fix}'
        )
        return msg


class DuplicateAliasError(AliasError):
    def __init__(self, kwargs: Dict[str, Any], aliases: Dict[str, str], canonical_key: str) -> None:
        self.kwargs: Dict[str, Any] = kwargs
        self.aliases: Dict[str, str] = aliases
        self.canonical_key: str = canonical_key
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        key_names: str = ', '.join(str(k) for k in self.kwargs if self.aliases.get(k) == self.canonical_key)
        msg: str = f'Got duplicate keys: ({key_names}) all map to "{self.canonical_key}"'
        return msg


class MultipleVersionGitDepsError(DependencyError):
    def __init__(self, git: Any, requested: List[Any]) -> None:
        self.git: Any = git
        self.requested: List[Any] = requested
        msg: str = f'git dependencies should contain exactly one version. {self.git} contains: {self.requested}'
        super().__init__(msg)


class DuplicateProjectDependencyError(DependencyError):
    def __init__(self, project_name: str) -> None:
        self.project_name: str = project_name
        msg: str = f'Found duplicate project "{self.project_name}". This occurs when a dependency has the same project name as some other dependency.'
        super().__init__(msg)


class DuplicateDependencyToRootError(DependencyError):
    def __init__(self, project_name: str) -> None:
        self.project_name: str = project_name
        msg: str = (
            f'Found a dependency with the same name as the root project "{self.project_name}". '
            'Package names must be unique in a project. Please rename one of these packages.'
        )
        super().__init__(msg)


class MismatchedDependencyTypeError(DependencyError):
    def __init__(self, new: Any, old: Any) -> None:
        self.new: Any = new
        self.old: Any = old
        msg: str = (
            f'Cannot incorporate {self.new} ({self.new.__class__.__name__}) in {self.old} '
            f'({self.old.__class__.__name__}): mismatched types'
        )
        super().__init__(msg)


class PackageVersionNotFoundError(DependencyError):
    def __init__(self, package_name: str, version_range: str, available_versions: Any, should_version_check: bool) -> None:
        self.package_name: str = package_name
        self.version_range: str = version_range
        self.available_versions: Any = available_versions
        self.should_version_check: bool = should_version_check
        super().__init__(self.get_message())

    def get_message(self) -> str:
        base_msg: str = (
            f'Could not find a matching compatible version for package {self.package_name}\n'
            f'  Requested range: {self.version_range}\n'
            f'  Compatible versions: {self.available_versions}\n'
        )
        addendum: str = (
            "\n  Not shown: package versions incompatible with installed version of dbt-core\n"
            "  To include them, run 'dbt --no-version-check deps'"
            if self.should_version_check
            else ''
        )
        msg: str = base_msg + addendum
        return msg


class PackageNotFoundError(DependencyError):
    def __init__(self, package_name: str) -> None:
        self.package_name: str = package_name
        msg: str = f'Package {self.package_name} was not found in the package index'
        super().__init__(msg)


class ProfileConfigError(DbtProfileError):
    def __init__(self, exc: Exception) -> None:
        self.exc: Exception = exc
        msg: str = self.validator_error_message(self.exc)
        super().__init__(msg=msg)


class ProjectContractError(DbtProjectError):
    def __init__(self, exc: Exception) -> None:
        self.exc: Exception = exc
        msg: str = self.validator_error_message(self.exc)
        super().__init__(msg=msg)


class ProjectContractBrokenError(DbtProjectError):
    def __init__(self, exc: Exception) -> None:
        self.exc: Exception = exc
        msg: str = self.validator_error_message(self.exc)
        super().__init__(msg=msg)


class ConfigContractBrokenError(DbtProjectError):
    def __init__(self, exc: Exception) -> None:
        self.exc: Exception = exc
        msg: str = self.validator_error_message(self.exc)
        super().__init__(msg=msg)


class NonUniquePackageNameError(CompilationError):
    def __init__(self, project_name: str) -> None:
        self.project_name: str = project_name
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg: str = (
            f'dbt found more than one package with the name "{self.project_name}" included in this project. '
            'Package names must be unique in a project. Please rename one of these packages.'
        )
        return msg


class UninstalledPackagesFoundError(CompilationError):
    def __init__(
        self,
        count_packages_specified: int,
        count_packages_installed: int,
        packages_specified_path: str,
        packages_install_path: str,
    ) -> None:
        self.count_packages_specified: int = count_packages_specified
        self.count_packages_installed: int = count_packages_installed
        self.packages_specified_path: str = packages_specified_path
        self.packages_install_path: str = packages_install_path
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg: str = (
            f'dbt found {self.count_packages_specified} package(s) specified in {self.packages_specified_path}, '
            f'but only {self.count_packages_installed} package(s) installed in {self.packages_install_path}. '
            'Run "dbt deps" to install package dependencies.'
        )
        return msg


class OptionNotYamlDictError(CompilationError):
    def __init__(self, var_type: type, option_name: str) -> None:
        self.var_type: type = var_type
        self.option_name: str = option_name
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        type_name: str = self.var_type.__name__
        msg: str = f"The --{self.option_name} argument must be a YAML dictionary, but was of type '{type_name}'"
        return msg


class UnrecognizedCredentialTypeError(CompilationError):
    def __init__(self, typename: str, supported_types: List[str]) -> None:
        self.typename: str = typename
        self.supported_types: List[str] = supported_types
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg: str = 'Unrecognized credentials type "{}" - supported types are ({})'.format(
            self.typename, ', '.join(f'"{t}"' for t in self.supported_types)
        )
        return msg


class PatchTargetNotFoundError(CompilationError):
    def __init__(self, patches: Any) -> None:
        self.patches: Any = patches
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        patch_list: str = '\n\t'.join(f'model {p.name} (referenced in path {p.original_file_path})' for p in self.patches.values())
        msg: str = f'dbt could not find models for the following patches:\n\t{patch_list}'
        return msg


class MissingRelationError(CompilationError):
    def __init__(self, relation: Any, model: Optional[Any] = None) -> None:
        self.relation: Any = relation
        self.model: Optional[Any] = model
        msg: str = f'Relation {self.relation} not found!'
        super().__init__(msg=msg)


class AmbiguousAliasError(CompilationError):
    def __init__(self, node_1: Any, node_2: Any, duped_name: Optional[str] = None) -> None:
        self.node_1: Any = node_1
        self.node_2: Any = node_2
        if duped_name is None:
            self.duped_name: str = f'{self.node_1.database}.{self.node_1.schema}.{self.node_1.alias}'
        else:
            self.duped_name: str = duped_name
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg: str = (
            f'dbt found two resources with the database representation "{self.duped_name}".\n'
            f'dbt cannot create two resources with identical database representations. To fix this,\n'
            f'change the configuration of one of these resources:\n'
            f'- {self.node_1.unique_id} ({self.node_1.original_file_path})\n'
            f'- {self.node_2.unique_id} ({self.node_2.original_file_path})'
        )
        return msg


class AmbiguousResourceNameRefError(CompilationError):
    def __init__(self, duped_name: str, unique_ids: List[str], node: Optional[Any] = None) -> None:
        self.duped_name: str = duped_name
        self.unique_ids: List[str] = unique_ids
        self.packages: List[str] = [uid.split('.')[1] for uid in unique_ids]
        super().__init__(msg=self.get_message(), node=node)

    def get_message(self) -> str:
        formatted_unique_ids: str = "'{0}'".format("', '".join(self.unique_ids))
        formatted_packages: str = "'{0}'".format("' or '".join(self.packages))
        msg: str = (
            f"When referencing '{self.duped_name}', dbt found nodes in multiple packages: {formatted_unique_ids}\n"
            f"To fix this, use two-argument 'ref', with the package name first: {formatted_packages}"
        )
        return msg


class AmbiguousCatalogMatchError(CompilationError):
    def __init__(self, unique_id: str, match_1: Dict[str, Any], match_2: Dict[str, Any]) -> None:
        self.unique_id: str = unique_id
        self.match_1: Dict[str, Any] = match_1
        self.match_2: Dict[str, Any] = match_2
        super().__init__(msg=self.get_message())

    def get_match_string(self, match: Dict[str, Any]) -> str:
        match_schema: str = match.get('metadata', {}).get('schema')
        match_name: str = match.get('metadata', {}).get('name')
        return f'{match_schema}.{match_name}'

    def get_message(self) -> str:
        msg: str = (
            f'dbt found two relations in your warehouse with similar database identifiers. dbt\n'
            f'is unable to determine which of these relations was created by the model "{self.unique_id}".\n'
            f'In order for dbt to correctly generate the catalog, one of the following relations must be deleted or renamed:\n\n'
            f' - {self.get_match_string(self.match_1)}\n'
            f' - {self.get_match_string(self.match_2)}'
        )
        return msg


class DependencyNotFoundError(CompilationError):
    def __init__(self, node: Any, node_description: str, required_pkg: str) -> None:
        self.node: Any = node
        self.node_description: str = node_description
        self.required_pkg: str = required_pkg
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg: str = (
            f'Error while parsing {self.node_description}.\n'
            f'The required package "{self.required_pkg}" was not found. Is the package installed?\n'
            f'Hint: You may need to run `dbt deps`.'
        )
        return msg


class DuplicatePatchPathError(CompilationError):
    def __init__(self, patch_1: Any, existing_patch_path: str) -> None:
        self.patch_1: Any = patch_1
        self.existing_patch_path: str = existing_patch_path
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        name: str = self.patch_1.name
        fix: str = self._fix_dupe_msg(self.patch_1.original_file_path, self.existing_patch_path, name, 'resource')
        msg: str = (
            f'dbt found two schema.yml entries for the same resource named {name}. '
            f'Resources and their associated columns may only be described a single time. To fix this, {fix}'
        )
        return msg


class DuplicateResourceNameError(CompilationError):
    def __init__(self, node_1: Any, node_2: Any) -> None:
        self.node_1: Any = node_1
        self.node_2: Any = node_2
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        duped_name: str = self.node_1.name
        node_type: NodeType = NodeType(self.node_1.resource_type)
        pluralized: str = node_type.pluralize() if self.node_1.resource_type == self.node_2.resource_type else 'resources'
        action: str = 'looking for'
        if node_type in REFABLE_NODE_TYPES:
            formatted_name: str = f'ref("{duped_name}")'
        elif node_type == NodeType.Source:
            duped_name = self.node_1.get_full_source_name()
            formatted_name = self.node_1.get_source_representation()
        elif node_type == NodeType.Documentation:
            formatted_name = f'doc("{duped_name}")'
        elif node_type == NodeType.Test and hasattr(self.node_1, 'test_metadata'):
            column_name: str = f'column "{self.node_1.column_name}" in ' if self.node_1.column_name else ''
            model_name: str = self.node_1.file_key_name
            duped_name = f'{self.node_1.name}" defined on {column_name}"{model_name}'
            action = 'running'
            formatted_name = 'tests'
        else:
            formatted_name = duped_name
        msg: str = (
            f'\ndbt found two {pluralized} with the name "{duped_name}".\n\n'
            f'Since these resources have the same name, dbt will be unable to find the correct resource\n'
            f'when {action} {formatted_name}.\n\n'
            f'To fix this, change the name of one of these resources:\n'
            f'- {self.node_1.unique_id} ({self.node_1.original_file_path})\n'
            f'- {self.node_2.unique_id} ({self.node_2.original_file_path})\n    '
        ).strip()
        return msg


class DuplicateVersionedUnversionedError(ParsingError):
    def __init__(self, versioned_node: Any, unversioned_node: Any) -> None:
        self.versioned_node: Any = versioned_node
        self.unversioned_node: Any = unversioned_node
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg: str = (
            f"""\ndbt found versioned and unversioned models with the name "{self.versioned_node.name}".\n
Since these resources have the same name, dbt will be unable to find the correct resource
when looking for ref('{self.versioned_node.name}').\n
To fix this, change the name of the unversioned resource
{self.unversioned_node.unique_id} ({self.unversioned_node.original_file_path})
or add the unversioned model to the versions in {self.versioned_node.patch_path}\n    """
        ).strip()
        return msg


class PropertyYMLError(CompilationError):
    def __init__(self, path: str, issue: str) -> None:
        self.path: str = path
        self.issue: str = issue
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg: str = (
            f'The yml property file at {self.path} is invalid because {self.issue}. '
            f'Please consult the documentation for more information on yml property file syntax:\n\n'
            'https://docs.getdbt.com/reference/configs-and-properties'
        )
        return msg


class ContractError(CompilationError):
    def __init__(self, yaml_columns: List[Dict[str, Any]], sql_columns: List[Dict[str, Any]]) -> None:
        self.yaml_columns: List[Dict[str, Any]] = yaml_columns
        self.sql_columns: List[Dict[str, Any]] = sql_columns
        super().__init__(msg=self.get_message())

    def get_mismatches(self) -> Any:
        from dbt_common.clients.agate_helper import table_from_data_flat

        column_names: List[str] = ['column_name', 'definition_type', 'contract_type', 'mismatch_reason']
        mismatches: List[Dict[str, Any]] = []
        sql_col_set: set = set()
        for sql_col in self.sql_columns:
            sql_col_set.add(sql_col['name'])
            for i, yaml_col in enumerate(self.yaml_columns):
                if sql_col['name'] == yaml_col['name']:
                    if sql_col['data_type'] == yaml_col['data_type']:
                        break
                    else:
                        row: List[Any] = [sql_col['name'], sql_col['data_type'], yaml_col['data_type'], 'data type mismatch']
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

    def get_message(self) -> str:
        if not self.yaml_columns:
            return "This model has an enforced contract, and its 'columns' specification is missing"
        table = self.get_mismatches()
        output: Any = io.StringIO()
        table.print_table(output=output, max_rows=None, max_column_width=50)
        mismatches: str = output.getvalue()
        msg: str = (
            "This model has an enforced contract that failed.\n"
            "Please ensure the name, data_type, and number of columns in your contract match the columns in your model's definition.\n\n"
            f"{mismatches}"
        )
        return msg


class UnknownAsyncIDException(Exception):
    CODE: int = 10012
    MESSAGE: str = 'RPC server got an unknown async ID'

    def __init__(self, task_id: Any) -> None:
        self.task_id: Any = task_id

    def __str__(self) -> str:
        return f'{self.MESSAGE}: {self.task_id}'


class RPCFailureResult(DbtRuntimeError):
    CODE: int = 10002
    MESSAGE: str = 'RPC execution error'


class RPCTimeoutException(DbtRuntimeError):
    CODE: int = 10008
    MESSAGE: str = 'RPC timeout error'

    def __init__(self, timeout: Optional[int] = None) -> None:
        super().__init__(self.MESSAGE)
        self.timeout: Optional[int] = timeout

    def data(self) -> Dict[str, Any]:
        result: Dict[str, Any] = super().data()
        result.update({'timeout': self.timeout, 'message': f'RPC timed out after {self.timeout}s'})
        return result


class RPCKilledException(DbtRuntimeError):
    CODE: int = 10009
    MESSAGE: str = 'RPC process killed'

    def __init__(self, signum: Any) -> None:
        self.signum: Any = signum
        self.msg: str = f'RPC process killed by signal {self.signum}'
        super().__init__(self.msg)

    def data(self) -> Dict[str, Any]:
        return {'signum': self.signum, 'message': self.msg}


class RPCCompiling(DbtRuntimeError):
    CODE: int = 10010
    MESSAGE: str = 'RPC server is compiling the project, call the "status" method for compile status'

    def __init__(self, msg: Optional[str] = None, node: Optional[Any] = None) -> None:
        if msg is None:
            msg = 'compile in progress'
        super().__init__(msg, node)


class RPCLoadException(DbtRuntimeError):
    CODE: int = 10011
    MESSAGE: str = 'RPC server failed to compile project, call the "status" method for compile status'

    def __init__(self, cause: Dict[str, Any]) -> None:
        self.cause: Dict[str, Any] = cause
        self.msg: str = f"{self.MESSAGE}: {self.cause['message']}"
        super().__init__(self.msg)

    def data(self) -> Dict[str, Any]:
        return {'cause': self.cause, 'message': self.msg}