import io
import json
import re
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Union
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

    breaking_changes: List[str]

    def __init__(self, breaking_changes: List[str], node: Optional[Any] = None) -> None:
        self.breaking_changes = breaking_changes
        super().__init__(self.message(), node)

    @property
    def type(self) -> str:
        return 'Breaking change to contract'

    def message(self) -> str:
        reasons = '\n  - '.join(self.breaking_changes)
        return (
            f'While comparing to previous project state, dbt detected a breaking change to an enforced contract.\n'
            f'  - {reasons}\n'
            f'Consider making an additive (non-breaking) change instead, if possible.\n'
            f'Otherwise, create a new model version: https://docs.getdbt.com/docs/collaborate/govern/model-versions'
        )


class ParsingError(DbtRuntimeError):
    CODE: int = 10015
    MESSAGE: str = 'Parsing Error'

    def __init__(self, msg: Optional[str] = None, node: Optional[Any] = None) -> None:
        super().__init__(msg or self.MESSAGE, node)

    @property
    def type(self) -> str:
        return 'Parsing'


class dbtPluginError(DbtRuntimeError):
    CODE: int = 10020
    MESSAGE: str = 'Plugin Error'


class JSONValidationError(DbtValidationError):
    typename: str
    errors: List[str]
    errors_message: str

    def __init__(self, typename: str, errors: List[str]) -> None:
        self.typename = typename
        self.errors = errors
        self.errors_message = ', '.join(errors)
        msg = f'Invalid arguments passed to "{self.typename}" instance: {self.errors_message}'
        super().__init__(msg)

    def __reduce__(self) -> Any:
        return (JSONValidationError, (self.typename, self.errors))


class AliasError(DbtValidationError):
    pass


class DependencyError(Exception):
    CODE: int = 10006
    MESSAGE: str = 'Dependency Error'

    def __init__(self, msg: Optional[str] = None) -> None:
        super().__init__(msg or self.MESSAGE)


class FailFastError(DbtRuntimeError):
    CODE: int = 10013
    MESSAGE: str = 'FailFast Error'

    result: Optional[Any]

    def __init__(self, msg: str, result: Optional[Any] = None, node: Optional[Any] = None) -> None:
        super().__init__(msg=msg, node=node)
        self.result = result

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

    name: str

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(name)

    def get_message(self) -> str:
        msg = f'Invalid selector name: {self.name}'
        return msg


class DuplicateYamlKeyError(CompilationError):
    pass


class GraphDependencyNotFoundError(CompilationError):

    node: Any
    dependency: str

    def __init__(self, node: Any, dependency: str) -> None:
        self.node = node
        self.dependency = dependency
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg = f"'{self.node.unique_id}' depends on '{self.dependency}' which is not in the graph!"
        return msg


class ForeignKeyConstraintToSyntaxError(CompilationError):

    node: Any
    expression: str

    def __init__(self, node: Any, expression: str) -> None:
        self.expression = expression
        self.node = node
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg = (
            f"'{self.node.unique_id}' defines a foreign key constraint 'to' expression which is not valid "
            f"'ref' or 'source' syntax: {self.expression}."
        )
        return msg


class NoSupportedLanguagesFoundError(CompilationError):

    node: Any
    msg: str

    def __init__(self, node: Any) -> None:
        self.node = node
        self.msg = f'No supported_languages found in materialization macro {self.node.name}'
        super().__init__(msg=self.msg)


class MaterializtionMacroNotUsedError(CompilationError):

    node: Any
    msg: str

    def __init__(self, node: Any) -> None:
        self.node = node
        self.msg = 'Only materialization macros can be used with this function'
        super().__init__(msg=self.msg)


class MacroNamespaceNotStringError(CompilationError):

    kwarg_type: str

    def __init__(self, kwarg_type: str) -> None:
        self.kwarg_type = kwarg_type
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg = f'The macro_namespace parameter to adapter.dispatch is a {self.kwarg_type}, not a string'
        return msg


class UnknownGitCloningProblemError(DbtRuntimeError):

    repo: str

    def __init__(self, repo: str) -> None:
        self.repo = scrub_secrets(repo, env_secrets())
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg = (
            f'        Something went wrong while cloning {self.repo}\n'
            f'        Check the debug logs for more information\n        '
        )
        return msg


class NoAdaptersAvailableError(DbtRuntimeError):

    def __init__(self) -> None:
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg = (
            'No adapters available. Learn how to install an adapter by going to '
            'https://docs.getdbt.com/docs/connect-adapters#install-using-the-cli'
        )
        return msg


class BadSpecError(DbtInternalError):

    repo: str
    revision: str
    stderr: str

    def __init__(self, repo: str, revision: str, error: Any) -> None:
        self.repo = repo
        self.revision = revision
        self.stderr = scrub_secrets(error.stderr.strip(), env_secrets())
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg = f"Error checking out spec='{self.revision}' for repo {self.repo}\n{self.stderr}"
        return msg


class GitCloningError(DbtInternalError):

    repo: str
    revision: str
    error: Any

    def __init__(self, repo: str, revision: str, error: Any) -> None:
        self.repo = repo
        self.revision = revision
        self.error = error
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        stderr: str = self.error.stderr.strip()
        if 'usage: git' in stderr:
            stderr = stderr.split('\nusage: git')[0]
        if re.match(r"fatal: destination path '(.+)' already exists", stderr):
            self.error.cmd = list(scrub_secrets(str(self.error.cmd), env_secrets()))
            raise self.error
        msg = f"Error checking out spec='{self.revision}' for repo {self.repo}\n{stderr}"
        return scrub_secrets(msg, env_secrets())


class GitCheckoutError(BadSpecError):
    pass


class OperationError(CompilationError):

    operation_name: str

    def __init__(self, operation_name: str) -> None:
        self.operation_name = operation_name
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg = (
            f'dbt encountered an error when attempting to create a {self.operation_name}.\n'
            'If this error persists, please create an issue at: \n\n'
            'https://github.com/dbt-labs/dbt-core'
        )
        return msg


class ZipStrictWrongTypeError(CompilationError):

    exc: Any

    def __init__(self, exc: Any) -> None:
        self.exc = exc
        msg = str(self.exc)
        super().__init__(msg=msg)


class SetStrictWrongTypeError(CompilationError):

    exc: Any

    def __init__(self, exc: Any) -> None:
        self.exc = exc
        msg = str(self.exc)
        super().__init__(msg=msg)


class LoadAgateTableValueError(CompilationError):

    exc: Any
    node: Any

    def __init__(self, exc: Any, node: Any) -> None:
        self.exc = exc
        self.node = node
        msg = str(self.exc)
        super().__init__(msg=msg)


class LoadAgateTableNotSeedError(CompilationError):

    resource_type: str
    node: Any
    msg: str

    def __init__(self, resource_type: str, node: Any) -> None:
        self.resource_type = resource_type
        self.node = node
        msg = f'can only load_agate_table for seeds (got a {self.resource_type})'
        super().__init__(msg=msg)


class PackageNotInDepsError(CompilationError):

    package_name: str
    node: Any
    msg: str

    def __init__(self, package_name: str, node: Any) -> None:
        self.package_name = package_name
        self.node = node
        msg = f'Node package named {self.package_name} not found!'
        super().__init__(msg=msg)


class OperationsCannotRefEphemeralNodesError(CompilationError):

    target_name: str
    node: Any
    msg: str

    def __init__(self, target_name: str, node: Any) -> None:
        self.target_name = target_name
        self.node = node
        msg = f'Operations can not ref() ephemeral nodes, but {target_name} is ephemeral'
        super().__init__(msg=msg)


class PersistDocsValueTypeError(CompilationError):

    persist_docs: Any
    msg: str

    def __init__(self, persist_docs: Any) -> None:
        self.persist_docs = persist_docs
        msg = (
            f"Invalid value provided for 'persist_docs'. Expected dict but received {type(self.persist_docs)}"
        )
        super().__init__(msg=msg)


class InlineModelConfigError(CompilationError):

    node: Any
    msg: str

    def __init__(self, node: Any) -> None:
        self.node = node
        msg = 'Invalid inline model config'
        super().__init__(msg=msg)


class ConflictingConfigKeysError(CompilationError):

    oldkey: str
    newkey: str
    node: Any
    msg: str

    def __init__(self, oldkey: str, newkey: str, node: Any) -> None:
        self.oldkey = oldkey
        self.newkey = newkey
        self.node = node
        msg = f'Invalid config, has conflicting keys "{self.oldkey}" and "{self.newkey}"'
        super().__init__(msg=msg)


class NumberSourceArgsError(CompilationError):

    args: List[Any]
    node: Any
    msg: str

    def __init__(self, args: List[Any], node: Any) -> None:
        self.args = args
        self.node = node
        msg = f'source() takes exactly two arguments ({len(self.args)} given)'
        super().__init__(msg=msg)


class RequiredVarNotFoundError(CompilationError):

    var_name: str
    merged: Dict[str, Any]
    node: Optional[Any]
    msg: str

    def __init__(self, var_name: str, merged: Dict[str, Any], node: Optional[Any]) -> None:
        self.var_name = var_name
        self.merged = merged
        self.node = node
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        if self.node is not None:
            node_name = self.node.name
        else:
            node_name = '<Configuration>'
        dct = {k: self.merged[k] for k in self.merged}
        pretty_vars = json.dumps(dct, sort_keys=True, indent=4)
        msg = (
            f"Required var '{self.var_name}' not found in config:\n"
            f"Vars supplied to {node_name} = {pretty_vars}"
        )
        return scrub_secrets(msg, self.var_secrets())

    def var_secrets(self) -> List[str]:
        return [v for k, v in self.merged.items() if k.startswith(SECRET_ENV_PREFIX) and v.strip()]


class PackageNotFoundForMacroError(CompilationError):

    package_name: str
    msg: str

    def __init__(self, package_name: str) -> None:
        self.package_name = package_name
        msg = f"Could not find package '{self.package_name}'"
        super().__init__(msg=msg)


class SecretEnvVarLocationError(ParsingError):

    env_var_name: str

    def __init__(self, env_var_name: str) -> None:
        self.env_var_name = env_var_name
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg = (
            f"Secret env vars are allowed only in profiles.yml or packages.yml. "
            f"Found '{self.env_var_name}' referenced elsewhere."
        )
        return msg


class BooleanError(CompilationError):

    return_value: Any
    macro_name: str
    msg: str

    def __init__(self, return_value: Any, macro_name: str) -> None:
        self.return_value = return_value
        self.macro_name = macro_name
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg = (
            f"Macro '{self.macro_name}' returns '{self.return_value}'.  It is not type 'bool' "
            f"and cannot not be converted reliably to a bool."
        )
        return msg


class RefArgsError(CompilationError):

    node: Any
    args: List[Any]
    msg: str

    def __init__(self, node: Any, args: List[Any]) -> None:
        self.node = node
        self.args = args
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg = f'ref() takes at most two arguments ({len(self.args)} given)'
        return msg


class MetricArgsError(CompilationError):

    node: Any
    args: List[Any]
    msg: str

    def __init__(self, node: Any, args: List[Any]) -> None:
        self.node = node
        self.args = args
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg = f'metric() takes at most two arguments ({len(self.args)} given)'
        return msg


class RefBadContextError(CompilationError):

    node: Union[Any, Dict[str, Any]]
    args: List[Any]
    kwargs: Dict[str, Any]
    msg: str

    def __init__(self, node: Union[Any, Dict[str, Any]], args: Any) -> None:
        self.node = node
        self.args = args.positional_args
        self.kwargs = args.keyword_args
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
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
        msg = (
            f'dbt was unable to infer all dependencies for the model "{model_name}".\n'
            f'This typically happens when ref() is placed within a conditional block.\n\n'
            f'To fix this, add the following hint to the top of the model "{model_name}":\n\n'
            f'-- depends_on: {ref_string}'
        )
        return msg


class DocArgsError(CompilationError):

    node: Any
    args: List[Any]
    msg: str

    def __init__(self, node: Any, args: List[Any]) -> None:
        self.node = node
        self.args = args
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg = f'doc() takes at most two arguments ({len(self.args)} given)'
        return msg


class DocTargetNotFoundError(CompilationError):

    node: Any
    target_doc_name: str
    target_doc_package: Optional[str]
    msg: str

    def __init__(self, node: Any, target_doc_name: str, target_doc_package: Optional[str] = None) -> None:
        self.node = node
        self.target_doc_name = target_doc_name
        self.target_doc_package = target_doc_package
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        target_package_string = ''
        if self.target_doc_package is not None:
            target_package_string = f"in package '{self.target_doc_package}' "
        msg = (
            f"Documentation for '{self.node.unique_id}' depends on doc '{self.target_doc_name}' "
            f"{target_package_string}which was not found"
        )
        return msg


class MacroDispatchArgError(CompilationError):

    macro_name: str
    msg: str

    def __init__(self, macro_name: str) -> None:
        self.macro_name = macro_name
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg = (
            '        The "packages" argument of adapter.dispatch() has been deprecated.\n'
            '        Use the "macro_namespace" argument instead.\n\n'
            f'        Raised during dispatch for: {self.macro_name}\n\n'
            '        For more information, see:\n\n'
            '        https://docs.getdbt.com/reference/dbt-jinja-functions/dispatch\n        '
        )
        return msg


class DuplicateMacroNameError(CompilationError):

    node_1: Any
    node_2: Any
    namespace: str
    msg: str

    def __init__(self, node_1: Any, node_2: Any, namespace: str) -> None:
        self.node_1 = node_1
        self.node_2 = node_2
        self.namespace = namespace
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        duped_name = self.node_1.name
        if self.node_1.package_name != self.node_2.package_name:
            extra = f' ("{self.node_1.package_name}" and "{self.node_2.package_name}" are both in the "{self.namespace}" namespace)'
        else:
            extra = ''
        msg = (
            f'dbt found two macros with the name "{duped_name}" in the namespace "{self.namespace}"{extra}. '
            'Since these macros have the same name and exist in the same namespace, dbt will be unable to decide which to call. '
            'To fix this, change the name of one of these macros:\n'
            f'- {self.node_1.unique_id} ({self.node_1.original_file_path})\n'
            f'- {self.node_2.unique_id} ({self.node_2.original_file_path})'
        )
        return msg


class MacroResultAlreadyLoadedError(CompilationError):

    result_name: str
    msg: str

    def __init__(self, result_name: str) -> None:
        self.result_name = result_name
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg = f"The 'statement' result named '{self.result_name}' has already been loaded into a variable"
        return msg


class DictParseError(ParsingError):

    exc: Any
    node: Any

    def __init__(self, exc: Any, node: Any) -> None:
        self.exc = exc
        self.node = node
        msg = self.validator_error_message(exc)
        super().__init__(msg=msg)

    def validator_error_message(self, exc: Any) -> str:
        # Assuming the existence of this method
        return str(exc)


class ConfigUpdateError(ParsingError):

    exc: Any
    node: Any

    def __init__(self, exc: Any, node: Any) -> None:
        self.exc = exc
        self.node = node
        msg = self.validator_error_message(exc)
        super().__init__(msg=msg)

    def validator_error_message(self, exc: Any) -> str:
        # Assuming the existence of this method
        return str(exc)


class PythonParsingError(ParsingError):

    exc: Any
    node: Any

    def __init__(self, exc: Any, node: Any) -> None:
        self.exc = exc
        self.node = node
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        validated_exc = self.validator_error_message(self.exc)
        msg = f'{validated_exc}\n{self.exc.text}'
        return msg

    def validator_error_message(self, exc: Any) -> str:
        # Assuming the existence of this method
        return str(exc)


class PythonLiteralEvalError(ParsingError):

    exc: Any
    node: Any

    def __init__(self, exc: Any, node: Any) -> None:
        self.exc = exc
        self.node = node
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg = (
            'Error when trying to literal_eval an arg to dbt.ref(), dbt.source(), dbt.config() or dbt.config.get() \n'
            f'{self.exc}\n'
            'https://docs.python.org/3/library/ast.html#ast.literal_eval\n'
            'In dbt python model, `dbt.ref`, `dbt.source`, `dbt.config`, `dbt.config.get` function args only support Python literal structures'
        )
        return msg


class ModelConfigError(ParsingError):

    exc: Any
    node: Any
    msg: str

    def __init__(self, exc: Any, node: Any) -> None:
        self.msg = self.validator_error_message(exc)
        self.node = node
        super().__init__(msg=self.msg)

    def validator_error_message(self, exc: Any) -> str:
        # Assuming the existence of this method
        return str(exc)


class YamlParseListError(ParsingError):

    path: str
    key: str
    yaml_data: Any
    cause: Union[str, ValidationError]
    msg: str

    def __init__(self, path: str, key: str, yaml_data: Any, cause: Union[str, ValidationError]) -> None:
        self.path = path
        self.key = key
        self.yaml_data = yaml_data
        self.cause = cause
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        if isinstance(self.cause, str):
            reason = self.cause
        elif isinstance(self.cause, ValidationError):
            reason = self.validator_error_message(self.cause)
        else:
            reason = self.cause.msg
        msg = f'Invalid {self.key} config given in {self.path} @ {self.key}: {self.yaml_data} - {reason}'
        return msg

    def validator_error_message(self, exc: Any) -> str:
        # Assuming the existence of this method
        return str(exc)


class YamlParseDictError(ParsingError):

    path: str
    key: str
    yaml_data: Any
    cause: Union[str, ValidationError]
    msg: str

    def __init__(self, path: str, key: str, yaml_data: Any, cause: Union[str, ValidationError]) -> None:
        self.path = path
        self.key = key
        self.yaml_data = yaml_data
        self.cause = cause
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        if isinstance(self.cause, str):
            reason = self.cause
        elif isinstance(self.cause, ValidationError):
            reason = self.validator_error_message(self.cause)
        else:
            reason = self.cause.msg
        msg = f'Invalid {self.key} config given in {self.path} @ {self.key}: {self.yaml_data} - {reason}'
        return msg

    def validator_error_message(self, exc: Any) -> str:
        # Assuming the existence of this method
        return str(exc)


class YamlLoadError(ParsingError):

    path: str
    exc: Any
    project_name: Optional[str]
    msg: str

    def __init__(self, path: str, exc: Any, project_name: Optional[str] = None) -> None:
        self.project_name = project_name
        self.path = path
        self.exc = exc
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        reason = self.validator_error_message(self.exc)
        msg = f'Error reading {self.project_name}: {self.path} - {reason}'
        return msg

    def validator_error_message(self, exc: Any) -> str:
        # Assuming the existence of this method
        return str(exc)


class TestConfigError(ParsingError):

    exc: Any
    node: Any
    msg: str

    def __init__(self, exc: Any, node: Any) -> None:
        self.msg = self.validator_error_message(exc)
        self.node = node
        super().__init__(msg=self.msg)

    def validator_error_message(self, exc: Any) -> str:
        # Assuming the existence of this method
        return str(exc)


class SchemaConfigError(ParsingError):

    exc: Any
    node: Any
    msg: str

    def __init__(self, exc: Any, node: Any) -> None:
        self.msg = self.validator_error_message(exc)
        self.node = node
        super().__init__(msg=self.msg)

    def validator_error_message(self, exc: Any) -> str:
        # Assuming the existence of this method
        return str(exc)


class SnapshopConfigError(ParsingError):

    exc: Any
    node: Any
    msg: str

    def __init__(self, exc: Any, node: Any) -> None:
        self.msg = self.validator_error_message(exc)
        self.node = node
        super().__init__(msg=self.msg)

    def validator_error_message(self, exc: Any) -> str:
        # Assuming the existence of this method
        return str(exc)


class DbtReferenceError(ParsingError):

    unique_id: str
    ref_unique_id: str
    access: AccessType
    scope: str
    scope_type: str
    msg: str

    def __init__(
        self, unique_id: str, ref_unique_id: str, access: AccessType, scope: str
    ) -> None:
        self.unique_id = unique_id
        self.ref_unique_id = ref_unique_id
        self.access = access
        self.scope = scope
        self.scope_type = 'group' if self.access == AccessType.Private else 'package'
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        return (
            f"Node {self.unique_id} attempted to reference node {self.ref_unique_id}, which is not allowed "
            f"because the referenced node is {self.access} to the '{self.scope}' {self.scope_type}."
        )


class InvalidAccessTypeError(ParsingError):

    unique_id: str
    field_value: Any
    materialization: Optional[str]
    msg: str

    def __init__(self, unique_id: str, field_value: Any, materialization: Optional[str] = None) -> None:
        self.unique_id = unique_id
        self.field_value = field_value
        self.materialization = materialization
        with_materialization = f"with '{self.materialization}' materialization " if self.materialization else ''
        msg = f'Node {self.unique_id} {with_materialization}has an invalid value ({self.field_value}) for the access field'
        super().__init__(msg=msg)


class InvalidUnitTestGivenInput(ParsingError):

    input: str
    msg: str

    def __init__(self, input: str) -> None:
        self.input = input
        msg = f"Unit test given inputs must be either a 'ref', 'source' or 'this' call. Got: '{self.input}'."
        super().__init__(msg=msg)


class SameKeyNestedError(CompilationError):

    msg: str

    def __init__(self) -> None:
        msg = 'Test cannot have the same key at the top-level and in config'
        super().__init__(msg=msg)


class TestArgIncludesModelError(CompilationError):

    msg: str

    def __init__(self) -> None:
        msg = 'Test arguments include "model", which is a reserved argument'
        super().__init__(msg=msg)


class UnexpectedTestNamePatternError(CompilationError):

    test_name: str
    msg: str

    def __init__(self, test_name: str) -> None:
        self.test_name = test_name
        msg = f'Test name string did not match expected pattern: {self.test_name}'
        super().__init__(msg=msg)


class CustomMacroPopulatingConfigValueError(CompilationError):

    target_name: str
    name: str
    key: str
    err_msg: str
    column_name: Optional[str]
    msg: str

    def __init__(
        self,
        target_name: str,
        name: str,
        key: str,
        err_msg: str,
        column_name: Optional[str] = None,
    ) -> None:
        self.target_name = target_name
        self.column_name = column_name
        self.name = name
        self.key = key
        self.err_msg = err_msg
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg = (
            f'''The {self.target_name}.{self.column_name} column's "{self.name}" test references an undefined macro in its {self.key} configuration argument. The macro {self.err_msg}.\n'''
            'Please note that the generic test configuration parser currently does not support using custom macros to populate configuration values'
        )
        return msg


class TagsNotListOfStringsError(CompilationError):

    tags: Any
    msg: str

    def __init__(self, tags: Any) -> None:
        self.tags = tags
        msg = f'got {self.tags} ({type(self.tags)}) for tags, expected a list of strings'
        super().__init__(msg=msg)


class TagNotStringError(CompilationError):

    tag: Any
    msg: str

    def __init__(self, tag: Any) -> None:
        self.tag = tag
        msg = f'got {self.tag} ({type(self.tag)}) for tag, expected a str'
        super().__init__(msg=msg)


class TestNameNotStringError(ParsingError):

    test_name: Any
    msg: str

    def __init__(self, test_name: Any) -> None:
        self.test_name = test_name
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg = f'test name must be a str, got {type(self.test_name)} (value {self.test_name})'
        return msg


class TestArgsNotDictError(ParsingError):

    test_args: Any
    msg: str

    def __init__(self, test_args: Any) -> None:
        self.test_args = test_args
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg = f'test arguments must be a dict, got {type(self.test_args)} (value {self.test_args})'
        return msg


class TestDefinitionDictLengthError(ParsingError):

    test: Any
    msg: str

    def __init__(self, test: Any) -> None:
        self.test = test
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg = f'test definition dictionary must have exactly one key, got {self.test} instead ({len(self.test)} keys)'
        return msg


class TestTypeError(ParsingError):

    test: Any
    msg: str

    def __init__(self, test: Any) -> None:
        self.test = test
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg = f'test must be dict or str, got {type(self.test)} (value {self.test})'
        return msg


class EnvVarMissingError(ParsingError):

    var: str
    msg: str

    def __init__(self, var: str) -> None:
        self.var = var
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg = f"Env var required but not provided: '{self.var}'"
        return msg


class TargetNotFoundError(CompilationError):

    node: Any
    target_name: str
    target_kind: str
    target_package: Optional[str]
    target_version: Optional[str]
    disabled: Optional[bool]
    msg: str

    def __init__(
        self,
        node: Any,
        target_name: str,
        target_kind: str,
        target_package: Optional[str] = None,
        target_version: Optional[str] = None,
        disabled: Optional[bool] = None,
    ) -> None:
        self.node = node
        self.target_name = target_name
        self.target_kind = target_kind
        self.target_package = target_package
        self.target_version = target_version
        self.disabled = disabled
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
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
        msg = (
            f"{resource_type_title} '{unique_id}' ({original_file_path}) depends on a {self.target_kind} named "
            f"'{self.target_name}' {target_version_string}{target_package_string}which {reason}"
        )
        return msg


class DuplicateSourcePatchNameError(CompilationError):

    patch_1: Any
    patch_2: Any
    msg: str

    def __init__(self, patch_1: Any, patch_2: Any) -> None:
        self.patch_1 = patch_1
        self.patch_2 = patch_2
        super().__init__(msg=self.get_message())

    def _fix_dupe_msg(
        self, path1: str, path2: str, name: str, resource_type: str
    ) -> str:
        # Assuming implementation exists
        return "please resolve the duplicate paths."

    def get_message(self) -> str:
        name = f'{self.patch_1.overrides}.{self.patch_1.name}'
        fix = self._fix_dupe_msg(
            self.patch_1.path, self.patch_2.path, name, 'sources'
        )
        msg = (
            f'dbt found two schema.yml entries for the same source named {self.patch_1.name} '
            f'in package {self.patch_1.overrides}. Sources may only be overridden a single time. To fix this, {fix}'
        )
        return msg


class DuplicateMacroPatchNameError(CompilationError):

    patch_1: Any
    existing_patch_path: str
    msg: str

    def __init__(self, patch_1: Any, existing_patch_path: str) -> None:
        self.patch_1 = patch_1
        self.existing_patch_path = existing_patch_path
        super().__init__(msg=self.get_message())

    def _fix_dupe_msg(
        self, path1: str, path2: str, name: str, resource_type: str
    ) -> str:
        # Assuming implementation exists
        return "please resolve the duplicate paths."

    def get_message(self) -> str:
        package_name = self.patch_1.package_name
        name = self.patch_1.name
        fix = self._fix_dupe_msg(
            self.patch_1.original_file_path,
            self.existing_patch_path,
            name,
            'macros',
        )
        msg = (
            f'dbt found two schema.yml entries for the same macro in package {package_name} named {name}. '
            'Macros may only be described a single time. To fix this, {fix}'
        )
        return msg


class DuplicateAliasError(AliasError):

    kwargs: Dict[str, Any]
    aliases: Dict[str, str]
    canonical_key: str
    msg: str

    def __init__(
        self, kwargs: Dict[str, Any], aliases: Dict[str, str], canonical_key: str
    ) -> None:
        self.kwargs = kwargs
        self.aliases = aliases
        self.canonical_key = canonical_key
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        key_names = ', '.join(
            ('{}'.format(k) for k in self.kwargs if self.aliases.get(k) == self.canonical_key)
        )
        msg = f'Got duplicate keys: ({key_names}) all map to "{self.canonical_key}"'
        return msg


class MultipleVersionGitDepsError(DependencyError):

    git: Any
    requested: Any
    msg: str

    def __init__(self, git: Any, requested: Any) -> None:
        self.git = git
        self.requested = requested
        msg = f'git dependencies should contain exactly one version. {self.git} contains: {self.requested}'
        super().__init__(msg)

    def get_message(self) -> str:
        return self.msg


class DuplicateProjectDependencyError(DependencyError):

    project_name: str
    msg: str

    def __init__(self, project_name: str) -> None:
        self.project_name = project_name
        msg = f'Found duplicate project "{self.project_name}". This occurs when a dependency has the same project name as some other dependency.'
        super().__init__(msg)


class DuplicateDependencyToRootError(DependencyError):

    project_name: str
    msg: str

    def __init__(self, project_name: str) -> None:
        self.project_name = project_name
        msg = (
            f'Found a dependency with the same name as the root project "{self.project_name}". '
            'Package names must be unique in a project. Please rename one of these packages.'
        )
        super().__init__(msg)


class MismatchedDependencyTypeError(DependencyError):

    new: Any
    old: Any
    msg: str

    def __init__(self, new: Any, old: Any) -> None:
        self.new = new
        self.old = old
        msg = f'Cannot incorporate {self.new} ({self.new.__class__.__name__}) in {self.old} ({self.old.__class__.__name__}): mismatched types'
        super().__init__(msg)


class PackageVersionNotFoundError(DependencyError):

    package_name: str
    version_range: str
    available_versions: List[str]
    should_version_check: bool
    msg: str

    def __init__(
        self,
        package_name: str,
        version_range: str,
        available_versions: List[str],
        should_version_check: bool,
    ) -> None:
        self.package_name = package_name
        self.version_range = version_range
        self.available_versions = available_versions
        self.should_version_check = should_version_check
        super().__init__(self.get_message())

    def get_message(self) -> str:
        base_msg = (
            'Could not find a matching compatible version for package {}\n'
            '  Requested range: {}\n'
            '  Compatible versions: {}\n'
        )
        addendum = (
            "\n  Not shown: package versions incompatible with installed version of dbt-core\n"
            "  To include them, run 'dbt --no-version-check deps'"
            if self.should_version_check
            else ''
        )
        msg = base_msg.format(
            self.package_name, self.version_range, self.available_versions
        ) + addendum
        return msg


class PackageNotFoundError(DependencyError):

    package_name: str
    msg: str

    def __init__(self, package_name: str) -> None:
        self.package_name = package_name
        msg = f'Package {self.package_name} was not found in the package index'
        super().__init__(msg)


class ProfileConfigError(DbtProfileError):
    pass


class ProjectContractError(DbtProjectError):
    pass


class ProjectContractBrokenError(DbtProjectError):
    pass


class ConfigContractBrokenError(DbtProjectError):
    pass


class NonUniquePackageNameError(CompilationError):

    project_name: str
    msg: str

    def __init__(self, project_name: str) -> None:
        self.project_name = project_name
        msg = (
            f'dbt found more than one package with the name "{self.project_name}" included in this project. '
            'Package names must be unique in a project. Please rename one of these packages.'
        )
        super().__init__(msg=msg)


class UninstalledPackagesFoundError(CompilationError):

    count_packages_specified: int
    count_packages_installed: int
    packages_specified_path: str
    packages_install_path: str
    msg: str

    def __init__(
        self,
        count_packages_specified: int,
        count_packages_installed: int,
        packages_specified_path: str,
        packages_install_path: str,
    ) -> None:
        self.count_packages_specified = count_packages_specified
        self.count_packages_installed = count_packages_installed
        self.packages_specified_path = packages_specified_path
        self.packages_install_path = packages_install_path
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg = (
            f'dbt found {self.count_packages_specified} package(s) specified in {self.packages_specified_path}, '
            f'but only {self.count_packages_installed} package(s) installed in {self.packages_install_path}. '
            'Run "dbt deps" to install package dependencies.'
        )
        return msg


class OptionNotYamlDictError(CompilationError):

    var_type: Any
    option_name: str
    msg: str

    def __init__(self, var_type: Any, option_name: str) -> None:
        self.var_type = var_type
        self.option_name = option_name
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        type_name = self.var_type.__name__
        msg = f"The --{self.option_name} argument must be a YAML dictionary, but was of type '{type_name}'"
        return msg


class UnrecognizedCredentialTypeError(CompilationError):

    typename: str
    supported_types: List[str]
    msg: str

    def __init__(self, typename: str, supported_types: List[str]) -> None:
        self.typename = typename
        self.supported_types = supported_types
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg = 'Unrecognized credentials type "{}" - supported types are ({})'.format(
            self.typename,
            ', '.join(('"{}"'.format(t) for t in self.supported_types)),
        )
        return msg


class PatchTargetNotFoundError(CompilationError):

    patches: Dict[Any, Any]
    msg: str

    def __init__(self, patches: Dict[Any, Any]) -> None:
        self.patches = patches
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        patch_list = '\n\t'.join(
            (
                f'model {p.name} (referenced in path {p.original_file_path})'
                for p in self.patches.values()
            )
        )
        msg = f'dbt could not find models for the following patches:\n\t{patch_list}'
        return msg


class MissingRelationError(CompilationError):

    relation: Any
    model: Optional[Any]
    msg: str

    def __init__(self, relation: Any, model: Optional[Any] = None) -> None:
        self.relation = relation
        self.model = model
        msg = f'Relation {self.relation} not found!'
        super().__init__(msg=msg)


class AmbiguousAliasError(CompilationError):

    node_1: Any
    node_2: Any
    duped_name: str
    msg: str

    def __init__(self, node_1: Any, node_2: Any, duped_name: Optional[str] = None) -> None:
        self.node_1 = node_1
        self.node_2 = node_2
        if duped_name is None:
            self.duped_name = f'{self.node_1.database}.{self.node_1.schema}.{self.node_1.alias}'
        else:
            self.duped_name = duped_name
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg = (
            f'dbt found two resources with the database representation "{self.duped_name}".\n'
            'dbt cannot create two resources with identical database representations. To fix this,\n'
            'change the configuration of one of these resources:\n'
            f'- {self.node_1.unique_id} ({self.node_1.original_file_path})\n'
            f'- {self.node_2.unique_id} ({self.node_2.original_file_path})'
        )
        return msg


class AmbiguousResourceNameRefError(CompilationError):

    duped_name: str
    unique_ids: List[str]
    packages: List[str]
    node: Optional[Any]
    msg: str

    def __init__(
        self, duped_name: str, unique_ids: List[str], node: Optional[Any] = None
    ) -> None:
        self.duped_name = duped_name
        self.unique_ids = unique_ids
        self.packages = [unique_id.split('.')[1] for unique_id in unique_ids]
        super().__init__(msg=self.get_message(), node=node)

    def get_message(self) -> str:
        formatted_unique_ids = "'{0}'".format("', '".join(self.unique_ids))
        formatted_packages = "'{0}'".format("' or '".join(self.packages))
        msg = (
            f"When referencing '{self.duped_name}', dbt found nodes in multiple packages: {formatted_unique_ids}\n"
            f"To fix this, use two-argument 'ref', with the package name first: {formatted_packages}"
        )
        return msg


class AmbiguousCatalogMatchError(CompilationError):

    unique_id: str
    match_1: Dict[str, Any]
    match_2: Dict[str, Any]
    msg: str

    def __init__(
        self, unique_id: str, match_1: Dict[str, Any], match_2: Dict[str, Any]
    ) -> None:
        self.unique_id = unique_id
        self.match_1 = match_1
        self.match_2 = match_2
        super().__init__(msg=self.get_message())

    def get_match_string(self, match: Dict[str, Any]) -> str:
        match_schema = match.get('metadata', {}).get('schema')
        match_name = match.get('metadata', {}).get('name')
        return f'{match_schema}.{match_name}'

    def get_message(self) -> str:
        msg = (
            f'dbt found two relations in your warehouse with similar database identifiers. dbt\n'
            f'is unable to determine which of these relations was created by the model "{self.unique_id}".\n'
            'In order for dbt to correctly generate the catalog, one of the following relations must be deleted or renamed:\n\n'
            f' - {self.get_match_string(self.match_1)}\n'
            f' - {self.get_match_string(self.match_2)}'
        )
        return msg


class DependencyNotFoundError(CompilationError):

    node: Any
    node_description: str
    required_pkg: str
    msg: str

    def __init__(
        self, node: Any, node_description: str, required_pkg: str
    ) -> None:
        self.node = node
        self.node_description = node_description
        self.required_pkg = required_pkg
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg = (
            f'Error while parsing {self.node_description}.\n'
            f'The required package "{self.required_pkg}" was not found. Is the package installed?\n'
            'Hint: You may need to run `dbt deps`.'
        )
        return msg


class DuplicatePatchPathError(CompilationError):

    patch_1: Any
    existing_patch_path: str
    msg: str

    def __init__(self, patch_1: Any, existing_patch_path: str) -> None:
        self.patch_1 = patch_1
        self.existing_patch_path = existing_patch_path
        super().__init__(msg=self.get_message())

    def _fix_dupe_msg(
        self, path1: str, path2: str, name: str, resource_type: str
    ) -> str:
        # Assuming implementation exists
        return "please resolve the duplicate paths."

    def get_message(self) -> str:
        name = self.patch_1.name
        fix = self._fix_dupe_msg(
            self.patch_1.original_file_path,
            self.existing_patch_path,
            name,
            'resource',
        )
        msg = (
            f'dbt found two schema.yml entries for the same resource named {name}. '
            'Resources and their associated columns may only be described a single time. '
            f'To fix this, {fix}'
        )
        return msg


class DuplicateResourceNameError(CompilationError):

    node_1: Any
    node_2: Any
    msg: str

    def __init__(self, node_1: Any, node_2: Any) -> None:
        self.node_1 = node_1
        self.node_2 = node_2
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        duped_name = self.node_1.name
        node_type = NodeType(self.node_1.resource_type)
        if self.node_1.resource_type == self.node_2.resource_type:
            pluralized = node_type.pluralize()
        else:
            pluralized = 'resources'
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
        msg = (
            f'\ndbt found two {pluralized} with the name "{duped_name}".\n\n'
            f'Since these resources have the same name, dbt will be unable to find the correct resource\n'
            f'when {action} {formatted_name}.\n\n'
            'To fix this, change the name of one of these resources:\n'
            f'- {self.node_1.unique_id} ({self.node_1.original_file_path})\n'
            f'- {self.node_2.unique_id} ({self.node_2.original_file_path})\n    '
        ).strip()
        return msg


class DuplicateVersionedUnversionedError(ParsingError):

    versioned_node: Any
    unversioned_node: Any
    msg: str

    def __init__(self, versioned_node: Any, unversioned_node: Any) -> None:
        self.versioned_node = versioned_node
        self.unversioned_node = unversioned_node
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg = (
            f"""\ndbt found versioned and unversioned models with the name "{self.versioned_node.name}".\n\n"""
            f"""Since these resources have the same name, dbt will be unable to find the correct resource\n"""
            f"""when looking for ref('{self.versioned_node.name}').\n\n"""
            'To fix this, change the name of the unversioned resource\n'
            f'{self.unversioned_node.unique_id} ({self.unversioned_node.original_file_path})\n'
            f'or add the unversioned model to the versions in {self.versioned_node.patch_path}\n    '
        ).strip()
        return msg


class PropertyYMLError(CompilationError):

    path: str
    issue: str
    msg: str

    def __init__(self, path: str, issue: str) -> None:
        self.path = path
        self.issue = issue
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg = (
            f'The yml property file at {self.path} is invalid because {self.issue}. '
            'Please consult the documentation for more information on yml property file syntax:\n\n'
            'https://docs.getdbt.com/reference/configs-and-properties'
        )
        return msg


class ContractError(CompilationError):

    yaml_columns: List[Dict[str, Any]]
    sql_columns: List[Dict[str, Any]]
    msg: str

    def __init__(self, yaml_columns: List[Dict[str, Any]], sql_columns: List[Dict[str, Any]]) -> None:
        self.yaml_columns = yaml_columns
        self.sql_columns = sql_columns
        super().__init__(msg=self.get_message())

    def get_mismatches(self) -> Any:
        from dbt_common.clients.agate_helper import table_from_data_flat

        column_names = ['column_name', 'definition_type', 'contract_type', 'mismatch_reason']
        mismatches: List[Dict[str, Any]] = []
        sql_col_set = set()
        for sql_col in self.sql_columns:
            sql_col_set.add(sql_col['name'])
            for i, yaml_col in enumerate(self.yaml_columns):
                if sql_col['name'] == yaml_col['name']:
                    if sql_col['data_type'] == yaml_col['data_type']:
                        break
                    else:
                        row = {
                            'column_name': sql_col['name'],
                            'definition_type': sql_col['data_type'],
                            'contract_type': yaml_col['data_type'],
                            'mismatch_reason': 'data type mismatch',
                        }
                        mismatches.append(row)
                        break
                if i == len(self.yaml_columns) - 1:
                    row = {
                        'column_name': sql_col['name'],
                        'definition_type': sql_col['data_type'],
                        'contract_type': '',
                        'mismatch_reason': 'missing in contract',
                    }
                    mismatches.append(row)
        for yaml_col in self.yaml_columns:
            if yaml_col['name'] not in sql_col_set:
                row = {
                    'column_name': yaml_col['name'],
                    'definition_type': '',
                    'contract_type': yaml_col['data_type'],
                    'mismatch_reason': 'missing in definition',
                }
                mismatches.append(row)
        mismatches_sorted = sorted(mismatches, key=lambda d: d['column_name'])
        return table_from_data_flat(mismatches_sorted, column_names)

    def get_message(self) -> str:
        if not self.yaml_columns:
            return "This model has an enforced contract, and its 'columns' specification is missing"
        table = self.get_mismatches()
        output = io.StringIO()
        table.print_table(output=output, max_rows=None, max_column_width=50)
        mismatches = output.getvalue()
        msg = (
            "This model has an enforced contract that failed.\n"
            "Please ensure the name, data_type, and number of columns in your contract match the columns in your model's definition.\n\n"
            f"{mismatches}"
        )
        return msg


class UnknownAsyncIDException(Exception):
    CODE: int = 10012
    MESSAGE: str = 'RPC server got an unknown async ID'

    task_id: Any

    def __init__(self, task_id: Any) -> None:
        self.task_id = task_id

    def __str__(self) -> str:
        return f'{self.MESSAGE}: {self.task_id}'


class RPCFailureResult(DbtRuntimeError):
    CODE: int = 10002
    MESSAGE: str = 'RPC execution error'


class RPCTimeoutException(DbtRuntimeError):
    CODE: int = 10008
    MESSAGE: str = 'RPC timeout error'

    timeout: Optional[int]

    def __init__(self, timeout: Optional[int] = None) -> None:
        super().__init__(self.MESSAGE)
        self.timeout = timeout

    def data(self) -> Dict[str, Any]:
        result = super().data()
        result.update({'timeout': self.timeout, 'message': f'RPC timed out after {self.timeout}s'})
        return result


class RPCKilledException(DbtRuntimeError):
    CODE: int = 10009
    MESSAGE: str = 'RPC process killed'

    signum: Any
    msg: str

    def __init__(self, signum: Any) -> None:
        self.signum = signum
        self.msg = f'RPC process killed by signal {self.signum}'
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

    cause: Dict[str, Any]
    msg: str

    def __init__(self, cause: Dict[str, Any]) -> None:
        self.cause = cause
        self.msg = f'{self.MESSAGE}: {self.cause["message"]}'
        super().__init__(self.msg)

    def data(self) -> Dict[str, Any]:
        return {'cause': self.cause, 'message': self.msg}
