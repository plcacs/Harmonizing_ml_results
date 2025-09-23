import io
import json
import re
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Union, Tuple
from dbt.node_types import REFABLE_NODE_TYPES, AccessType, NodeType
from dbt_common.constants import SECRET_ENV_PREFIX
from dbt_common.dataclass_schema import ValidationError
from dbt_common.exceptions import CommandResultError, CompilationError, DbtConfigError, DbtInternalError, DbtRuntimeError, DbtValidationError, env_secrets, scrub_secrets

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
        return f'While comparing to previous project state, dbt detected a breaking change to an enforced contract.\n  - {reasons}\nConsider making an additive (non-breaking) change instead, if possible.\nOtherwise, create a new model version: https://docs.getdbt.com/docs/collaborate/govern/model-versions'

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

    def __reduce__(self) -> Tuple[Any, ...]:
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
        msg: str = f"'{self.node.unique_id}' defines a foreign key constraint 'to' expression which is not valid 'ref' or 'source' syntax: {self.expression}."
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

    def __init__(self, kwarg_type: type) -> None:
        self.kwarg_type: type = kwarg_type
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg: str = f'The macro_namespace parameter to adapter.dispatch is a {self.kwarg_type}, not a string'
        return msg

class UnknownGitCloningProblemError(DbtRuntimeError):

    def __init__(self, repo: str) -> None:
        self.repo: str = scrub_secrets(repo, env_secrets())
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg: str = f'        Something went wrong while cloning {self.repo}\n        Check the debug logs for more information\n        '
        return msg

class NoAdaptersAvailableError(DbtRuntimeError):

    def __init__(self) -> None:
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg: str = 'No adapters available. Learn how to install an adapter by going to https://docs.getdbt.com/docs/connect-adapters#install-using-the-cli'
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
        msg: str = f'dbt encountered an error when attempting to create a {self.operation_name}. If this error persists, please create an issue at: \n\nhttps://github.com/dbt-labs/dbt-core'
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

    def __init__(self, var_name: str, merged: Dict[str, Any], node: Optional[Any] = None) -> None:
        self.var_name: str = var_name
        self.merged: Dict[str, Any] = merged
        self.node: Optional[Any] = node
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        if self.node is not None:
            node_name: str = self.node.name
        else:
            node_name = '<Configuration>'
        dct: Dict[str, Any] = {k: self.merged[k] for k in self.merged}
        pretty_vars: str = json.dumps(dct, sort_keys=True, indent=4)
        msg: str = f"Required var '{self.var_name}' not found in config:\nVars supplied to {node_name} = {pretty_vars}"
        return scrub_secrets(msg, self.var_secrets())

    def var_secrets(self) -> List[str]:
        return [v for (k, v) in self.merged.items() if k.startswith(SECRET_ENV_PREFIX) and v.strip()]

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
        msg: str = f"Macro '{self.macro_name}' returns '{self.return_value}'.  It is not type 'bool' and cannot not be converted reliably to a bool."
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
        self.args: List[Any] = args.positional_args
        self.kwargs: Dict[str, Any] = args.keyword_args
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        if isinstance(self.node, dict):
            model_name: str = self.node['name']
        else:
            model_name = self.node.name
        ref_args: str = ', '.join(("'{}'".format(a) for a in self.args))
        keyword_args: str = ''
        if self.kwargs:
            keyword_args = ', '.join(("{}='{}'".format(k, v) for (k, v) in self.kwargs.items()))
            keyword_args = ',' + keyword_args
        ref_string: str = f'{{{{ ref({ref_args}{keyword_args}) }}}}'
        msg: str = f'dbt was unable to infer all dependencies for the model "{model_name}".\nThis typically happens when ref() is placed within a conditional block.\n\nTo fix this, add the following hint to the top of the model "{model_name}":\n\n-- depends_on: {ref_string}'
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
        msg: str = f"Documentation for '{self.node.unique_id}' depends on doc '{self.target_doc_name}' {target_package_string}which was not found"
        return msg

class MacroDispatchArgError(CompilationError):

    def __init__(self, macro_name: str) -> None:
        self.macro_name: str = macro_name
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        msg: str = f'        The "packages" argument of adapter.dispatch() has been deprecated.\n        Use the "macro_namespace" argument instead.\n\n        Raised during dispatch for: {self.macro_name}\n\n        For more information, see:\n\n        https://docs.getdbt.com/reference/dbt-jinja-functions/dispatch\n        '
        return msg

class DuplicateMacroNameError(CompilationError):

    def __init__(self, node_1: Any, node_2: Any, namespace: str) -> None:
        self.node_1: Any = node_1
        self.node_2: Any = node_2
        self.namespace: str = namespace
        super().__init__(msg=self.get_message())

    def get_message(self) -> str:
        duped_name: str = self.node_