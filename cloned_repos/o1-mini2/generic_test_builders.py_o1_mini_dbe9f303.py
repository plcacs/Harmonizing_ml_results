import re
from copy import deepcopy
from typing import Any, Dict, Generic, List, Optional, Tuple, Type, TypeVar

from dbt.artifacts.resources import NodeVersion
from dbt.clients.jinja import GENERIC_TEST_KWARGS_NAME, get_rendered
from dbt.contracts.graph.nodes import UnpatchedSourceDefinition
from dbt.contracts.graph.unparsed import UnparsedModelUpdate, UnparsedNodeUpdate
from dbt.exceptions import (
    CustomMacroPopulatingConfigValueError,
    SameKeyNestedError,
    TagNotStringError,
    TagsNotListOfStringsError,
    TestArgIncludesModelError,
    TestArgsNotDictError,
    TestDefinitionDictLengthError,
    TestNameNotStringError,
    TestTypeError,
    UnexpectedTestNamePatternError,
)
from dbt.parser.common import Testable
from dbt.utils import md5
from dbt_common.exceptions.macros import UndefinedMacroError


T = TypeVar("T")

def synthesize_generic_test_names(
    test_type: str, test_name: str, args: Dict[str, Any]
) -> Tuple[str, str]:
    # Using the type, name, and arguments to this generic test, synthesize a (hopefully) unique name
    # Will not be unique if multiple tests have same name + arguments, and only configs differ
    # Returns a shorter version (hashed/truncated, for the compiled file)
    # as well as the full name (for the unique_id + FQN)
    flat_args: List[str] = []
    for arg_name in sorted(args):
        # the model is already embedded in the name, so skip it
        if arg_name == "model":
            continue
        arg_val = args[arg_name]

        if isinstance(arg_val, dict):
            parts = list(arg_val.values())
        elif isinstance(arg_val, (list, tuple)):
            parts = list(arg_val)
        else:
            parts = [arg_val]

        flat_args.extend([str(part) for part in parts])

    clean_flat_args: List[str] = [re.sub("[^0-9a-zA-Z_]+", "_", arg) for arg in flat_args]
    unique: str = "__".join(clean_flat_args)

    # for the file path + alias, the name must be <64 characters
    # if the full name is too long, include the first 30 identifying chars plus
    # a 32-character hash of the full contents

    test_identifier: str = f"{test_type}_{test_name}"
    full_name: str = f"{test_identifier}_{unique}"

    if len(full_name) >= 64:
        test_trunc_identifier: str = test_identifier[:30]
        label: str = md5(full_name)
        short_name: str = f"{test_trunc_identifier}_{label}"
    else:
        short_name: str = full_name

    return short_name, full_name


class TestBuilder(Generic[T]):
    """An object to hold assorted test settings and perform basic parsing

    Test names have the following pattern:
        - the test name itself may be namespaced (package.test)
        - or it may not be namespaced (test)

    """

    # The 'test_name' is used to find the 'macro' that implements the test
    TEST_NAME_PATTERN: re.Pattern = re.compile(
        r"((?P<test_namespace>([a-zA-Z_][0-9a-zA-Z_]*))\.)?"
        r"(?P<test_name>([a-zA-Z_][0-9a-zA-Z_]*))"
    )
    # args in the test entry representing test configs
    CONFIG_ARGS: Tuple[str, ...] = (
        "severity",
        "tags",
        "enabled",
        "where",
        "limit",
        "warn_if",
        "error_if",
        "fail_calc",
        "store_failures",
        "store_failures_as",
        "meta",
        "database",
        "schema",
        "alias",
    )

    def __init__(
        self,
        data_test: Dict[str, Any],
        target: T,
        package_name: str,
        render_ctx: Dict[str, Any],
        column_name: Optional[str] = None,
        version: Optional[NodeVersion] = None,
    ) -> None:
        test_name: str
        test_args: Dict[str, Any]
        test_name, test_args = self.extract_test_args(data_test, column_name)
        self.args: Dict[str, Any] = test_args
        if "model" in self.args:
            raise TestArgIncludesModelError()
        self.package_name: str = package_name
        self.target: T = target
        self.version: Optional[NodeVersion] = version
        self.render_ctx: Dict[str, Any] = render_ctx
        self.column_name: Optional[str] = column_name
        self.args["model"] = self.build_model_str()

        match = self.TEST_NAME_PATTERN.match(test_name)
        if match is None:
            raise UnexpectedTestNamePatternError(test_name)

        groups: Dict[str, Optional[str]] = match.groupdict()
        self.name: str = groups["test_name"]  # type: ignore
        self.namespace: Optional[str] = groups["test_namespace"]
        self.config: Dict[str, Any] = {}
        # Process legacy args
        self.config.update(self._process_legacy_args())

        # Process config args if present
        if "config" in self.args:
            config_value: Dict[str, Any] = self.args.pop("config", {})
            self.config.update(self._render_values(config_value))

        if self.namespace is not None:
            self.package_name = self.namespace

        # If the user has provided a description for this generic test, use it
        # Then delete the "description" argument to:
        # 1. Avoid passing it into the test macro
        # 2. Avoid passing it into the test name synthesis
        # Otherwise, use an empty string
        self.description: str = ""

        if "description" in self.args:
            self.description = self.args["description"]
            del self.args["description"]

        # If the user has provided a custom name for this generic test, use it
        # Then delete the "name" argument to avoid passing it into the test macro
        # Otherwise, use an auto-generated name synthesized from test inputs
        self.compiled_name: str = ""
        self.fqn_name: str = ""

        if "name" in self.args:
            # Assign the user-defined name here, which will be checked for uniqueness later
            # we will raise an error if two tests have same name for same model + column combo
            self.compiled_name = self.args["name"]
            self.fqn_name = self.args["name"]
            del self.args["name"]
        else:
            short_name: str
            full_name: str
            short_name, full_name = self.get_synthetic_test_names()
            self.compiled_name = short_name
            self.fqn_name = full_name
            # use hashed name as alias if full name is too long
            if short_name != full_name and "alias" not in self.config:
                self.config["alias"] = short_name

    def _process_legacy_args(self) -> Dict[str, Any]:
        config: Dict[str, Any] = {}
        for key in self.CONFIG_ARGS:
            value: Any = self.args.pop(key, None)
            if value and "config" in self.args and key in self.args["config"]:
                raise SameKeyNestedError()
            if not value and "config" in self.args:
                value = self.args["config"].pop(key, None)
            config[key] = value

        return self._render_values(config)

    def _render_values(self, config: Dict[str, Any]) -> Dict[str, Any]:
        rendered_config: Dict[str, Any] = {}
        for key, value in config.items():
            if isinstance(value, str):
                try:
                    value = get_rendered(value, self.render_ctx, native=True)
                except UndefinedMacroError as e:
                    raise CustomMacroPopulatingConfigValueError(
                        target_name=self.target.name,  # type: ignore
                        column_name=self.column_name,
                        name=self.name,
                        key=key,
                        err_msg=e.msg,
                    )
            if value is not None:
                rendered_config[key] = value
        return rendered_config

    def _bad_type(self) -> TypeError:
        return TypeError(f'invalid target type "{type(self.target)}"')

    @staticmethod
    def extract_test_args(data_test: Dict[str, Any], name: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        if not isinstance(data_test, dict):
            raise TestTypeError(data_test)

        # If the test is a dictionary with top-level keys, the test name is "test_name"
        # and the rest are arguments
        # {'name': 'my_favorite_test', 'test_name': 'unique', 'config': {'where': '1=1'}}
        if "test_name" in data_test.keys():
            test_name: str = data_test.pop("test_name")
            test_args: Dict[str, Any] = data_test
        # If the test is a nested dictionary with one top-level key, the test name
        # is the dict name, and nested keys are arguments
        # {'unique': {'name': 'my_favorite_test', 'config': {'where': '1=1'}}}
        else:
            data_test_items: List[Tuple[str, Any]] = list(data_test.items())
            if len(data_test_items) != 1:
                raise TestDefinitionDictLengthError(data_test)
            test_name, test_args = data_test_items[0]

        if not isinstance(test_args, dict):
            raise TestArgsNotDictError(test_args)
        if not isinstance(test_name, str):
            raise TestNameNotStringError(test_name)
        test_args_copy: Dict[str, Any] = deepcopy(test_args)
        if name is not None:
            test_args_copy["column_name"] = name
        return test_name, test_args_copy

    def tags(self) -> List[str]:
        tags: Any = self.config.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]
        if not isinstance(tags, list):
            raise TagsNotListOfStringsError(tags)
        for tag in tags:
            if not isinstance(tag, str):
                raise TagNotStringError(tag)
        return tags[:]

    def macro_name(self) -> str:
        macro_name: str = f"test_{self.name}"
        if self.namespace is not None:
            macro_name = f"{self.namespace}.{macro_name}"
        return macro_name

    def get_synthetic_test_names(self) -> Tuple[str, str]:
        # Returns two names: shorter (for the compiled file), full (for the unique_id + FQN)
        target_name: str = self.target.name  # type: ignore
        if isinstance(self.target, UnparsedModelUpdate):
            name: str = self.name
            if self.version:
                target_name = f"{self.target.name}_v{self.version}"  # type: ignore
        elif isinstance(self.target, UnparsedNodeUpdate):
            name = self.name
        elif isinstance(self.target, UnpatchedSourceDefinition):
            name = f"source_{self.name}"
        else:
            raise self._bad_type()
        if self.namespace is not None:
            name = f"{self.namespace}_{name}"
        return synthesize_generic_test_names(name, target_name, self.args)

    def construct_config(self) -> str:
        configs: List[str] = [
            f"{key}="
            + (
                f'"{value.replace(\'"\', \'\\\\"\')}"'
                if isinstance(value, str)
                else str(value)
            )
            for key, value in self.config.items()
        ]
        if configs:
            return f"{{{{ config({','.join(configs)}) }}}}"
        else:
            return ""

    # this is the 'raw_code' that's used in 'render_update' and execution
    # of the test macro
    def build_raw_code(self) -> str:
        return "{{{{ {macro}(**{kwargs_name}) }}}}{config}".format(
            macro=self.macro_name(),
            config=self.construct_config(),
            kwargs_name=GENERIC_TEST_KWARGS_NAME,
        )

    def build_model_str(self) -> str:
        targ: T = self.target
        if isinstance(targ, UnparsedModelUpdate):
            if self.version:
                target_str: str = f"ref('{targ.name}', version='{self.version}')"
            else:
                target_str = f"ref('{targ.name}')"
        elif isinstance(targ, UnparsedNodeUpdate):
            target_str = f"ref('{targ.name}')"
        elif isinstance(targ, UnpatchedSourceDefinition):
            target_str = f"source('{targ.source.name}', '{targ.table.name}')"
        else:
            raise self._bad_type()
        return f"{{{{ get_where_subquery({target_str}) }}}}"
