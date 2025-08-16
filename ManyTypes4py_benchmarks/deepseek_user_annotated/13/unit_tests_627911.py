import csv
import os
from copy import deepcopy
from csv import DictReader
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, cast

from dbt import utils
from dbt.artifacts.resources import ModelConfig, UnitTestConfig, UnitTestFormat
from dbt.config import RuntimeConfig
from dbt.context.context_config import ContextConfig
from dbt.context.providers import generate_parse_exposure, get_rendered
from dbt.contracts.files import FileHash, SchemaSourceFile
from dbt.contracts.graph.manifest import Manifest
from dbt.contracts.graph.model_config import UnitTestNodeConfig
from dbt.contracts.graph.nodes import (
    DependsOn,
    ModelNode,
    UnitTestDefinition,
    UnitTestNode,
    UnitTestSourceDefinition,
    SeedNode,
    SnapshotNode,
)
from dbt.contracts.graph.unparsed import UnparsedUnitTest, UnitTestGiven, UnitTestExpect
from dbt.exceptions import InvalidUnitTestGivenInput, ParsingError
from dbt.graph import UniqueId
from dbt.node_types import NodeType
from dbt.parser.schemas import (
    JSONValidationError,
    ParseResult,
    SchemaParser,
    ValidationError,
    YamlBlock,
    YamlParseDictError,
    YamlReader,
)
from dbt.utils import get_pseudo_test_path
from dbt_common.events.functions import fire_event
from dbt_common.events.types import SystemStdErr
from dbt_extractor import ExtractionError, py_extract_from_source  # type: ignore


class UnitTestManifestLoader:
    def __init__(self, manifest: Manifest, root_project: RuntimeConfig, selected: Set[UniqueId]) -> None:
        self.manifest: Manifest = manifest
        self.root_project: RuntimeConfig = root_project
        self.selected: Set[UniqueId] = selected
        self.unit_test_manifest = Manifest(macros=manifest.macros)

    def load(self) -> Manifest:
        for unique_id in self.selected:
            if unique_id in self.manifest.unit_tests:
                unit_test_case: UnitTestDefinition = self.manifest.unit_tests[unique_id]
                if not unit_test_case.config.enabled:
                    continue
                self.parse_unit_test_case(unit_test_case)
        return self.unit_test_manifest

    def parse_unit_test_case(self, test_case: UnitTestDefinition) -> None:
        tested_node_unique_id = test_case.depends_on.nodes[0]
        tested_node = self.manifest.nodes[tested_node_unique_id]
        assert isinstance(tested_node, ModelNode)

        name = test_case.name
        if tested_node.is_versioned:
            name = name + f"_v{tested_node.version}"
        expected_sql: Optional[str] = None
        expected_rows: List[Dict[str, Any]] = []
        if test_case.expect.format == UnitTestFormat.SQL:
            expected_sql = cast(str, test_case.expect.rows)
        else:
            assert isinstance(test_case.expect.rows, List)
            expected_rows = deepcopy(test_case.expect.rows)

        unit_test_node = UnitTestNode(
            name=name,
            resource_type=NodeType.Unit,
            package_name=test_case.package_name,
            path=get_pseudo_test_path(name, test_case.original_file_path),
            original_file_path=test_case.original_file_path,
            unique_id=test_case.unique_id,
            config=UnitTestNodeConfig(
                materialized="unit", expected_rows=expected_rows, expected_sql=expected_sql
            ),
            raw_code=tested_node.raw_code,
            database=tested_node.database,
            schema=tested_node.schema,
            alias=name,
            fqn=test_case.unique_id.split("."),
            checksum=FileHash.empty(),
            tested_node_unique_id=tested_node.unique_id,
            overrides=test_case.overrides,
        )

        ctx = generate_parse_exposure(
            unit_test_node,
            self.root_project,
            self.manifest,
            test_case.package_name,
        )
        get_rendered(unit_test_node.raw_code, ctx, unit_test_node, capture_macros=True)

        self.unit_test_manifest.nodes[unit_test_node.unique_id] = unit_test_node

        for given in test_case.given:
            original_input_node = self._get_original_input_node(
                given.input, tested_node, test_case.name
            )
            input_name = original_input_node.name
            common_fields: Dict[str, Any] = {
                "resource_type": NodeType.Model,
                "original_file_path": unit_test_node.original_file_path,
                "config": ModelConfig(materialized="ephemeral"),
                "database": original_input_node.database,
                "alias": original_input_node.identifier,
                "schema": original_input_node.schema,
                "fqn": original_input_node.fqn,
                "checksum": FileHash.empty(),
                "raw_code": self._build_fixture_raw_code(given.rows, None, given.format),
                "package_name": original_input_node.package_name,
                "unique_id": f"model.{original_input_node.package_name}.{input_name}",
                "name": input_name,
                "path": f"{input_name}.sql",
            }
            resource_type = original_input_node.resource_type

            if resource_type in (
                NodeType.Model,
                NodeType.Seed,
                NodeType.Snapshot,
            ):
                input_node = ModelNode(
                    **common_fields,
                    defer_relation=original_input_node.defer_relation,
                )
                if resource_type == NodeType.Model:
                    if original_input_node.version:
                        input_node.version = original_input_node.version
                    if original_input_node.latest_version:
                        input_node.latest_version = original_input_node.latest_version

            elif resource_type == NodeType.Source:
                input_node = UnitTestSourceDefinition(
                    **common_fields,
                    source_name=original_input_node.source_name,
                )
                self.unit_test_manifest.sources[input_node.unique_id] = input_node

            self.unit_test_manifest.nodes[input_node.unique_id] = input_node

            if original_input_node == tested_node:
                unit_test_node.this_input_node_unique_id = input_node.unique_id

            unit_test_node.depends_on.nodes.append(input_node.unique_id)

    def _build_fixture_raw_code(
        self, rows: Union[List[Dict[str, Any]], str, None], column_name_to_data_types: Optional[Dict[str, str]], fixture_format: UnitTestFormat
    ) -> str:
        if fixture_format == UnitTestFormat.SQL:
            return cast(str, rows)
        else:
            return ("{{{{ get_fixture_sql({rows}, {column_name_to_data_types}) }}}}").format(
                rows=rows, column_name_to_data_types=column_name_to_data_types
            )

    def _get_original_input_node(self, input: str, tested_node: ModelNode, test_case_name: str) -> Union[ModelNode, SeedNode, SnapshotNode, UnitTestSourceDefinition]:
        if input.strip() == "this":
            original_input_node = tested_node
        else:
            try:
                statically_parsed = py_extract_from_source(f"{{{{ {input} }}}}")
            except ExtractionError:
                raise InvalidUnitTestGivenInput(input=input)

            if statically_parsed["refs"]:
                ref = list(stically_parsed["refs"])[0]
                name = ref.get("name")
                package = ref.get("package")
                version = ref.get("version")
                original_input_node = self.manifest.ref_lookup.find(
                    name, package, version, self.manifest
                )
            elif statically_parsed["sources"]:
                source = list(stically_parsed["sources"])[0]
                input_source_name, input_name = source
                original_input_node = self.manifest.source_lookup.find(
                    f"{input_source_name}.{input_name}",
                    None,
                    self.manifest,
                )
            else:
                raise InvalidUnitTestGivenInput(input=input)

        if not original_input_node:
            msg = f"Unit test '{test_case_name}' had an input ({input}) which was not found in the manifest."
            raise ParsingError(msg)

        return original_input_node


class UnitTestParser(YamlReader):
    def __init__(self, schema_parser: SchemaParser, yaml: YamlBlock) -> None:
        super().__init__(schema_parser, yaml, "unit_tests")
        self.schema_parser = schema_parser
        self.yaml = yaml

    def parse(self) -> ParseResult:
        for data in self.get_key_dicts():
            unit_test: UnparsedUnitTest = self._get_unit_test(data)
            tested_model_node = find_tested_model_node(
                self.manifest, self.project.project_name, unit_test.model
            )
            unit_test_case_unique_id = (
                f"{NodeType.Unit}.{self.project.project_name}.{unit_test.model}.{unit_test.name}"
            )
            unit_test_fqn = self._build_fqn(
                self.project.project_name,
                self.yaml.path.original_file_path,
                unit_test.model,
                unit_test.name,
            )
            unit_test_config = self._build_unit_test_config(unit_test_fqn, unit_test.config)

            unit_test_definition = UnitTestDefinition(
                name=unit_test.name,
                model=unit_test.model,
                resource_type=NodeType.Unit,
                package_name=self.project.project_name,
                path=self.yaml.path.relative_path,
                original_file_path=self.yaml.path.original_file_path,
                unique_id=unit_test_case_unique_id,
                given=unit_test.given,
                expect=unit_test.expect,
                description=unit_test.description,
                overrides=unit_test.overrides,
                depends_on=DependsOn(),
                fqn=unit_test_fqn,
                config=unit_test_config,
                versions=unit_test.versions,
            )

            if tested_model_node:
                unit_test_definition.depends_on.nodes.append(tested_model_node.unique_id)
                unit_test_definition.schema = tested_model_node.schema

            self._validate_and_normalize_given(unit_test_definition)
            self._validate_and_normalize_expect(unit_test_definition)

            unit_test_definition.build_unit_test_checksum()
            assert isinstance(self.yaml.file, SchemaSourceFile)
            if unit_test_config.enabled:
                self.manifest.add_unit_test(self.yaml.file, unit_test_definition)
            else:
                self.manifest.add_disabled(self.yaml.file, unit_test_definition)

        return ParseResult()

    def _get_unit_test(self, data: Dict[str, Any]) -> UnparsedUnitTest:
        try:
            UnparsedUnitTest.validate(data)
            return UnparsedUnitTest.from_dict(data)
        except (ValidationError, JSONValidationError) as exc:
            raise YamlParseDictError(self.yaml.path, self.key, data, exc)

    def _build_unit_test_config(
        self, unit_test_fqn: List[str], config_dict: Dict[str, Any]
    ) -> UnitTestConfig:
        config = ContextConfig(
            self.schema_parser.root_project,
            unit_test_fqn,
            NodeType.Unit,
            self.schema_parser.project.project_name,
        )
        unit_test_config_dict = config.build_config_dict(patch_config_dict=config_dict)
        unit_test_config_dict = self.render_entry(unit_test_config_dict)

        return UnitTestConfig.from_dict(unit_test_config_dict)

    def _build_fqn(self, package_name: str, original_file_path: str, model_name: str, test_name: str) -> List[str]:
        path = Path(original_file_path)
        relative_path = str(path.relative_to(*path.parts[:1]))
        no_ext = os.path.splitext(relative_path)[0]
        fqn = [package_name]
        fqn.extend(utils.split_path(no_ext)[:-1])
        fqn.append(model_name)
        fqn.append(test_name)
        return fqn

    def _get_fixture(self, fixture_name: str, project_name: str) -> Any:
        fixture_unique_id = f"{NodeType.Fixture}.{project_name}.{fixture_name}"
        if fixture_unique_id in self.manifest.fixtures:
            fixture = self.manifest.fixtures[fixture_unique_id]
            return fixture
        else:
            raise ParsingError(
                f"File not found for fixture '{fixture_name}' in unit tests in {self.yaml.path.original_file_path}"
            )

    def _validate_and_normalize_given(self, unit_test_definition: UnitTestDefinition) -> None:
        for ut_input in unit_test_definition.given:
            self._validate_and_normalize_rows(ut_input, unit_test_definition, "input")

    def _validate_and_normalize_expect(self, unit_test_definition: UnitTestDefinition) -> None:
        self._validate_and_normalize_rows(
            unit_test_definition.expect, unit_test_definition, "expected"
        )

    def _validate_and_normalize_rows(self, ut_fixture: Union[UnitTestGiven, UnitTestExpect], unit_test_definition: UnitTestDefinition, fixture_type: str) -> None:
        if ut_fixture.format == UnitTestFormat.Dict:
            if ut_fixture.rows is None and ut_fixture.fixture is None:
                ut_fixture.rows = self._load_rows_from_seed(ut_fixture.input)
            if not isinstance(ut_fixture.rows, list):
                raise ParsingError(
                    f"Unit test {unit_test_definition.name} has {fixture_type} rows "
                    f"which do not match format {ut_fixture.format}"
                )
        elif ut_fixture.format == UnitTestFormat.CSV:
            if not (isinstance(ut_fixture.rows, str) or isinstance(ut_fixture.fixture, str)):
                raise ParsingError(
                    f"Unit test {unit_test_definition.name} has {fixture_type} rows or fixtures "
                    f"which do not match format {ut_fixture.format}.  Expected string."
                )

            if ut_fixture.fixture:
                csv_rows = self.get_fixture_file_rows(
                    ut_fixture.fixture, self.project.project_name, unit_test_definition.unique_id
                )
            else:
                csv_rows = self._convert_csv_to_list_of_dicts(cast(str, ut_fixture.rows))

            ut_fixture.rows = [
                {k: (None if v == "" else v) for k, v in row.items()} for row in csv_rows
            ]

        elif ut_fixture.format == UnitTestFormat.SQL:
            if not (isinstance(ut_fixture.rows, str) or isinstance(ut_fixture.fixture, str)):
                raise ParsingError(
                    f"Unit test {unit_test_definition.name} has {fixture_type} rows or fixtures "
                    f"which do not match format {ut_fixture.format}.  Expected string."
                )

            if ut_fixture.fixture:
                ut_fixture.rows = self.get_fixture_file_rows(
                    ut_fixture.fixture, self.project.project_name, unit_test_definition.unique_id
                )

        if ut_fixture.rows and (
            ut_fixture.format == UnitTestFormat.Dict or ut_fixture.format == UnitTestFormat.CSV
        ):
            self._promote_first_non_none_row(ut_fixture)

    def _promote_first_non_none_row(self, ut_fixture: Union[UnitTestGiven, UnitTestExpect]) -> None:
        non_none_row_index = None

        for index, row in enumerate(ut_fixture.rows):
            if all(value is not None for value in row.values()):
                non_none_row_index = index
                break

        if non_none_row_index is None:
            fire_event(
                SystemStdErr(
                    bmsg="Unit Test fixtures benefit from having at least one row free of Null values to ensure consistent column types. Failure to meet this recommendation can result in type mismatch errors between unit test source models and `expected` fixtures."
                )
            )
        else:
            ut_fixture.rows[0], ut_fixture.rows[non_none_row_index] = (
                ut_fixture.rows[non_none_row_index],
                ut_fixture.rows[0],
            )

    def get_fixture_file_rows(self, fixture_name: str, project_name: str, utdef_unique_id: str) -> List[Dict[str, Any]]:
        fixture = self._get_fixture(fixture_name, project_name)
        fixture_source_file = self.manifest.files[fixture.file_id]
        fixture_source_file.unit_tests.append(utdef_unique_id)
        return fixture.rows

    def _convert_csv_to_list_of_dicts(self, csv_string: str) -> List[Dict[str, Any]]:
        dummy_file = StringIO(csv_string)
        reader = csv.DictReader(dummy_file)
        rows = []
        for row in reader:
            rows.append(row)
        return rows

    def _load_rows_from_seed(self, ref_str: str) -> List[Dict[str, Any]]:
        ref = py_extract_from_source("{{ " + ref_str + " }}")["refs"][0]

        rows: List[Dict[str, Any]] = []

        seed_name = ref["name"]
        package_name = ref.get("package", self.project.project_name)

        seed_node = self.manifest.ref_lookup.find(seed_name, package_name, None, self.manifest)

        if not seed_node or seed_node.resource_type != NodeType.Seed:
            if package_name != self.project.project_name:
                raise ParsingError(
                    f"Unable to find seed '{package_name}.{seed_name}' for unit tests in '{package_name}' package"
                )
            else:
                raise ParsingError(
                    f"Unable to find seed '{package_name}.{seed_name}' for unit tests in directories: {self.project.seed_paths}"
                )

        seed_path = Path(seed_node.root_path) / seed_node.original_file_path
        with open(seed_path, "r") as f:
            for row in DictReader(f):
                rows.append(row)

        return rows


def find_tested_model_node(
    manifest: Manifest, current_project: str, unit_test_model: str
) -> Optional[ModelNode]:
    model_name_split = unit_test_model.split()
    model_name = model_name_split[0]
    model_version = model_name_split