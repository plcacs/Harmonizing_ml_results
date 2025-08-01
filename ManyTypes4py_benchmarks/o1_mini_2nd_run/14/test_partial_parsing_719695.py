import os
import re
from argparse import Namespace
from typing import Any, Dict, List
from unittest import mock
import pytest
import yaml
import dbt.flags as flags
from dbt.contracts.files import ParseFileType
from dbt.contracts.results import TestStatus
from dbt.exceptions import CompilationError
from dbt.plugins.manifest import ModelNodeArgs, PluginNodes
from dbt.tests.fixtures.project import write_project_files
from dbt.tests.util import (
    get_manifest,
    rename_dir,
    rm_file,
    run_dbt,
    run_dbt_and_capture,
    write_file,
)
from tests.functional.partial_parsing.fixtures import (
    custom_schema_tests1_sql,
    custom_schema_tests2_sql,
    customers1_md,
    customers2_md,
    customers_sql,
    empty_schema_with_version_yml,
    empty_schema_yml,
    generic_schema_yml,
    generic_test_edited_sql,
    generic_test_schema_yml,
    generic_test_sql,
    gsm_override2_sql,
    gsm_override_sql,
    local_dependency__dbt_project_yml,
    local_dependency__macros__dep_macro_sql,
    local_dependency__models__model_to_import_sql,
    local_dependency__models__schema_yml,
    local_dependency__seeds__seed_csv,
    macros_schema_yml,
    macros_yml,
    model_a_sql,
    model_b_sql,
    model_four1_sql,
    model_four2_sql,
    model_one_sql,
    model_three_disabled2_sql,
    model_three_disabled_sql,
    model_three_modified_sql,
    model_three_sql,
    model_two_disabled_sql,
    model_two_sql,
    models_schema1_yml,
    models_schema2_yml,
    models_schema2b_yml,
    models_schema3_yml,
    models_schema4_yml,
    models_schema4b_yml,
    my_analysis_sql,
    my_macro2_sql,
    my_macro_sql,
    my_test_sql,
    orders_sql,
    raw_customers_csv,
    ref_override2_sql,
    ref_override_sql,
    schema_models_c_yml,
    schema_sources1_yml,
    schema_sources2_yml,
    schema_sources3_yml,
    schema_sources4_yml,
    schema_sources5_yml,
    snapshot2_sql,
    snapshot_sql,
    sources_tests1_sql,
    sources_tests2_sql,
    test_macro2_sql,
    test_macro_sql,
)
from tests.functional.utils import up_one

os.environ["DBT_PP_TEST"] = "true"


def normalize(path: str) -> str:
    return os.path.normcase(os.path.normpath(path))


class TestModels:
    @pytest.fixture(scope="class")
    def models(self) -> Dict[str, str]:
        return {"model_one.sql": model_one_sql}

    def test_pp_models(self, project: Any) -> None:
        results = run_dbt(["run"])
        assert len(results) == 1
        write_file(model_two_sql, project.project_root, "models", "model_two.sql")
        results = run_dbt(["--partial-parse", "run"])
        assert len(results) == 2
        write_file(models_schema1_yml, project.project_root, "models", "schema.yml")
        results = run_dbt(["--partial-parse", "run"])
        assert len(results) == 2
        manifest = get_manifest(project.project_root)
        assert "model.test.model_one" in manifest.nodes
        model_one_node = manifest.nodes["model.test.model_one"]
        assert model_one_node.description == "The first model"
        assert model_one_node.patch_path == "test://" + normalize("models/schema.yml")
        write_file(models_schema2_yml, project.project_root, "models", "schema.yml")
        write_file(model_three_sql, project.project_root, "models", "model_three.sql")
        results = run_dbt(["--partial-parse", "test"], expect_pass=False)
        assert len(results) == 1
        manifest = get_manifest(project.project_root)
        project_files = [f for f in manifest.files if f.startswith("test://")]
        assert len(project_files) == 4
        model_3_file_id = "test://" + normalize("models/model_three.sql")
        assert model_3_file_id in manifest.files
        model_three_file = manifest.files[model_3_file_id]
        assert model_three_file.parse_file_type == ParseFileType.Model
        assert type(model_three_file).__name__ == "SourceFile"
        model_three_node = manifest.nodes[model_three_file.nodes[0]]
        schema_file_id = "test://" + normalize("models/schema.yml")
        assert model_three_node.patch_path == schema_file_id
        assert model_three_node.description == "The third model"
        schema_file = manifest.files[schema_file_id]
        assert type(schema_file).__name__ == "SchemaSourceFile"
        assert len(schema_file.data_tests) == 1
        tests = schema_file.get_all_test_ids()
        assert tests == ["test.test.unique_model_three_id.6776ac8160"]
        unique_test_id = tests[0]
        assert unique_test_id in manifest.nodes
        write_file(model_three_modified_sql, project.project_root, "models", "model_three.sql")
        results = run_dbt(["--partial-parse", "run"])
        manifest = get_manifest(project.project_root)
        model_id = "model.test.model_three"
        assert model_id in manifest.nodes
        model_three_node = manifest.nodes[model_id]
        assert model_three_node.description == "The third model"
        write_file(models_schema2b_yml, project.project_root, "models", "schema.yml")
        results = run_dbt(["--partial-parse", "test"], expect_pass=False)
        manifest = get_manifest(project.project_root)
        schema_file_id = "test://" + normalize("models/schema.yml")
        schema_file = manifest.files[schema_file_id]
        tests = schema_file.get_all_test_ids()
        assert tests == ["test.test.not_null_model_three_id.3162ce0a6f"]
        not_null_test_id = tests[0]
        assert not_null_test_id in manifest.nodes.keys()
        assert unique_test_id not in manifest.nodes.keys()
        assert len(results) == 1
        write_file(models_schema1_yml, project.project_root, "models", "schema.yml")
        rm_file(project.project_root, "models", "model_three.sql")
        results = run_dbt(["--partial-parse", "run"])
        assert len(results) == 2
        write_file(model_three_sql, project.project_root, "models", "model_three.sql")
        rm_file(project.project_root, "models", "schema.yml")
        results = run_dbt(["--partial-parse", "run"])
        assert len(results) == 3
        manifest = get_manifest(project.project_root)
        schema_file_id = "test://" + normalize("models/schema.yml")
        assert schema_file_id not in manifest.files
        project_files = [f for f in manifest.files if f.startswith("test://")]
        assert len(project_files) == 3
        write_file(models_schema2_yml, project.project_root, "models", "schema.yml")
        rm_file(project.project_root, "models", "model_three.sql")
        with pytest.raises(CompilationError):
            results = run_dbt(["--partial-parse", "--warn-error", "run"])
        write_file(model_three_sql, project.project_root, "models", "model_three.sql")
        results = run_dbt(["--partial-parse", "run"])
        assert len(results) == 3
        write_file(model_four1_sql, project.project_root, "models", "model_four.sql")
        results = run_dbt(["--partial-parse", "run"])
        assert len(results) == 4
        rm_file(project.project_root, "models", "model_three.sql")
        write_file(model_four2_sql, project.project_root, "models", "model_four.sql")
        write_file(models_schema1_yml, project.project_root, "models", "schema.yml")
        results = run_dbt(["--partial-parse", "run"])
        assert len(results) == 3
        write_file(model_three_sql, project.project_root, "models", "model_three.sql")
        write_file(models_schema2_yml, project.project_root, "models", "schema.yml")
        rm_file(project.project_root, "models", "model_four.sql")
        results = run_dbt(["--partial-parse", "run"])
        assert len(results) == 3
        write_file(models_schema4_yml, project.project_root, "models", "schema.yml")
        results = run_dbt(["--partial-parse", "run"])
        assert len(results) == 2
        write_file(models_schema4b_yml, project.project_root, "models", "schema.yml")
        results = run_dbt(["--partial-parse", "run"])
        assert len(results) == 3
        write_file(models_schema4_yml, project.project_root, "models", "schema.yml")
        results = run_dbt(["--partial-parse", "run"])
        assert len(results) == 2
        write_file(models_schema4b_yml, project.project_root, "models", "schema.yml")
        results = run_dbt(["--partial-parse", "run"])
        assert len(results) == 3
        write_file(my_macro_sql, project.project_root, "macros", "my_macro.sql")
        results = run_dbt(["--partial-parse", "run"])
        assert len(results) == 3
        manifest = get_manifest(project.project_root)
        macro_id = "macro.test.do_something"
        assert macro_id in manifest.macros
        write_file(my_macro2_sql, project.project_root, "macros", "my_macro.sql")
        results = run_dbt(["--partial-parse", "run"])
        assert len(results) == 3
        write_file(models_schema3_yml, project.project_root, "models", "schema.yml")
        results = run_dbt(["--partial-parse", "run"])
        assert len(results) == 3
        rm_file(project.project_root, "macros", "my_macro.sql")
        with pytest.raises(CompilationError):
            results = run_dbt(["--partial-parse", "--warn-error", "run"])
        write_file(models_schema2_yml, project.project_root, "models", "schema.yml")
        write_file(my_macro_sql, project.project_root, "macros", "my_macro.sql")
        write_file(macros_yml, project.project_root, "macros", "macros.yml")
        results = run_dbt(["--partial-parse", "run"])
        rm_file(project.project_root, "macros", "my_macro.sql")
        rm_file(project.project_root, "macros", "macros.yml")
        results = run_dbt(["--partial-parse", "run"])
        assert len(results) == 3
        write_file(empty_schema_yml, project.project_root, "models", "eschema.yml")
        results = run_dbt(["--partial-parse", "run"])
        assert len(results) == 3
        write_file(empty_schema_with_version_yml, project.project_root, "models", "eschema.yml")
        results = run_dbt(["--partial-parse", "run"])
        assert len(results) == 3
        write_file(model_three_disabled_sql, project.project_root, "models", "model_three.sql")
        results = run_dbt(["--partial-parse", "run"])
        assert len(results) == 2
        manifest = get_manifest(project.project_root)
        model_id = "model.test.model_three"
        assert model_id in manifest.disabled
        assert model_id not in manifest.nodes
        write_file(model_three_disabled2_sql, project.project_root, "models", "model_three.sql")
        results = run_dbt(["--partial-parse", "run"])
        assert len(results) == 2
        manifest = get_manifest(project.project_root)
        model_id = "model.test.model_three"
        assert model_id in manifest.disabled
        assert model_id not in manifest.nodes
        write_file(model_three_sql, project.project_root, "models", "model_three.sql")
        results = run_dbt(["--partial-parse", "run"])
        assert len(results) == 3
        manifest = get_manifest(project.project_root)
        model_id = "model.test.model_three"
        assert model_id in manifest.nodes
        assert model_id not in manifest.disabled


class TestSources:
    @pytest.fixture(scope="class")
    def models(self) -> Dict[str, str]:
        return {"model_one.sql": model_one_sql}

    def test_pp_sources(self, project: Any) -> None:
        write_file(raw_customers_csv, project.project_root, "seeds", "raw_customers.csv")
        write_file(sources_tests1_sql, project.project_root, "macros", "tests.sql")
        results = run_dbt(["run"])
        assert len(results) == 1
        run_dbt(["--partial-parse", "seed"])
        manifest = get_manifest(project.project_root)
        seed_file_id = "test://" + normalize("seeds/raw_customers.csv")
        assert seed_file_id in manifest.files
        write_file(raw_customers_csv, project.project_root, "seeds", "more_customers.csv")
        run_dbt(["--partial-parse", "run"])
        seed_file_id = "test://" + normalize("seeds/more_customers.csv")
        manifest = get_manifest(project.project_root)
        assert seed_file_id in manifest.files
        seed_id = "seed.test.more_customers"
        assert seed_id in manifest.nodes
        rm_file(project.project_root, "seeds", "more_customers.csv")
        write_file(schema_sources1_yml, project.project_root, "models", "sources.yml")
        results = run_dbt(["--partial-parse", "run"])
        manifest = get_manifest(project.project_root)
        assert len(manifest.sources) == 1
        file_id = "test://" + normalize("models/sources.yml")
        assert file_id in manifest.files
        write_file(customers_sql, project.project_root, "models", "customers.sql")
        results = run_dbt(["--partial-parse", "run"])
        assert len(results) == 2
        rm_file(project.project_root, "models", "sources.yml")
        with pytest.raises(CompilationError):
            results = run_dbt(["--partial-parse", "run"])
        write_file(schema_sources2_yml, project.project_root, "models", "sources.yml")
        results = run_dbt(["--partial-parse", "run"])
        rm_file(project.project_root, "seeds", "raw_customers.csv")
        with pytest.raises(CompilationError):
            results = run_dbt(["--partial-parse", "run"])
        write_file(raw_customers_csv, project.project_root, "seeds", "raw_customers.csv")
        write_file(schema_sources3_yml, project.project_root, "models", "sources.yml")
        results = run_dbt(["--partial-parse", "run"])
        write_file(schema_sources4_yml, project.project_root, "models", "sources.yml")
        results = run_dbt(["--partial-parse", "run"])
        write_file(schema_sources5_yml, project.project_root, "models", "sources.yml")
        with pytest.raises(CompilationError):
            results = run_dbt(["--partial-parse", "--warn-error", "run"])
        write_file(schema_sources4_yml, project.project_root, "models", "sources.yml")
        results = run_dbt(["--partial-parse", "run"])
        write_file(customers1_md, project.project_root, "models", "customers.md")
        results = run_dbt(["--partial-parse", "run"])
        write_file(customers2_md, project.project_root, "models", "customers.md")
        results = run_dbt(["--partial-parse", "run"])
        rm_file(project.project_root, "models", "customers.md")
        results = run_dbt(["--partial-parse", "run"])
        write_file(test_macro_sql, project.project_root, "macros", "test-macro.sql")
        write_file(my_test_sql, project.project_root, "tests", "my_test.sql")
        results = run_dbt(["--partial-parse", "test"])
        manifest = get_manifest(project.project_root)
        assert len(manifest.nodes) == 9
        test_id = "test.test.my_test"
        assert test_id in manifest.nodes
        write_file(test_macro2_sql, project.project_root, "macros", "test-macro.sql")
        results = run_dbt(["--partial-parse", "test"])
        manifest = get_manifest(project.project_root)
        write_file(my_analysis_sql, project.project_root, "analyses", "my_analysis.sql")
        results = run_dbt(["--partial-parse", "run"])
        manifest = get_manifest(project.project_root)
        rm_file(project.project_root, "tests", "my_test.sql")
        results = run_dbt(["--partial-parse", "test"])
        manifest = get_manifest(project.project_root)
        assert len(manifest.nodes) == 9
        rm_file(project.project_root, "analyses", "my_analysis.sql")
        results = run_dbt(["--partial-parse", "run"])
        manifest = get_manifest(project.project_root)
        assert len(manifest.nodes) == 8
        write_file(sources_tests2_sql, project.project_root, "macros", "tests.sql")
        results = run_dbt(["--partial-parse", "run"])


class TestPartialParsingDependency:
    @pytest.fixture(scope="class")
    def models(self) -> Dict[str, str]:
        return {"model_one.sql": model_one_sql}

    @pytest.fixture(scope="class", autouse=True)
    def setUp(self, project_root: Any) -> None:
        local_dependency_files = {
            "dbt_project.yml": local_dependency__dbt_project_yml,
            "models": {
                "schema.yml": local_dependency__models__schema_yml,
                "model_to_import.sql": local_dependency__models__model_to_import_sql,
            },
            "macros": {"dep_macro.sql": local_dependency__macros__dep_macro_sql},
            "seeds": {"seed.csv": local_dependency__seeds__seed_csv},
        }
        write_project_files(project_root, "local_dependency", local_dependency_files)

    @pytest.fixture(scope="class")
    def packages(self) -> Dict[str, Any]:
        return {"packages": [{"local": "local_dependency"}]}

    def test_parsing_with_dependency(self, project: Any) -> None:
        run_dbt(["clean"])
        run_dbt(["deps"])
        run_dbt(["seed"])
        run_dbt(["run"])
        write_file(schema_models_c_yml, project.project_root, "models", "schema.yml")
        results = run_dbt(["--partial-parse", "run"])
        assert len(results) == 2
        manifest = get_manifest(project.project_root)
        assert len(manifest.sources) == 1
        source_id = "source.local_dep.seed_source.seed"
        assert source_id in manifest.sources
        assert len(manifest.nodes) == 5
        test_id = "test.local_dep.source_unique_seed_source_seed_id.afa94935ed"
        assert test_id in manifest.nodes
        rm_file(project.project_root, "models", "schema.yml")
        results = run_dbt(["--partial-parse", "run"])
        manifest = get_manifest(project.project_root)
        assert len(manifest.sources) == 1


class TestNestedMacros:
    @pytest.fixture(scope="class")
    def models(self) -> Dict[str, str]:
        return {
            "model_a.sql": model_a_sql,
            "model_b.sql": model_b_sql,
            "schema.yml": macros_schema_yml,
        }

    @pytest.fixture(scope="class")
    def macros(self) -> Dict[str, str]:
        return {"custom_schema_tests.sql": custom_schema_tests1_sql}

    def test_nested_macros(self, project: Any) -> None:
        results = run_dbt()
        assert len(results) == 2
        manifest = get_manifest(project.project_root)
        macro_child_map = manifest.build_macro_child_map()
        macro_unique_id = "macro.test.test_type_two"
        assert macro_unique_id in macro_child_map
        results = run_dbt(["test"], expect_pass=False)
        results = sorted(results, key=lambda r: r.node.name)
        assert len(results) == 2
        assert results[0].status == TestStatus.Fail
        assert re.search("union all", results[0].node.compiled_code)
        assert results[1].status == TestStatus.Warn
        assert results[1].node.config.severity == "WARN"
        write_file(custom_schema_tests2_sql, project.project_root, "macros", "custom_schema_tests.sql")
        results = run_dbt(["--partial-parse", "test"], expect_pass=False)
        manifest = get_manifest(project.project_root)
        test_node_id = "test.test.type_two_model_a_.842bc6c2a7"
        assert test_node_id in manifest.nodes
        results = sorted(results, key=lambda r: r.node.name)
        assert len(results) == 2
        assert results[1].status == TestStatus.Fail
        assert results[1].node.config.severity == "ERROR"


class TestSkipMacros:
    @pytest.fixture(scope="class")
    def models(self) -> Dict[str, str]:
        return {"model_one.sql": model_one_sql, "eschema.yml": empty_schema_yml}

    def test_skip_macros(self, project: Any) -> None:
        results = run_dbt()
        write_file(ref_override_sql, project.project_root, "macros", "ref_override.sql")
        results, log_output = run_dbt_and_capture(["--partial-parse", "run"])
        assert "Starting full parse." in log_output
        write_file(ref_override2_sql, project.project_root, "macros", "ref_override.sql")
        results, log_output = run_dbt_and_capture(["--partial-parse", "run"])
        assert "Starting full parse." in log_output
        rm_file(project.project_root, "macros", "ref_override.sql")
        results, log_output = run_dbt_and_capture(["--partial-parse", "run"])
        assert "Starting full parse." in log_output
        write_file(gsm_override_sql, project.project_root, "macros", "gsm_override.sql")
        results, log_output = run_dbt_and_capture(["--partial-parse", "run"])
        assert "Starting full parse." in log_output
        write_file(gsm_override2_sql, project.project_root, "macros", "gsm_override.sql")
        results, log_output = run_dbt_and_capture(["--partial-parse", "run"])
        assert "Starting full parse." in log_output


class TestSnapshots:
    @pytest.fixture(scope="class")
    def models(self) -> Dict[str, str]:
        return {"orders.sql": orders_sql}

    def test_pp_snapshots(self, project: Any) -> None:
        results = run_dbt()
        assert len(results) == 1
        write_file(snapshot_sql, project.project_root, "snapshots", "snapshot.sql")
        results = run_dbt(["--partial-parse", "run"])
        assert len(results) == 1
        manifest = get_manifest(project.project_root)
        snapshot_id = "snapshot.test.orders_snapshot"
        assert snapshot_id in manifest.nodes
        snapshot2_id = "snapshot.test.orders2_snapshot"
        assert snapshot2_id in manifest.nodes
        results = run_dbt(["--partial-parse", "snapshot"])
        assert len(results) == 2
        write_file(snapshot2_sql, project.project_root, "snapshots", "snapshot.sql")
        results = run_dbt(["--partial-parse", "run"])
        assert len(results) == 1
        rm_file(project.project_root, "snapshots", "snapshot.sql")
        results = run_dbt(["--partial-parse", "run"])
        assert len(results) == 1


class TestTests:
    @pytest.fixture(scope="class")
    def models(self) -> Dict[str, str]:
        return {"orders.sql": orders_sql, "schema.yml": generic_schema_yml}

    @pytest.fixture(scope="class")
    def tests(self) -> Dict[str, Any]:
        return {"generic": {"readme.md": ""}}

    def test_pp_generic_tests(self, project: Any) -> None:
        results = run_dbt()
        assert len(results) == 1
        manifest = get_manifest(project.project_root)
        expected_nodes = [
            "model.test.orders",
            "test.test.unique_orders_id.1360ecc70e",
        ]
        assert expected_nodes == list(manifest.nodes.keys())
        write_file(generic_test_sql, project.project_root, "tests", "generic", "generic_test.sql")
        write_file(
            generic_test_schema_yml, project.project_root, "models", "schema.yml"
        )
        results = run_dbt(["--partial-parse", "run"])
        assert len(results) == 1
        manifest = get_manifest(project.project_root)
        test_id = "test.test.is_odd_orders_id.82834fdc5b"
        assert test_id in manifest.nodes
        expected_nodes = [
            "model.test.orders",
            "test.test.unique_orders_id.1360ecc70e",
            "test.test.is_odd_orders_id.82834fdc5b",
        ]
        assert expected_nodes == list(manifest.nodes.keys())
        write_file(
            generic_test_edited_sql, project.project_root, "tests", "generic", "generic_test.sql"
        )
        results = run_dbt(["--partial-parse", "run"])
        assert len(results) == 1
        manifest = get_manifest(project.project_root)
        test_id = "test.test.is_odd_orders_id.82834fdc5b"
        assert test_id in manifest.nodes
        expected_nodes = [
            "model.test.orders",
            "test.test.unique_orders_id.1360ecc70e",
            "test.test.is_odd_orders_id.82834fdc5b",
        ]
        assert expected_nodes == list(manifest.nodes.keys())


class TestExternalModels:
    @pytest.fixture(scope="class")
    def external_model_node(self) -> ModelNodeArgs:
        return ModelNodeArgs(
            name="external_model",
            package_name="external",
            identifier="test_identifier",
            schema="test_schema",
        )

    @pytest.fixture(scope="class")
    def external_model_node_versioned(self) -> ModelNodeArgs:
        return ModelNodeArgs(
            name="external_model_versioned",
            package_name="external",
            identifier="test_identifier_v1",
            schema="test_schema",
            version=1,
        )

    @pytest.fixture(scope="class")
    def external_model_node_depends_on(self) -> ModelNodeArgs:
        return ModelNodeArgs(
            name="external_model_depends_on",
            package_name="external",
            identifier="test_identifier_depends_on",
            schema="test_schema",
            depends_on_nodes=["model.external.external_model_depends_on_parent"],
        )

    @pytest.fixture(scope="class")
    def external_model_node_depends_on_parent(self) -> ModelNodeArgs:
        return ModelNodeArgs(
            name="external_model_depends_on_parent",
            package_name="external",
            identifier="test_identifier_depends_on_parent",
            schema="test_schema",
        )

    @pytest.fixture(scope="class")
    def external_model_node_merge(self) -> ModelNodeArgs:
        return ModelNodeArgs(
            name="model_two",
            package_name="test",
            identifier="test_identifier",
            schema="test_schema",
        )

    @pytest.fixture(scope="class")
    def models(self) -> Dict[str, str]:
        return {"model_one.sql": model_one_sql}

    @mock.patch("dbt.plugins.get_plugin_manager")
    def test_pp_external_models(
        self,
        get_plugin_manager: mock.MagicMock,
        project: Any,
        external_model_node: ModelNodeArgs,
        external_model_node_versioned: ModelNodeArgs,
        external_model_node_depends_on: ModelNodeArgs,
        external_model_node_depends_on_parent: ModelNodeArgs,
        external_model_node_merge: ModelNodeArgs,
    ) -> None:
        external_nodes = PluginNodes()
        external_nodes.add_model(external_model_node)
        get_plugin_manager.return_value.get_nodes.return_value = external_nodes
        manifest = run_dbt(["parse"])
        assert len(manifest.nodes) == 2
        assert set(manifest.nodes.keys()) == {
            "model.external.external_model",
            "model.test.model_one",
        }
        assert len(manifest.external_node_unique_ids) == 1
        assert manifest.external_node_unique_ids == ["model.external.external_model"]
        write_file(model_two_sql, project.project_root, "models", "model_two.sql")
        manifest = run_dbt(["--partial-parse", "parse"])
        assert len(manifest.nodes) == 3
        external_nodes.add_model(external_model_node_merge)
        manifest = run_dbt(["--partial-parse", "parse"])
        assert len(manifest.nodes) == 3
        assert len(manifest.external_node_unique_ids) == 1
        write_file(model_two_disabled_sql, project.project_root, "models", "model_two.sql")
        manifest = run_dbt(["--partial-parse", "parse"])
        assert len(manifest.nodes) == 2
        assert len(manifest.disabled) == 1
        assert len(manifest.external_node_unique_ids) == 1
        write_file(model_two_sql, project.project_root, "models", "model_two.sql")
        external_nodes.add_model(external_model_node_versioned)
        manifest = run_dbt(["--partial-parse", "parse"])
        assert len(manifest.nodes) == 4
        assert len(manifest.external_node_unique_ids) == 2
        write_file(
            "SELECT * FROM {{ref('external', 'external_model')}}",
            project.project_root,
            "models",
            "model_depends_on_external.sql",
        )
        manifest = run_dbt(["--partial-parse", "parse"])
        assert len(manifest.nodes) == 5
        assert len(manifest.external_node_unique_ids) == 2
        rm_file(project.project_root, "models", "model_depends_on_external.sql")
        manifest = run_dbt(["--partial-parse", "parse"])
        assert len(manifest.nodes) == 4
        external_nodes.add_model(external_model_node_depends_on)
        external_nodes.add_model(external_model_node_depends_on_parent)
        manifest = run_dbt(["--partial-parse", "parse"])
        assert len(manifest.nodes) == 6
        assert len(manifest.external_node_unique_ids) == 4
        run_dbt(["--partial-parse", "parse"])
        assert len(manifest.nodes) == 6
        assert len(manifest.external_node_unique_ids) == 4


class TestPortablePartialParsing:
    @pytest.fixture(scope="class")
    def models(self) -> Dict[str, str]:
        return {"model_one.sql": model_one_sql}

    @pytest.fixture(scope="class")
    def packages(self) -> Dict[str, Any]:
        return {"packages": [{"local": "local_dependency"}]}

    @pytest.fixture(scope="class")
    def local_dependency_files(self) -> Dict[str, Any]:
        return {
            "dbt_project.yml": local_dependency__dbt_project_yml,
            "models": {
                "schema.yml": local_dependency__models__schema_yml,
                "model_to_import.sql": local_dependency__models__model_to_import_sql,
            },
            "macros": {"dep_macro.sql": local_dependency__macros__dep_macro_sql},
            "seeds": {"seed.csv": local_dependency__seeds__seed_csv},
        }

    def rename_project_root(
        self, project: Any, new_project_root: str
    ) -> None:
        with up_one(new_project_root):
            rename_dir(project.project_root, new_project_root)
            project.project_root = new_project_root
            flags.set_from_args(Namespace(PROJECT_DIR=new_project_root), None)

    @pytest.fixture(scope="class", autouse=True)
    def initial_run_and_rename_project_dir(
        self, project: Any, local_dependency_files: Dict[str, Any]
    ) -> None:
        initial_project_root = project.project_root
        renamed_project_root = os.path.join(project.project_root.dirname, "renamed_project_dir")
        write_project_files(project.project_root, "local_dependency", local_dependency_files)
        run_dbt(["deps"])
        assert len(run_dbt(["seed"])) == 1
        assert len(run_dbt(["run"])) == 2
        self.rename_project_root(project, renamed_project_root)
        yield
        self.rename_project_root(project, initial_project_root)

    def test_pp_renamed_project_dir_unchanged_project_contents(self, project: Any) -> None:
        run_dbt(["deps"])
        assert len(run_dbt(["--partial-parse", "seed"])) == 1
        assert len(run_dbt(["--partial-parse", "run"])) == 2

    def test_pp_renamed_project_dir_changed_project_contents(self, project: Any) -> None:
        write_file(model_two_sql, project.project_root, "models", "model_two.sql")
        run_dbt(["deps"])
        assert len(run_dbt(["--partial-parse", "seed"])) == 1
        assert len(run_dbt(["--partial-parse", "run"])) == 3


class TestProfileChanges:
    @pytest.fixture(scope="class")
    def models(self) -> Dict[str, str]:
        return {"model.sql": "select 1 as id"}

    def test_profile_change(
        self, project: Any, dbt_profile_data: Dict[str, Any]
    ) -> None:
        _, stdout = run_dbt_and_capture(["parse"])
        assert "Unable to do partial parsing because saved manifest not found" in stdout
        _, stdout = run_dbt_and_capture(["parse"])
        assert "Unable to do partial parsing" not in stdout
        dbt_profile_data["test"]["outputs"]["default"]["dbname"] = "dbt2"
        write_file(
            yaml.safe_dump(dbt_profile_data),
            project.profiles_dir,
            "profiles.yml",
        )
        _, stdout = run_dbt_and_capture(["parse"])
        assert "Unable to do partial parsing because profile has changed" in stdout
        dbt_profile_data["test"]["outputs"]["default"]["pass"] = "another_password"
        write_file(
            yaml.safe_dump(dbt_profile_data),
            project.profiles_dir,
            "profiles.yml",
        )
        _, stdout = run_dbt_and_capture(["parse"])
        assert "Unable to do partial parsing" not in stdout
