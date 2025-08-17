import os
import re
from argparse import Namespace
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union
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
        return {
            "model_one.sql": model_one_sql,
        }

    def test_pp_models(self, project: Any) -> None:
        # initial run
        # run_dbt(['clean'])
        results = run_dbt(["run"])
        assert len(results) == 1

        # add a model file
        write_file(model_two_sql, project.project_root, "models", "model_two.sql")
        results = run_dbt(["--partial-parse", "run"])
        assert len(results) == 2

        # add a schema file
        write_file(models_schema1_yml, project.project_root, "models", "schema.yml")
        results = run_dbt(["--partial-parse", "run"])
        assert len(results) == 2
        manifest = get_manifest(project.project_root)
        assert "model.test.model_one" in manifest.nodes
        model_one_node = manifest.nodes["model.test.model_one"]
        assert model_one_node.description == "The first model"
        assert model_one_node.patch_path == "test://" + normalize("models/schema.yml")

        # add a model and a schema file (with a test) at the same time
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

        # modify model sql file, ensure description still there
        write_file(model_three_modified_sql, project.project_root, "models", "model_three.sql")
        results = run_dbt(["--partial-parse", "run"])
        manifest = get_manifest(project.project_root)
        model_id = "model.test.model_three"
        assert model_id in manifest.nodes
        model_three_node = manifest.nodes[model_id]
        assert model_three_node.description == "The third model"

        # Change the model 3 test from unique to not_null
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

        # go back to previous version of schema file, removing patch, test, and model for model three
        write_file(models_schema1_yml, project.project_root, "models", "schema.yml")
        rm_file(project.project_root, "models", "model_three.sql")
        results = run_dbt(["--partial-parse", "run"])
        assert len(results) == 2

        # remove schema file, still have 3 models
        write_file(model_three_sql, project.project_root, "models", "model_three.sql")
        rm_file(project.project_root, "models", "schema.yml")
        results = run_dbt(["--partial-parse", "run"])
        assert len(results) == 3
        manifest = get_manifest(project.project_root)
        schema_file_id = "test://" + normalize("models/schema.yml")
        assert schema_file_id not in manifest.files
        project_files = [f for f in manifest.files if f.startswith("test://")]
        assert len(project_files) == 3

        # Put schema file back and remove a model
        # referred to in schema file
        write_file(models_schema2_yml, project.project_root, "models", "schema.yml")
        rm_file(project.project_root, "models", "model_three.sql")
        with pytest.raises(CompilationError):
            results = run_dbt(["--partial-parse", "--warn-error", "run"])

        # Put model back again
        write_file(model_three_sql, project.project_root, "models", "model_three.sql")
        results = run_dbt(["--partial-parse", "run"])
        assert len(results) == 3

        # Add model four refing model three
        write_file(model_four1_sql, project.project_root, "models", "model_four.sql")
        results = run_dbt(["--partial-parse", "run"])
        assert len(results) == 4

        # Remove model_three and change model_four to ref model_one
        # and change schema file to remove model_three
        rm_file(project.project_root, "models", "model_three.sql")
        write_file(model_four2_sql, project.project_root, "models", "model_four.sql")
        write_file(models_schema1_yml, project.project_root, "models", "schema.yml")
        results = run_dbt(["--partial-parse", "run"])
        assert len(results) == 3

        # Remove model four, put back model three, put back schema file
        write_file(model_three_sql, project.project_root, "models", "model_three.sql")
        write_file(models_schema2_yml, project.project_root, "models", "schema.yml")
        rm_file(project.project_root, "models", "model_four.sql")
        results = run_dbt(["--partial-parse", "run"])
        assert len(results) == 3

        # disable model three in the schema file
        write_file(models_schema4_yml, project.project_root, "models", "schema.yml")
        results = run_dbt(["--partial-parse", "run"])
        assert len(results) == 2

        # update enabled config to be true for model three in the schema file
        write_file(models_schema4b_yml, project.project_root, "models", "schema.yml")
        results = run_dbt(["--partial-parse", "run"])
        assert len(results) == 3

        # disable model three in the schema file again
        write_file(models_schema4_yml, project.project_root, "models", "schema.yml")
        results = run_dbt(["--partial-parse", "run"])
        assert len(results) == 2

        # remove disabled config for model three in the schema file to check it gets enabled
        write_file(models_schema4b_yml, project.project_root, "models", "schema.yml")
        results = run_dbt(["--partial-parse", "run"])
        assert len(results) == 3

        # Add a macro
        write_file(my_macro_sql, project.project_root, "macros", "my_macro.sql")
        results = run_dbt(["--partial-parse", "run"])
        assert len(results) == 3
        manifest = get_manifest(project.project_root)
        macro_id = "macro.test.do_something"
        assert macro_id in manifest.macros

        # Modify the macro
        write_file(my_macro2_sql, project.project_root, "macros", "my_macro.sql")
        results = run_dbt(["--partial-parse", "run"])
        assert len(results) == 3

        # Add a macro patch
        write_file(models_schema3_yml, project.project_root, "models", "schema.yml")
        results = run_dbt(["--partial-parse", "run"])
        assert len(results) == 3

        # Remove the macro
        rm_file(project.project_root, "macros", "my_macro.sql")
        with pytest.raises(CompilationError):
            results = run_dbt(["--partial-parse", "--warn-error", "run"])

        # put back macro file, got back to schema file with no macro
        # add separate macro patch schema file
        write_file(models_schema2_yml, project.project_root, "models", "schema.yml")
        write_file(my_macro_sql, project.project_root, "macros", "my_macro.sql")
        write_file(macros_yml, project.project_root, "macros", "macros.yml")
        results = run_dbt(["--partial-parse", "run"])

        # delete macro and schema file
        rm_file(project.project_root, "macros", "my_macro.sql")
        rm_file(project.project_root, "macros", "macros.yml")
        results = run_dbt(["--partial-parse", "run"])
        assert len(results) == 3

        # Add an empty schema file
        write_file(empty_schema_yml, project.project_root, "models", "eschema.yml")
        results = run_dbt(["--partial-parse", "run"])
        assert len(results) == 3

        # Add version to empty schema file
        write_file(empty_schema_with_version_yml, project.project_root, "models", "eschema.yml")
        results = run_dbt(["--partial-parse", "run"])
        assert len(results) == 3

        # Disable model_three
        write_file(model_three_disabled_sql, project.project_root, "models", "model_three.sql")
        results = run_dbt(["--partial-parse", "run"])
        assert len(results) == 2
        manifest = get_manifest(project.project_root)
        model_id = "model.test.model_three"
        assert model_id in manifest.disabled
        assert model_id not in manifest.nodes

        # Edit disabled model three
        write_file(model_three_disabled2_sql, project.project_root, "models", "model_three.sql")
        results = run_dbt(["--partial-parse", "run"])
        assert len(results) == 2
        manifest = get_manifest(project.project_root)
        model_id = "model.test.model_three"
        assert model_id in manifest.disabled
        assert model_id not in manifest.nodes

        # Remove disabled from model three
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
        return {
            "model_one.sql": model_one_sql,
        }

    def test_pp_sources(self, project: Any) -> None:
        # initial run
        write_file(raw_customers_csv, project.project_root, "seeds", "raw_customers.csv")
        write_file(sources_tests1_sql, project.project_root, "macros", "tests.sql")
        results = run_dbt(["run"])
        assert len(results) == 1

        # Partial parse running 'seed'
        run_dbt(["--partial-parse", "seed"])
        manifest = get_manifest(project.project_root)
        seed_file_id = "test://" + normalize("seeds/raw_customers.csv")
        assert seed_file_id in manifest.files

        # Add another seed file
        write_file(raw_customers_csv, project.project_root, "seeds", "more_customers.csv")
        run_dbt(["--partial-parse", "run"])
        seed_file_id = "test://" + normalize("seeds/more_customers.csv")
        manifest = get_manifest(project.project_root)
        assert seed_file_id in manifest.files
        seed_id = "seed.test.more_customers"
        assert seed_id in manifest.nodes

        # Remove seed file and add a schema files with a source referring to raw_customers
        rm_file(project.project_root, "seeds", "more_customers.csv")
        write_file(schema_sources1_yml, project.project_root, "models", "sources.yml")
        results = run_dbt(["--partial-parse", "run"])
        manifest = get_manifest(project.project_root)
        assert len(manifest.sources) == 1
        file_id = "test://" + normalize("models/sources.yml")
        assert file_id in manifest.files

        # add a model referring to raw_customers source
        write_file(customers_sql, project.project_root, "models", "customers.sql")
        results = run_dbt(["--partial-parse", "run"])
        assert len(results) == 2

        # remove sources schema file
        rm_file(project.project_root, "models", "sources.yml")
        with pytest.raises(CompilationError):
            results = run_dbt(["--partial-parse", "run"])

        # put back sources and add an exposures file
        write_file(schema_sources2_yml, project.project_root, "models", "sources.yml")
        results = run_dbt(["--partial-parse", "run"])

        # remove seed referenced in exposures file
        rm_file(project.project_root, "seeds", "raw_customers.csv")
        with pytest.raises(CompilationError):
            results = run_dbt(["--partial-parse", "run"])

        # put back seed and remove depends_on from exposure
        write_file(raw_customers_csv, project.project_root, "seeds", "raw_customers.csv")
        write_file(schema_sources3_yml, project.project_root, "models", "sources.yml")
        results = run_dbt(["--partial-parse", "run"])

        # Add seed config with test to schema.yml, remove exposure
        write_file(schema_sources4_yml, project.project_root, "models", "sources.yml")
        results = run_dbt(["--partial-parse", "run"])

        # Change seed name to wrong name
        write_file(schema_sources5_yml, project.project_root, "models", "sources.yml")
        with pytest.raises(CompilationError):
            results = run_dbt(["--partial-parse", "--warn-error", "run"])

        # Put back seed name to right name