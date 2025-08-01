import os
import pytest
import yaml
from typing import Any, Dict, List, Optional, Tuple
from dbt.artifacts.schemas.results import RunStatus
from dbt.tests.util import (
    check_table_does_exist,
    mkdir,
    rm_dir,
    rm_file,
    run_dbt,
    run_dbt_and_capture,
    write_file,
)
from dbt_common.exceptions import UndefinedMacroError
from tests.functional.run_operations.fixtures import happy_macros_sql, model_sql, sad_macros_sql


class TestOperations:
    @pytest.fixture(scope="class")
    def models(self) -> Dict[str, str]:
        return {"model.sql": model_sql}

    @pytest.fixture(scope="class")
    def macros(self) -> Dict[str, str]:
        return {"happy_macros.sql": happy_macros_sql, "sad_macros.sql": sad_macros_sql}

    @pytest.fixture(scope="class")
    def dbt_profile_data(self, unique_schema: str) -> Dict[str, Any]:
        return {
            "test": {
                "outputs": {
                    "default": {
                        "type": "postgres",
                        "threads": 4,
                        "host": "localhost",
                        "port": int(os.getenv("POSTGRES_TEST_PORT", 5432)),
                        "user": os.getenv("POSTGRES_TEST_USER", "root"),
                        "pass": os.getenv("POSTGRES_TEST_PASS", "password"),
                        "dbname": os.getenv("POSTGRES_TEST_DATABASE", "dbt"),
                        "schema": unique_schema,
                    },
                    "noaccess": {
                        "type": "postgres",
                        "threads": 4,
                        "host": "localhost",
                        "port": int(os.getenv("POSTGRES_TEST_PORT", 5432)),
                        "user": "noaccess",
                        "pass": "password",
                        "dbname": os.getenv("POSTGRES_TEST_DATABASE", "dbt"),
                        "schema": unique_schema,
                    },
                },
                "target": "default",
            }
        }

    def run_operation(
        self, macro: str, expect_pass: bool = True, extra_args: Optional[List[str]] = None, **kwargs: Any
    ) -> Any:
        args: List[str] = ["run-operation", macro]
        if kwargs:
            args.extend(("--args", yaml.safe_dump(kwargs)))
        if extra_args:
            args.extend(extra_args)
        return run_dbt(args, expect_pass=expect_pass)

    def test_macro_noargs(self, project: Any) -> None:
        self.run_operation("no_args")
        check_table_does_exist(project.adapter, "no_args")

    def test_macro_args(self, project: Any) -> None:
        self.run_operation("table_name_args", table_name="my_fancy_table")
        check_table_does_exist(project.adapter, "my_fancy_table")

    def test_macro_exception(self, project: Any) -> None:
        self.run_operation("syntax_error", expect_pass=False)

    def test_macro_missing(self, project: Any) -> None:
        with pytest.raises(
            UndefinedMacroError,
            match="dbt could not find a macro with the name 'this_macro_does_not_exist' in any package",
        ):
            self.run_operation("this_macro_does_not_exist", expect_pass=False)

    def test_cannot_connect(self, project: Any) -> None:
        self.run_operation("no_args", extra_args=["--target", "noaccess"], expect_pass=False)

    def test_vacuum(self, project: Any) -> None:
        run_dbt(["run"])
        self.run_operation("vacuum", table_name="model")

    def test_vacuum_ref(self, project: Any) -> None:
        run_dbt(["run"])
        self.run_operation("vacuum_ref", ref_target="model")

    def test_select(self, project: Any) -> None:
        self.run_operation("select_something", name="world")

    def test_access_graph(self, project: Any) -> None:
        self.run_operation("log_graph")

    def test_print(self, project: Any) -> None:
        self.run_operation("print_something")

    def test_run_operation_local_macro(self, project: Any) -> None:
        pkg_macro: str = (
            "\n{% macro something_cool() %}\n    {{ log('something cool', info=true) }}\n{% endmacro %}\n        "
        )
        mkdir("pkg/macros")
        write_file(pkg_macro, "pkg/macros/something_cool.sql")
        pkg_yaml: str = "\npackages:\n    - local: pkg\n        "
        write_file(pkg_yaml, "packages.yml")
        pkg_dbt_project: str = "\nname: 'pkg'\n        "
        write_file(pkg_dbt_project, "pkg/dbt_project.yml")
        run_dbt(["deps"])
        results_and_log: Tuple[List[Any], str] = run_dbt_and_capture(["run-operation", "something_cool"])
        results, log_output = results_and_log
        for result in results:
            if result.status == RunStatus.Skipped:
                continue
            timing_keys: List[str] = [timing.name for timing in result.timing]
            assert timing_keys == ["compile", "execute"]
        assert "something cool" in log_output

        results_and_log = run_dbt_and_capture(["run-operation", "pkg.something_cool"])
        results, log_output = results_and_log
        for result in results:
            if result.status == RunStatus.Skipped:
                continue
            timing_keys = [timing.name for timing in result.timing]
            assert timing_keys == ["compile", "execute"]
        assert "something cool" in log_output

        rm_dir("pkg")
        rm_file("packages.yml")