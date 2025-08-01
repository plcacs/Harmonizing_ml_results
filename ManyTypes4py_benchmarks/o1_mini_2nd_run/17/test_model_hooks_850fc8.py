from pathlib import Path
import pytest
from dbt.exceptions import ParsingError
from dbt.tests.util import run_dbt, write_file
from dbt_common.exceptions import CompilationError
from tests.functional.adapter.hooks.fixtures import (
    macros__before_and_after,
    models__hooked,
    models__hooks,
    models__hooks_configured,
    models__hooks_error,
    models__hooks_kwargs,
    models__post,
    models__pre,
    properties__model_hooks,
    properties__model_hooks_list,
    properties__seed_models,
    properties__test_snapshot_models,
    seeds__example_seed_csv,
    snapshots__test_snapshot,
)
from typing import Any, Dict, List, Optional
MODEL_PRE_HOOK: str = """
   insert into {{this.schema}}.on_model_hook (
        test_state,
        target_dbname,
        target_host,
        target_name,
        target_schema,
        target_type,
        target_user,
        target_pass,
        target_threads,
        run_started_at,
        invocation_id,
        thread_id
   ) VALUES (
    'start',
    '{{ target.dbname }}',
    '{{ target.host }}',
    '{{ target.name }}',
    '{{ target.schema }}',
    '{{ target.type }}',
    '{{ target.user }}',
    '{{ target.get("pass", "") }}',
    {{ target.threads }},
    '{{ run_started_at }}',
    '{{ invocation_id }}',
    '{{ thread_id }}'
   )
"""
MODEL_POST_HOOK: str = """
   insert into {{this.schema}}.on_model_hook (
        test_state,
        target_dbname,
        target_host,
        target_name,
        target_schema,
        target_type,
        target_user,
        target_pass,
        target_threads,
        run_started_at,
        invocation_id,
        thread_id
   ) VALUES (
    'end',
    '{{ target.dbname }}',
    '{{ target.host }}',
    '{{ target.name }}',
    '{{ target.schema }}',
    '{{ target.type }}',
    '{{ target.user }}',
    '{{ target.get("pass", "") }}',
    {{ target.threads }},
    '{{ run_started_at }}',
    '{{ invocation_id }}',
    '{{ thread_id }}'
   )
"""


class BaseTestPrePost:
    @pytest.fixture(scope="class", autouse=True)
    def setUp(self, project: Any) -> None:
        project.run_sql_file(project.test_data_dir / Path("seed_model.sql"))

    def get_ctx_vars(
        self, state: str, count: int, project: Any
    ) -> List[Dict[str, Any]]:
        fields: List[str] = [
            "test_state",
            "target_dbname",
            "target_host",
            "target_name",
            "target_schema",
            "target_threads",
            "target_type",
            "target_user",
            "target_pass",
            "run_started_at",
            "invocation_id",
            "thread_id",
        ]
        field_list: str = ', '.join(['"{}"'.format(f) for f in fields])
        query: str = f"select {field_list} from {project.test_schema}.on_model_hook where test_state = '{state}'"
        vals: List[Any] = project.run_sql(query, fetch="all")
        assert len(vals) != 0, "nothing inserted into hooks table"
        assert len(vals) >= count, "too few rows in hooks table"
        assert len(vals) <= count, "too many rows in hooks table"
        return [{k: v for k, v in zip(fields, val)} for val in vals]

    def check_hooks(
        self, state: str, project: Any, host: Optional[str], count: int = 1
    ) -> None:
        ctxs: List[Dict[str, Any]] = self.get_ctx_vars(state, count=count, project=project)
        for ctx in ctxs:
            assert ctx["test_state"] == state
            assert ctx["target_dbname"] == "dbt"
            assert ctx["target_host"] == host
            assert ctx["target_name"] == "default"
            assert ctx["target_schema"] == project.test_schema
            assert ctx["target_threads"] == 4
            assert ctx["target_type"] == "postgres"
            assert ctx["target_user"] == "root"
            assert ctx["target_pass"] == ""
            assert (
                ctx["run_started_at"] is not None
                and len(ctx["run_started_at"]) > 0
            ), "run_started_at was not set"
            assert (
                ctx["invocation_id"] is not None
                and len(ctx["invocation_id"]) > 0
            ), "invocation_id was not set"
            assert ctx["thread_id"].startswith("Thread-")


class TestPrePostModelHooks(BaseTestPrePost):
    @pytest.fixture(scope="class")
    def project_config_update(self) -> Dict[str, Any]:
        return {
            "models": {
                "test": {
                    "pre-hook": [
                        MODEL_PRE_HOOK,
                        {
                            "sql": "vacuum {{ this.schema }}.on_model_hook",
                            "transaction": False,
                        },
                    ],
                    "post-hook": [
                        {
                            "sql": "vacuum {{ this.schema }}.on_model_hook",
                            "transaction": False,
                        },
                        MODEL_POST_HOOK,
                    ],
                }
            }
        }

    @pytest.fixture(scope="class")
    def models(self) -> Dict[str, str]:
        return {"hooks.sql": models__hooks}

    def test_pre_and_post_run_hooks(
        self, project: Any, dbt_profile_target: Dict[str, Any]
    ) -> None:
        run_dbt()
        self.check_hooks("start", project, dbt_profile_target.get("host", None))
        self.check_hooks("end", project, dbt_profile_target.get("host", None))


class TestPrePostModelHooksUnderscores(TestPrePostModelHooks):
    @pytest.fixture(scope="class")
    def project_config_update(self) -> Dict[str, Any]:
        return {
            "models": {
                "test": {
                    "pre-hook": [
                        MODEL_PRE_HOOK,
                        {
                            "sql": "vacuum {{ this.schema }}.on_model_hook",
                            "transaction": False,
                        },
                    ],
                    "post-hook": [
                        {
                            "sql": "vacuum {{ this.schema }}.on_model_hook",
                            "transaction": False,
                        },
                        MODEL_POST_HOOK,
                    ],
                }
            }
        }


class TestHookRefs(BaseTestPrePost):
    @pytest.fixture(scope="class")
    def project_config_update(self) -> Dict[str, Any]:
        return {
            "models": {
                "test": {
                    "hooked": {
                        "post-hook": [
                            """
                        insert into {{this.schema}}.on_model_hook select
                        test_state,
                        '{{ target.dbname }}' as target_dbname,
                        '{{ target.host }}' as target_host,
                        '{{ target.name }}' as target_name,
                        '{{ target.schema }}' as target_schema,
                        '{{ target.type }}' as target_type,
                        '{{ target.user }}' as target_user,
                        '{{ target.get(pass, "") }}' as target_pass,
                        {{ target.threads }} as target_threads,
                        '{{ run_started_at }}' as run_started_at,
                        '{{ invocation_id }}' as invocation_id,
                        '{{ thread_id }}' as thread_id
                        from {{ ref('post') }}
                        """.strip()
                        ]
                    }
                }
            }
        }

    @pytest.fixture(scope="class")
    def models(self) -> Dict[str, str]:
        return {
            "hooked.sql": models__hooked,
            "post.sql": models__post,
            "pre.sql": models__pre,
        }

    def test_pre_post_model_hooks_refed(
        self, project: Any, dbt_profile_target: Dict[str, Any]
    ) -> None:
        run_dbt()
        self.check_hooks("start", project, dbt_profile_target.get("host", None))
        self.check_hooks("end", project, dbt_profile_target.get("host", None))


class TestPrePostModelHooksOnSeeds:
    @pytest.fixture(scope="class")
    def seeds(self) -> Dict[str, str]:
        return {"example_seed.csv": seeds__example_seed_csv}

    @pytest.fixture(scope="class")
    def models(self) -> Dict[str, str]:
        return {"schema.yml": properties__seed_models}

    @pytest.fixture(scope="class")
    def project_config_update(self) -> Dict[str, Any]:
        return {
            "seed-paths": ["seeds"],
            "models": {},
            "seeds": {
                "post-hook": [
                    "alter table {{ this }} add column new_col int",
                    "update {{ this }} set new_col = 1",
                    "select null::{{ dbt.type_int() }} as id",
                ],
                "quote_columns": False,
            },
        }

    def test_hooks_on_seeds(self, project: Any) -> None:
        res: List[Any] = run_dbt(["seed"])
        assert len(res) == 1, "Expected exactly one item"
        res = run_dbt(["test"])
        assert len(res) == 1, "Expected exactly one item"


class TestPrePostModelHooksWithMacros(BaseTestPrePost):
    @pytest.fixture(scope="class")
    def macros(self) -> Dict[str, str]:
        return {"before-and-after.sql": macros__before_and_after}

    @pytest.fixture(scope="class")
    def models(self) -> Dict[str, str]:
        return {
            "schema.yml": properties__model_hooks,
            "hooks.sql": models__hooks,
        }

    def test_pre_and_post_run_hooks(
        self, project: Any, dbt_profile_target: Dict[str, Any]
    ) -> None:
        run_dbt()
        self.check_hooks("start", project, dbt_profile_target.get("host", None))
        self.check_hooks("end", project, dbt_profile_target.get("host", None))


class TestPrePostModelHooksListWithMacros(TestPrePostModelHooksWithMacros):
    @pytest.fixture(scope="class")
    def models(self) -> Dict[str, str]:
        return {
            "schema.yml": properties__model_hooks_list,
            "hooks.sql": models__hooks,
        }


class TestHooksRefsOnSeeds:
    """
    This should not succeed, and raise an explicit error
    https://github.com/dbt-labs/dbt-core/issues/6806
    """

    @pytest.fixture(scope="class")
    def seeds(self) -> Dict[str, str]:
        return {"example_seed.csv": seeds__example_seed_csv}

    @pytest.fixture(scope="class")
    def models(self) -> Dict[str, str]:
        return {"schema.yml": properties__seed_models, "post.sql": models__post}

    @pytest.fixture(scope="class")
    def project_config_update(self) -> Dict[str, Any]:
        return {"seeds": {"post-hook": ["select * from {{ ref('post') }}"]}}

    def test_hook_with_ref_on_seeds(self, project: Any) -> None:
        with pytest.raises(ParsingError) as excinfo:
            run_dbt(["parse"])
        assert "Seeds cannot depend on other nodes" in str(excinfo.value)


class TestPrePostModelHooksOnSeedsPlusPrefixed(TestPrePostModelHooksOnSeeds):
    @pytest.fixture(scope="class")
    def project_config_update(self) -> Dict[str, Any]:
        return {
            "seed-paths": ["seeds"],
            "models": {},
            "seeds": {
                "+post-hook": [
                    "alter table {{ this }} add column new_col int",
                    "update {{ this }} set new_col = 1",
                ],
                "quote_columns": False,
            },
        }


class TestPrePostModelHooksOnSeedsPlusPrefixedWhitespace(TestPrePostModelHooksOnSeeds):
    @pytest.fixture(scope="class")
    def project_config_update(self) -> Dict[str, Any]:
        return {
            "seed-paths": ["seeds"],
            "models": {},
            "seeds": {
                "+post-hook": [
                    "alter table {{ this }} add column new_col int",
                    "update {{ this }} set new_col = 1",
                ],
                "quote_columns": False,
            },
        }


class TestPrePostModelHooksOnSnapshots:
    @pytest.fixture(scope="class", autouse=True)
    def setUp(self, project: Any) -> None:
        path: Path = Path(project.project_root) / "test-snapshots"
        Path.mkdir(path, exist_ok=True)
        write_file(snapshots__test_snapshot, path, "snapshot.sql")

    @pytest.fixture(scope="class")
    def models(self) -> Dict[str, str]:
        return {"schema.yml": properties__test_snapshot_models}

    @pytest.fixture(scope="class")
    def seeds(self) -> Dict[str, str]:
        return {"example_seed.csv": seeds__example_seed_csv}

    @pytest.fixture(scope="class")
    def project_config_update(self) -> Dict[str, Any]:
        return {
            "seed-paths": ["seeds"],
            "snapshot-paths": ["test-snapshots"],
            "models": {},
            "snapshots": {
                "post-hook": [
                    "alter table {{ this }} add column new_col int",
                    "update {{ this }} set new_col = 1",
                ]
            },
            "seeds": {"quote_columns": False},
        }

    def test_hooks_on_snapshots(self, project: Any) -> None:
        res: List[Any] = run_dbt(["seed"])
        assert len(res) == 1, "Expected exactly one item"
        res = run_dbt(["snapshot"])
        assert len(res) == 1, "Expected exactly one item"
        res = run_dbt(["test"])
        assert len(res) == 1, "Expected exactly one item"


class PrePostModelHooksInConfigSetup(BaseTestPrePost):
    @pytest.fixture(scope="class")
    def project_config_update(self) -> Dict[str, Any]:
        return {"macro-paths": ["macros"]}

    @pytest.fixture(scope="class")
    def models(self) -> Dict[str, str]:
        return {"hooks.sql": models__hooks_configured}


class TestPrePostModelHooksInConfig(PrePostModelHooksInConfigSetup):
    def test_pre_and_post_model_hooks_model(
        self, project: Any, dbt_profile_target: Dict[str, Any]
    ) -> None:
        run_dbt()
        self.check_hooks("start", project, dbt_profile_target.get("host", None))
        self.check_hooks("end", project, dbt_profile_target.get("host", None))


class TestPrePostModelHooksInConfigWithCount(PrePostModelHooksInConfigSetup):
    @pytest.fixture(scope="class")
    def project_config_update(self) -> Dict[str, Any]:
        return {
            "models": {
                "test": {
                    "pre-hook": [
                        MODEL_PRE_HOOK,
                        {
                            "sql": "vacuum {{ this.schema }}.on_model_hook",
                            "transaction": False,
                        },
                    ],
                    "post-hook": [
                        {
                            "sql": "vacuum {{ this.schema }}.on_model_hook",
                            "transaction": False,
                        },
                        MODEL_POST_HOOK,
                    ],
                }
            }
        }

    def test_pre_and_post_model_hooks_model_and_project(
        self, project: Any, dbt_profile_target: Dict[str, Any]
    ) -> None:
        run_dbt()
        self.check_hooks(
            "start", project, dbt_profile_target.get("host", None), count=2
        )
        self.check_hooks("end", project, dbt_profile_target.get("host", None), count=2)


class TestPrePostModelHooksInConfigKwargs(TestPrePostModelHooksInConfig):
    @pytest.fixture(scope="class")
    def models(self) -> Dict[str, str]:
        return {"hooks.sql": models__hooks_kwargs}


class TestPrePostSnapshotHooksInConfigKwargs(TestPrePostModelHooksOnSnapshots):
    @pytest.fixture(scope="class", autouse=True)
    def setUp(self, project: Any) -> None:
        path: Path = Path(project.project_root) / "test-kwargs-snapshots"
        Path.mkdir(path, exist_ok=True)
        write_file(snapshots__test_snapshot, path, "snapshot.sql")

    @pytest.fixture(scope="class")
    def project_config_update(self) -> Dict[str, Any]:
        return {
            "seed-paths": ["seeds"],
            "snapshot-paths": ["test-kwargs-snapshots"],
            "models": {},
            "snapshots": {
                "post-hook": [
                    "alter table {{ this }} add column new_col int",
                    "update {{ this }} set new_col = 1",
                ]
            },
            "seeds": {"quote_columns": False},
        }


class TestDuplicateHooksInConfigs:
    @pytest.fixture(scope="class")
    def models(self) -> Dict[str, str]:
        return {"hooks.sql": models__hooks_error}

    def test_run_duplicate_hook_defs(self, project: Any) -> None:
        with pytest.raises(CompilationError) as exc:
            run_dbt()
        assert "pre_hook" in str(exc.value)
        assert "pre-hook" in str(exc.value)
