from collections import namedtuple
from typing import Any, Dict, Optional

import pytest

from dbt.tests.util import check_relations_equal, run_dbt

models__delete_insert_incremental_predicates_sql: str = """
{{ config(
    materialized = 'incremental',
    unique_key = 'id'
) }}

{% if not is_incremental() %}

select 1 as id, 'hello' as msg, 'blue' as color
union all
select 2 as id, 'goodbye' as msg, 'red' as color

{% else %}

-- delete will not happen on the above record where id = 2, so new record will be inserted instead
select 1 as id, 'hey' as msg, 'blue' as color
union all
select 2 as id, 'yo' as msg, 'green' as color
union all
select 3 as id, 'anyway' as msg, 'purple' as color

{% endif %}
"""

seeds__expected_delete_insert_incremental_predicates_csv: str = """id,msg,color
1,hey,blue
2,goodbye,red
2,yo,green
3,anyway,purple
"""

ResultHolder = namedtuple(
    "ResultHolder",
    [
        "seed_count",
        "model_count",
        "seed_rows",
        "inc_test_model_count",
        "opt_model_count",
        "relation",
    ],
)


class BaseIncrementalPredicates:
    @pytest.fixture(scope="class")
    def models(self) -> Dict[str, str]:
        return {
            "delete_insert_incremental_predicates.sql": models__delete_insert_incremental_predicates_sql
        }

    @pytest.fixture(scope="class")
    def seeds(self) -> Dict[str, str]:
        return {
            "expected_delete_insert_incremental_predicates.csv": seeds__expected_delete_insert_incremental_predicates_csv
        }

    @pytest.fixture(scope="class")
    def project_config_update(self) -> Dict[str, Any]:
        return {
            "models": {
                "+incremental_predicates": ["id != 2"],
                "+incremental_strategy": "delete+insert",
            }
        }

    def update_incremental_model(self, incremental_model: str) -> int:
        """update incremental model after the seed table has been updated"""
        model_result_set = run_dbt(["run", "--select", incremental_model])
        return len(model_result_set)

    def get_test_fields(
        self,
        project: Any,
        seed: str,
        incremental_model: str,
        update_sql_file: Optional[str],
        opt_model_count: Optional[int] = None,
    ) -> ResultHolder:
        seed_count: int = len(run_dbt(["seed", "--select", seed, "--full-refresh"]))

        model_count: int = len(run_dbt(["run", "--select", incremental_model, "--full-refresh"]))
        relation: str = incremental_model
        row_count_query: str = "select * from {}.{}".format(project.test_schema, seed)
        seed_rows: int = len(project.run_sql(row_count_query, fetch="all"))
        inc_test_model_count: int = self.update_incremental_model(incremental_model=incremental_model)

        return ResultHolder(
            seed_count, model_count, seed_rows, inc_test_model_count, opt_model_count, relation
        )

    def check_scenario_correctness(
        self, expected_fields: ResultHolder, test_case_fields: ResultHolder, project: Any
    ) -> None:
        """Invoke assertions to verify correct build functionality"""
        assert expected_fields.seed_count == test_case_fields.seed_count
        assert expected_fields.model_count == test_case_fields.model_count
        assert expected_fields.seed_rows == test_case_fields.seed_rows
        assert expected_fields.inc_test_model_count == test_case_fields.inc_test_model_count
        if expected_fields.opt_model_count and test_case_fields.opt_model_count:
            assert expected_fields.opt_model_count == test_case_fields.opt_model_count
        check_relations_equal(
            project.adapter, [expected_fields.relation, test_case_fields.relation]
        )

    def get_expected_fields(self, relation: str, seed_rows: int, opt_model_count: Optional[int] = None) -> ResultHolder:
        return ResultHolder(
            seed_count=1,
            model_count=1,
            inc_test_model_count=1,
            seed_rows=seed_rows,
            opt_model_count=opt_model_count,
            relation=relation,
        )

    def test__incremental_predicates(self, project: Any) -> None:
        """seed should match model after two incremental runs"""
        expected_fields: ResultHolder = self.get_expected_fields(
            relation="expected_delete_insert_incremental_predicates", seed_rows=4
        )
        test_case_fields: ResultHolder = self.get_test_fields(
            project,
            seed="expected_delete_insert_incremental_predicates",
            incremental_model="delete_insert_incremental_predicates",
            update_sql_file=None,
        )
        self.check_scenario_correctness(expected_fields, test_case_fields, project)


class TestIncrementalPredicatesDeleteInsert(BaseIncrementalPredicates):
    pass


class TestPredicatesDeleteInsert(BaseIncrementalPredicates):
    @pytest.fixture(scope="class")
    def project_config_update(self) -> Dict[str, Any]:
        return {"models": {"+predicates": ["id != 2"], "+incremental_strategy": "delete+insert"}}