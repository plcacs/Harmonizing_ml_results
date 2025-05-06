from collections import namedtuple
import pytest
from dbt.tests.util import check_relations_equal, run_dbt
from typing import Dict, Optional, Any

models__merge_exclude_columns_sql: str = """
{{ config(
    materialized = 'incremental',
    unique_key = 'id',
    incremental_strategy='merge',
    merge_exclude_columns=['msg']
) }}

{% if not is_incremental() %}

-- data for first invocation of model

select 1 as id, 'hello' as msg, 'blue' as color
union all
select 2 as id, 'goodbye' as msg, 'red' as color

{% else %}

-- data for subsequent incremental update

select 1 as id, 'hey' as msg, 'blue' as color
union all
select 2 as id, 'yo' as msg, 'green' as color
union all
select 3 as id, 'anyway' as msg, 'purple' as color

{% endif %}
"""

seeds__expected_merge_exclude_columns_csv: str = 'id,msg,color\n1,hello,blue\n2,goodbye,green\n3,anyway,purple\n'

ResultHolder = namedtuple(
    'ResultHolder',
    ['seed_count', 'model_count', 'seed_rows', 'inc_test_model_count', 'relation']
)

class BaseMergeExcludeColumns:

    @pytest.fixture(scope='class')
    def models(self) -> Dict[str, str]:
        return {'merge_exclude_columns.sql': models__merge_exclude_columns_sql}

    @pytest.fixture(scope='class')
    def seeds(self) -> Dict[str, str]:
        return {'expected_merge_exclude_columns.csv': seeds__expected_merge_exclude_columns_csv}

    def update_incremental_model(self, incremental_model: str) -> int:
        """update incremental model after the seed table has been updated"""
        model_result_set = run_dbt(['run', '--select', incremental_model])
        return len(model_result_set)

    def get_test_fields(
        self,
        project: Any,
        seed: str,
        incremental_model: str,
        update_sql_file: Optional[str]
    ) -> ResultHolder:
        seed_count = len(run_dbt(['seed', '--select', seed, '--full-refresh']))
        model_count = len(run_dbt(['run', '--select', incremental_model, '--full-refresh']))
        relation = incremental_model
        row_count_query = f'select * from {project.test_schema}.{seed}'
        seed_rows = len(project.run_sql(row_count_query, fetch='all'))
        inc_test_model_count = self.update_incremental_model(incremental_model=incremental_model)
        return ResultHolder(seed_count, model_count, seed_rows, inc_test_model_count, relation)

    def check_scenario_correctness(
        self,
        expected_fields: ResultHolder,
        test_case_fields: ResultHolder,
        project: Any
    ) -> None:
        """Invoke assertions to verify correct build functionality"""
        assert expected_fields.seed_count == test_case_fields.seed_count
        assert expected_fields.model_count == test_case_fields.model_count
        assert expected_fields.seed_rows == test_case_fields.seed_rows
        assert expected_fields.inc_test_model_count == test_case_fields.inc_test_model_count
        check_relations_equal(project.adapter, [expected_fields.relation, test_case_fields.relation])

    def test__merge_exclude_columns(self, project: Any) -> None:
        """seed should match model after two incremental runs"""
        expected_fields = ResultHolder(
            seed_count=1,
            model_count=1,
            seed_rows=3,
            inc_test_model_count=1,
            relation='expected_merge_exclude_columns'
        )
        test_case_fields = self.get_test_fields(
            project,
            seed='expected_merge_exclude_columns',
            incremental_model='merge_exclude_columns',
            update_sql_file=None
        )
        self.check_scenario_correctness(expected_fields, test_case_fields, project)

class TestMergeExcludeColumns(BaseMergeExcludeColumns):
    pass
