from collections import namedtuple
from pathlib import Path
import pytest
from dbt.artifacts.schemas.results import RunStatus
from dbt.tests.util import check_relations_equal, run_dbt

models__trinary_unique_key_list_sql: str = "\n-- a multi-argument unique key list should see overwriting on rows in the model\n--   where all unique key fields apply\n\n{{\n    config(\n        materialized='incremental',\n        unique_key=['state', 'county', 'city']\n    )\n}}\n\nselect\n    state as state,\n    county as county,\n    city as city,\n    last_visit_date as last_visit_date\nfrom {{ ref('seed') }}\n\n{% if is_incremental() %}\n    where last_visit_date > (select max(last_visit_date) from {{ this }})\n{% endif %}\n\n"
models__nontyped_trinary_unique_key_list_sql: str = "\n-- a multi-argument unique key list should see overwriting on rows in the model\n--   where all unique key fields apply\n--   N.B. needed for direct comparison with seed\n\n{{\n    config(\n        materialized='incremental',\n        unique_key=['state', 'county', 'city']\n    )\n}}\n\nselect\n    state as state,\n    county as county,\n    city as city,\n    last_visit_date as last_visit_date\nfrom {{ ref('seed') }}\n\n{% if is_incremental() %}\n    where last_visit_date > (select max(last_visit_date) from {{ this }})\n{% endif %}\n\n"
models__unary_unique_key_list_sql: str = "\n-- a one argument unique key list should result in overwritting semantics for\n--   that one matching field\n\n{{\n    config(\n        materialized='incremental',\n        unique_key=['state']\n    )\n}}\n\nselect\n    state as state,\n    county as county,\n    city as city,\n    last_visit_date as last_visit_date\nfrom {{ ref('seed') }}\n\n{% if is_incremental() %}\n    where last_visit_date > (select max(last_visit_date) from {{ this }})\n{% endif %}\n\n"
models__not_found_unique_key_sql: str = "\n-- a model with a unique key not found in the table itself will error out\n\n{{\n    config(\n        materialized='incremental',\n        unique_key='thisisnotacolumn'\n    )\n}}\n\nselect\n    *\nfrom {{ ref('seed') }}\n\n{% if is_incremental() %}\n    where last_visit_date > (select max(last_visit_date) from {{ this }})\n{% endif %}\n\n"
models__empty_unique_key_list_sql: str = "\n-- model with empty list unique key should build normally\n\n{{\n    config(\n        materialized='incremental',\n        unique_key=[]\n    )\n}}\n\nselect * from {{ ref('seed') }}\n\n{% if is_incremental() %}\n    where last_visit_date > (select max(last_visit_date) from {{ this }})\n{% endif %}\n\n"
models__no_unique_key_sql: str = "\n-- no specified unique key should cause no special build behavior\n\n{{\n    config(\n        materialized='incremental'\n    )\n}}\n\nselect\n    *\nfrom {{ ref('seed') }}\n\n{% if is_incremental() %}\n    where last_visit_date > (select max(last_visit_date) from {{ this }})\n{% endif %}\n\n"
models__empty_str_unique_key_sql: str = "\n-- ensure model with empty string unique key should build normally\n\n{{\n    config(\n        materialized='incremental',\n        unique_key=''\n    )\n}}\n\nselect\n    *\nfrom {{ ref('seed') }}\n\n{% if is_incremental() %}\n    where last_visit_date > (select max(last_visit_date) from {{ this }})\n{% endif %}\n\n"
models__str_unique_key_sql: str = "\n-- a unique key with a string should trigger to overwrite behavior when\n--   the source has entries in conflict (i.e. more than one row per unique key\n--   combination)\n\n{{\n    config(\n        materialized='incremental',\n        unique_key='state'\n    )\n}}\n\nselect\n    state as state,\n    county as county,\n    city as city,\n    last_visit_date as last_visit_date\nfrom {{ ref('seed') }}\n\n{% if is_incremental() %}\n    where last_visit_date > (select max(last_visit_date) from {{ this }})\n{% endif %}\n\n"
models__duplicated_unary_unique_key_list_sql: str = "\n{{\n    config(\n        materialized='incremental',\n        unique_key=['state', 'state']\n    )\n}}\n\nselect\n    state as state,\n    county as county,\n    city as city,\n    last_visit_date as last_visit_date\nfrom {{ ref('seed') }}\n\n{% if is_incremental() %}\n    where last_visit_date > (select max(last_visit_date) from {{ this }})\n{% endif %}\n\n"
models__not_found_unique_key_list_sql: str = "\n-- a unique key list with any element not in the model itself should error out\n\n{{\n    config(\n        materialized='incremental',\n        unique_key=['state', 'thisisnotacolumn']\n    )\n}}\n\nselect * from {{ ref('seed') }}\n\n"
models__expected__one_str__overwrite_sql: str = "\n{{\n    config(\n        materialized='table'\n    )\n}}\n\nselect\n    'CT' as state,\n    'Hartford' as county,\n    'Hartford' as city,\n    cast('2022-02-14' as date) as last_visit_date\nunion all\nselect 'MA','Suffolk','Boston','2020-02-12'\nunion all\nselect 'NJ','Mercer','Trenton','2022-01-01'\nunion all\nselect 'NY','Kings','Brooklyn','2021-04-02'\nunion all\nselect 'NY','New York','Manhattan','2021-04-01'\nunion all\nselect 'PA','Philadelphia','Philadelphia','2021-05-21'\n\n"
models__expected__unique_key_list__inplace_overwrite_sql: str = "\n{{\n    config(\n        materialized='table'\n    )\n}}\n\nselect\n    'CT' as state,\n    'Hartford' as county,\n    'Hartford' as city,\n    cast('2022-02-14' as date) as last_visit_date\nunion all\nselect 'MA','Suffolk','Boston','2020-02-12'\nunion all\nselect 'NJ','Mercer','Trenton','2022-01-01'\nunion all\nselect 'NY','Kings','Brooklyn','2021-04-02'\nunion all\nselect 'NY','New York','Manhattan','2021-04-01'\nunion all\nselect 'PA','Philadelphia','Philadelphia','2021-05-21'\n\n"
seeds__duplicate_insert_sql: str = "\n-- Insert statement which when applied to seed.csv triggers the inplace\n--   overwrite strategy of incremental models. Seed and incremental model\n--   diverge.\n\n-- insert new row, which should not be in incremental model\n--  with primary or first three columns unique\ninsert into {schema}.seed\n    (state, county, city, last_visit_date)\nvalues ('CT','Hartford','Hartford','2022-02-14');\n\n"
seeds__seed_csv: str = 'state,county,city,last_visit_date\nCT,Hartford,Hartford,2020-09-23\nMA,Suffolk,Boston,2020-02-12\nNJ,Mercer,Trenton,2022-01-01\nNY,Kings,Brooklyn,2021-04-02\nNY,New York,Manhattan,2021-04-01\nPA,Philadelphia,Philadelphia,2021-05-21\n'
seeds__add_new_rows_sql: str = "\n-- Insert statement which when applied to seed.csv sees incremental model\n--   grow in size while not (necessarily) diverging from the seed itself.\n\n-- insert two new rows, both of which should be in incremental model\n--   with any unique columns\ninsert into {schema}.seed\n    (state, county, city, last_visit_date)\nvalues ('WA','King','Seattle','2022-02-01');\n\ninsert into {schema}.seed\n    (state, county, city, last_visit_date)\nvalues ('CA','Los Angeles','Los Angeles','2022-02-01');\n\n"
ResultHolder = namedtuple('ResultHolder', ['seed_count', 'model_count', 'seed_rows', 'inc_test_model_count', 'opt_model_count', 'relation'])

class BaseIncrementalUniqueKey:

    @pytest.fixture(scope='class')
    def models(self) -> dict:
        return {'trinary_unique_key_list.sql': models__trinary_unique_key_list_sql, 'nontyped_trinary_unique_key_list.sql': models__nontyped_trinary_unique_key_list_sql, 'unary_unique_key_list.sql': models__unary_unique_key_list_sql, 'not_found_unique_key.sql': models__not_found_unique_key_sql, 'empty_unique_key_list.sql': models__empty_unique_key_list_sql, 'no_unique_key.sql': models__no_unique_key_sql, 'empty_str_unique_key.sql': models__empty_str_unique_key_sql, 'str_unique_key.sql': models__str_unique_key_sql, 'duplicated_unary_unique_key_list.sql': models__duplicated_unary_unique_key_list_sql, 'not_found_unique_key_list.sql': models__not_found_unique_key_list_sql, 'expected': {'one_str__overwrite.sql': models__expected__one_str__overwrite_sql, 'unique_key_list__inplace_overwrite.sql': models__expected__unique_key_list__inplace_overwrite_sql}}

    @pytest.fixture(scope='class')
    def seeds(self) -> dict:
        return {'duplicate_insert.sql': seeds__duplicate_insert_sql, 'seed.csv': seeds__seed_csv, 'add_new_rows.sql': seeds__add_new_rows_sql}

    @pytest.fixture(autouse=True)
    def clean_up(self, project) -> None:
        yield
        with project.adapter.connection_named('__test'):
            relation = project.adapter.Relation.create(database=project.database, schema=project.test_schema)
            project.adapter.drop_schema(relation)
    pass

    def update_incremental_model(self, incremental_model: str) -> int:
        """update incremental model after the seed table has been updated"""
        model_result_set = run_dbt(['run', '--select', incremental_model])
        return len(model_result_set)

    def get_test_fields(self, project, seed: str, incremental_model: str, update_sql_file: str, opt_model_count: int = None) -> ResultHolder:
        """build a test case and return values for assertions
        [INFO] Models must be in place to test incremental model
        construction and merge behavior. Database touches are side
        effects to extract counts (which speak to health of unique keys)."""
        seed_count = len(run_dbt(['seed', '--select', seed, '--full-refresh']))
        model_count = len(run_dbt(['run', '--select', incremental_model, '--full-refresh']))
        relation = incremental_model
        row_count_query = 'select * from {}.{}'.format(project.test_schema, seed)
        project.run_sql_file(Path('seeds') / Path(update_sql_file + '.sql'))
        seed_rows = len(project.run_sql(row_count_query, fetch='all'))
        inc_test_model_count = self.update_incremental_model(incremental_model=incremental_model)
        return ResultHolder(seed_count, model_count, seed_rows, inc_test_model_count, opt_model_count, relation)

    def check_scenario_correctness(self, expected_fields: ResultHolder, test_case_fields: ResultHolder, project) -> None:
        """Invoke assertions to verify correct build functionality"""
        assert expected_fields.seed_count == test_case_fields.seed_count
        assert expected_fields.model_count == test_case_fields.model_count
        assert expected_fields.seed_rows == test_case_fields.seed_rows
        assert expected_fields.inc_test_model_count == test_case_fields.inc_test_model_count
        if expected_fields.opt_model_count and test_case_fields.opt_model_count:
            assert expected_fields.opt_model_count == test_case_fields.opt_model_count
        check_relations_equal(project.adapter, [expected_fields.relation, test_case_fields.relation])

    def get_expected_fields(self, relation: str, seed_rows: int, opt_model_count: int = None) -> ResultHolder:
        return ResultHolder(seed_count=1, model_count=1, seed_rows=seed_rows, inc_test_model_count=1, opt_model_count=opt_model_count, relation=relation)

    def fail_to_build_inc_missing_unique_key_column(self, incremental_model_name: str) -> tuple:
        """should pass back error state when trying build an incremental
        model whose unique key or keylist includes a column missing
        from the incremental model"""
        seed_count = len(run_dbt(['seed', '--select', 'seed', '--full-refresh']))
        run_dbt(['run', '--select', incremental_model_name, '--full-refresh'], expect_pass=True)
        run_result = run_dbt(['run', '--select', incremental_model_name], expect_pass=False).results[0]
        return (run_result.status, run_result.message)

    def test__no_unique_keys(self, project) -> None:
        """with no unique keys, seed and model should match"""
        expected_fields = self.get_expected_fields(relation='seed', seed_rows=8)
        test_case_fields = self.get_test_fields(project, seed='seed', incremental_model='no_unique_key', update_sql_file='add_new_rows')
        self.check_scenario_correctness(expected_fields, test_case_fields, project)

    def test__empty_str_unique_key(self, project) -> None:
        """with empty string for unique key, seed and model should match"""
        expected_fields = self.get_expected_fields(relation='seed', seed_rows=8)
        test_case_fields = self.get_test_fields(project, seed='seed', incremental_model='empty_str_unique_key', update_sql_file='add_new_rows')
        self.check_scenario_correctness(expected_fields, test_case_fields, project)

    def test__one_unique_key(self, project) -> None:
        """with one unique key, model will overwrite existing row"""
        expected_fields = self.get_expected_fields(relation='one_str__overwrite', seed_rows=7, opt_model_count=1)
        test_case_fields = self.get_test_fields(project, seed='seed', incremental_model='str_unique_key', update_sql_file='duplicate_insert', opt_model_count=self.update_incremental_model('one_str__overwrite'))
        self.check_scenario_correctness(expected_fields, test_case_fields, project)

    def test__bad_unique_key(self, project) -> None:
        """expect compilation error from unique key not being a column"""
        status, exc = self.fail_to_build_inc_missing_unique_key_column(incremental_model_name='not_found_unique_key')
        assert status == RunStatus.Error
        assert 'thisisnotacolumn' in exc.lower()

    def test__empty_unique_key_list(self, project) -> None:
        """with no unique keys, seed and model should match"""
        expected_fields = self.get_expected_fields(relation='seed', seed_rows=8)
        test_case_fields = self.get_test_fields(project, seed='seed', incremental_model='empty_unique_key_list', update_sql_file='add_new_rows')
        self.check_scenario_correctness(expected_fields, test_case_fields, project)

    def test__unary_unique_key_list(self, project) -> None:
        """with one unique key, model will overwrite existing row"""
        expected_fields = self.get_expected_fields(relation='unique_key_list__inplace_overwrite', seed_rows=7, opt_model_count=1)
        test_case_fields = self.get_test_fields(project, seed='seed', incremental_model='unary_unique_key_list', update_sql_file='duplicate_insert', opt_model_count=self.update_incremental_model('unique_key_list__inplace_overwrite'))
        self.check_scenario_correctness(expected_fields, test_case_fields, project)

    def test__duplicated_unary_unique_key_list(self, project) -> None:
        """with two of the same unique key, model will overwrite existing row"""
        expected_fields = self.get_expected_fields(relation='unique_key_list__inplace_overwrite', seed_rows=7, opt_model_count=1)
        test_case_fields = self.get_test_fields(project, seed='seed', incremental_model='duplicated_unary_unique_key_list', update_sql_file='duplicate_insert', opt_model_count=self.update_incremental_model('unique_key_list__inplace_overwrite'))
        self.check_scenario_correctness(expected_fields, test_case_fields, project)

    def test__trinary_unique_key_list(self, project) -> None:
        """with three unique keys, model will overwrite existing row"""
        expected_fields = self.get_expected_fields(relation='unique_key_list__inplace_overwrite', seed_rows=7, opt_model_count=1)
        test_case_fields = self.get_test_fields(project, seed='seed', incremental_model='trinary_unique_key_list', update_sql_file='duplicate_insert', opt_model_count=self.update_incremental_model('unique_key_list__inplace_overwrite'))
        self.check_scenario_correctness(expected_fields, test_case_fields, project)

    def test__trinary_unique_key_list_no_update(self, project) -> None:
        """even with three unique keys, adding distinct rows to seed does not
        cause seed and model to diverge"""
        expected_fields = self.get_expected_fields(relation='seed', seed_rows=8)
        test_case_fields = self.get_test_fields(project, seed='seed', incremental_model='nontyped_trinary_unique_key_list', update_sql_file='add_new_rows')
        self.check_scenario_correctness(expected_fields, test_case_fields, project)

    def test__bad_unique_key_list(self, project) -> None:
        """expect compilation error from unique key not being a column"""
        status, exc = self.fail_to_build_inc_missing_unique_key_column(incremental_model_name='not_found_unique_key_list')
        assert status == RunStatus.Error
        assert 'thisisnotacolumn' in exc.lower()

class TestIncrementalUniqueKey(BaseIncrementalUniqueKey):
    pass
