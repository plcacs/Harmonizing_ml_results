from collections import namedtuple
from pathlib import Path
import pytest
from dbt.artifacts.schemas.results import RunStatus
from dbt.tests.util import check_relations_equal, run_dbt

# ...

class ResultHolder(namedtuple('ResultHolder', ['seed_count', 'model_count', 'seed_rows', 'inc_test_model_count', 'opt_model_count', 'relation'])):
    pass

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
