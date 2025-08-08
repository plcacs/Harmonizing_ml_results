from collections import namedtuple
from typing import Dict, Optional
import pytest

models__delete_insert_incremental_predicates_sql: str
seeds__expected_delete_insert_incremental_predicates_csv: str
ResultHolder = namedtuple('ResultHolder', ['seed_count', 'model_count', 'seed_rows', 'inc_test_model_count', 'opt_model_count', 'relation'])

class BaseIncrementalPredicates:

    def models(self) -> Dict[str, str]:
        ...

    def seeds(self) -> Dict[str, str]:
        ...

    def project_config_update(self) -> Dict[str, str]:
        ...

    def update_incremental_model(self, incremental_model: str) -> int:
        ...

    def get_test_fields(self, project, seed, incremental_model, update_sql_file, opt_model_count=None) -> ResultHolder:
        ...

    def check_scenario_correctness(self, expected_fields: ResultHolder, test_case_fields: ResultHolder, project) -> None:
        ...

    def get_expected_fields(self, relation: str, seed_rows: int, opt_model_count: Optional[int] = None) -> ResultHolder:
        ...

    def test__incremental_predicates(self, project) -> None:
        ...

class TestIncrementalPredicatesDeleteInsert(BaseIncrementalPredicates):
    pass

class TestPredicatesDeleteInsert(BaseIncrementalPredicates):

    def project_config_update(self) -> Dict[str, str]:
        ...
