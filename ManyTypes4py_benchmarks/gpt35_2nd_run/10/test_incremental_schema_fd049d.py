from typing import Dict, Any

class TestIncrementalSchemaChange:

    def properties(self) -> Dict[str, Any]:
    
    def models(self) -> Dict[str, Any]:
    
    def tests(self) -> Dict[str, Any]:
    
    def run_twice_and_assert(self, include: str, compare_source: str, compare_target: str, project: Any) -> None:
    
    def run_incremental_append_new_columns(self, project: Any) -> None:
    
    def run_incremental_append_new_columns_remove_one(self, project: Any) -> None:
    
    def run_incremental_sync_all_columns(self, project: Any) -> None:
    
    def run_incremental_sync_remove_only(self, project: Any) -> None:
    
    def test_run_incremental_ignore(self, project: Any) -> None:
    
    def test_run_incremental_append_new_columns(self, project: Any) -> None:
    
    def test_run_incremental_sync_all_columns(self, project: Any) -> None:
    
    def test_run_incremental_fail_on_schema_change(self, project: Any) -> None:
