from typing import List

class TestRunCommands:

    def drop_snapshots(self, happy_path_project, project_root) -> None:
    
    def test_run_commmand(self, happy_path_project, dbt_command: List[str]) -> None:

class TestSelectResourceType:

    def catcher(self) -> EventCatcher:
    
    def runner(self, catcher: EventCatcher) -> dbtRunner:
    
    def test_select_by_resource_type(self, resource_type: str, happy_path_project, runner: dbtRunner, catcher: EventCatcher) -> None:
