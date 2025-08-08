from typing import Dict, List
import pytest
from click.testing import CliRunner
from _pytest.fixtures import FixtureRequest

def test_list_registered_pipelines(fake_project_cli, fake_metadata, yaml_dump_mock, pipelines_dict: Dict[str, List[str]]):
    ...

class TestRegistryDescribeCommand:

    def test_describe_registered_pipeline(self, fake_project_cli, fake_metadata, yaml_dump_mock, pipeline_name: str, pipelines_dict: Dict[str, List[str]]):
        ...

    def test_registered_pipeline_not_found(self, fake_project_cli, fake_metadata):
        ...

    def test_describe_registered_pipeline_default(self, fake_project_cli, fake_metadata, yaml_dump_mock, pipelines_dict: Dict[str, List[str]]):
        ...
