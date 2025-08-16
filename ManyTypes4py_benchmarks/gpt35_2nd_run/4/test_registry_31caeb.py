from typing import Dict, List
import pytest
from click.testing import CliRunner
from _pytest.fixtures import FixtureRequest

def test_list_registered_pipelines(fake_project_cli, fake_metadata, yaml_dump_mock, pipelines_dict: Dict[str, List[str]]) -> None:
    result = CliRunner().invoke(fake_project_cli, ['registry', 'list'], obj=fake_metadata)
    assert not result.exit_code
    yaml_dump_mock.assert_called_once_with(sorted(pipelines_dict.keys()))

class TestRegistryDescribeCommand:

    def test_describe_registered_pipeline(self, fake_project_cli, fake_metadata, yaml_dump_mock, pipeline_name: str, pipelines_dict: Dict[str, List[str]]) -> None:
        result = CliRunner().invoke(fake_project_cli, ['registry', 'describe', pipeline_name], obj=fake_metadata)
        assert not result.exit_code
        expected_dict: Dict[str, List[str]] = {'Nodes': pipelines_dict[pipeline_name]}
        yaml_dump_mock.assert_called_once_with(expected_dict)

    def test_registered_pipeline_not_found(self, fake_project_cli, fake_metadata) -> None:
        result = CliRunner().invoke(fake_project_cli, ['registry', 'describe', 'missing'], obj=fake_metadata)
        assert result.exit_code
        expected_output = "Error: 'missing' pipeline not found. Existing pipelines: [__default__, data_engineering, data_processing, data_science]\n"
        assert expected_output in result.output

    def test_describe_registered_pipeline_default(self, fake_project_cli, fake_metadata, yaml_dump_mock, pipelines_dict: Dict[str, List[str]]) -> None:
        result = CliRunner().invoke(fake_project_cli, ['registry', 'describe'], obj=fake_metadata)
        assert not result.exit_code
        expected_dict: Dict[str, List[str]] = {'Nodes': pipelines_dict['__default__']}
        yaml_dump_mock.assert_called_once_with(expected_dict)
