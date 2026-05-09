import os
from pathlib import Path
from unittest import mock
from unittest.mock import Mock, call
import click
import pytest
import yaml
from dbt.exceptions import DbtRuntimeError
from dbt.tests.util import run_dbt

class TestInitProjectWithExistingProfilesYml:
    def test_init_task_in_project_with_existing_profiles_yml(self, 
        mock_get_adapter: mock.MagicMock, 
        mock_confirm: mock.MagicMock, 
        mock_prompt: mock.MagicMock, 
        project: object) -> None:
        # ... rest of the method

class TestInitProjectWithoutExistingProfilesYml:
    def test_init_task_in_project_without_existing_profiles_yml(self, 
        exists: mock.MagicMock, 
        mock_prompt: mock.MagicMock, 
        mock_confirm: mock.MagicMock, 
        mock_get_adapter: mock.MagicMock, 
        project: object) -> None:
        # ... rest of the method

class TestInitInvalidProfileTemplate:
    def test_init_task_in_project_with_invalid_profile_template(self, 
        mock_prompt: mock.MagicMock, 
        mock_confirm: mock.MagicMock, 
        mock_get_adapter: mock.MagicMock, 
        project: object) -> None:
        # ... rest of the method

class TestInitOutsideOfProjectBase:
    def test_init_task_outside_of_project(self, 
        mock_prompt: mock.MagicMock, 
        mock_confirm: mock.MagicMock, 
        mock_get_adapter: mock.MagicMock, 
        project: object, 
        project_name: str) -> None:
        # ... rest of the method

class TestInitInvalidProjectNameCLI:
    def test_init_invalid_project_name_cli(self, 
        mock_prompt: mock.MagicMock, 
        mock_confirm: mock.MagicMock, 
        mock_get_adapter: mock.MagicMock, 
        project_name: str, 
        project: object) -> None:
        # ... rest of the method

class TestInitInvalidProjectNamePrompt:
    def test_init_invalid_project_name_prompt(self, 
        mock_prompt: mock.MagicMock, 
        mock_confirm: mock.MagicMock, 
        mock_get_adapter: mock.MagicMock, 
        project_name: str, 
        project: object) -> None:
        # ... rest of the method

class TestInitProvidedProjectNameAndSkipProfileSetup:
    def test_init_provided_project_name_and_skip_profile_setup(self, 
        mock_prompt: mock.MagicMock, 
        mock_confirm: mock.MagicMock, 
        mock_get_adapter: mock.MagicMock, 
        project: object, 
        project_name: str) -> None:
        # ... rest of the method

class TestInitInsideProjectAndSkipProfileSetup:
    def test_init_inside_project_and_skip_profile_setup(self, 
        mock_prompt: mock.MagicMock, 
        mock_confirm: mock.MagicMock, 
        mock_get_adapter: mock.MagicMock, 
        project: object, 
        project_name: str) -> None:
        # ... rest of the method

class TestInitOutsideOfProjectWithSpecifiedProfile:
    def test_init_task_outside_of_project_with_specified_profile(self, 
        mock_prompt: mock.MagicMock, 
        mock_get_adapter: mock.MagicMock, 
        project: object, 
        project_name: str, 
        unique_schema: str, 
        dbt_profile_data: dict) -> None:
        # ... rest of the method

class TestInitOutsideOfProjectSpecifyingInvalidProfile:
    def test_init_task_outside_project_specifying_invalid_profile_errors(self, 
        mock_prompt: mock.MagicMock, 
        mock_get_adapter: mock.MagicMock, 
        project: object, 
        project_name: str) -> None:
        # ... rest of the method

class TestInitOutsideOfProjectSpecifyingProfileNoProfilesYml:
    def test_init_task_outside_project_specifying_profile_no_profiles_yml_errors(self, 
        mock_prompt: mock.MagicMock, 
        mock_get_adapter: mock.MagicMock, 
        project: object, 
        project_name: str) -> None:
        # ... rest of the method
