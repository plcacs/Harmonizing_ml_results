#!/usr/bin/env python3
import os
from pathlib import Path
from unittest import mock
from unittest.mock import Mock, call
from typing import Any
import click
import pytest
import yaml
from dbt.exceptions import DbtRuntimeError
from dbt.tests.util import run_dbt


class TestInitProjectWithExistingProfilesYml:
    @mock.patch('dbt.task.init._get_adapter_plugin_names')
    @mock.patch('click.confirm')
    @mock.patch('click.prompt')
    def test_init_task_in_project_with_existing_profiles_yml(
        self,
        mock_prompt: Any,
        mock_confirm: Any,
        mock_get_adapter: Any,
        project: Any
    ) -> None:
        manager: Any = Mock()
        manager.attach_mock(mock_prompt, 'prompt')
        manager.attach_mock(mock_confirm, 'confirm')
        manager.confirm.side_effect = ['y']
        manager.prompt.side_effect = [
            1,
            'localhost',
            5432,
            'test_user',
            'test_password',
            'test_db',
            'test_schema',
            4,
        ]
        mock_get_adapter.return_value = [project.adapter.type()]
        run_dbt(['init'])
        manager.assert_has_calls([
            call.confirm(f'The profile test already exists in {os.path.join(project.profiles_dir, "profiles.yml")}. Continue and overwrite it?'),
            call.prompt("Which database would you like to use?\n[1] postgres\n\n(Don't see the one you want? https://docs.getdbt.com/docs/available-adapters)\n\nEnter a number", type=click.INT),
            call.prompt('host (hostname for the instance)', default=None, hide_input=False, type=None),
            call.prompt('port', default=5432, hide_input=False, type=click.INT),
            call.prompt('user (dev username)', default=None, hide_input=False, type=None),
            call.prompt('pass (dev password)', default=None, hide_input=True, type=None),
            call.prompt('dbname (default database that dbt will build objects in)', default=None, hide_input=False, type=None),
            call.prompt('schema (default schema that dbt will build objects in)', default=None, hide_input=False, type=None),
            call.prompt('threads (1 or more)', default=1, hide_input=False, type=click.INT)
        ])
        with open(os.path.join(project.profiles_dir, 'profiles.yml'), 'r') as f:
            content: str = f.read()
            assert content == (
                'test:\n  outputs:\n    dev:\n      dbname: test_db\n'
                '      host: localhost\n      pass: test_password\n'
                '      port: 5432\n      schema: test_schema\n'
                '      threads: 4\n      type: postgres\n      user: test_user\n'
                '  target: dev\n'
            )

    def test_init_task_in_project_specifying_profile_errors(self, project: Any) -> None:
        with pytest.raises(DbtRuntimeError) as error:
            run_dbt(['init', '--profile', 'test'], expect_pass=False)
            assert 'Can not init existing project with specified profile' in str(error.value)


class TestInitProjectWithoutExistingProfilesYml:
    @mock.patch('dbt.task.init._get_adapter_plugin_names')
    @mock.patch('click.prompt')
    @mock.patch.object(Path, 'exists', autospec=True)
    def test_init_task_in_project_without_existing_profiles_yml(
        self,
        exists: Any,
        mock_prompt: Any,
        mock_get_adapter: Any,
        project: Any
    ) -> None:
        def exists_side_effect(path: Path) -> bool:
            return {'profiles.yml': False}.get(path.name, os.path.exists(path))
        exists.side_effect = exists_side_effect
        manager: Any = Mock()
        manager.attach_mock(mock_prompt, 'prompt')
        manager.prompt.side_effect = [
            1,
            'localhost',
            5432,
            'test_user',
            'test_password',
            'test_db',
            'test_schema',
            4,
        ]
        mock_get_adapter.return_value = [project.adapter.type()]
        run_dbt(['init'])
        manager.assert_has_calls([
            call.prompt("Which database would you like to use?\n[1] postgres\n\n(Don't see the one you want? https://docs.getdbt.com/docs/available-adapters)\n\nEnter a number", type=click.INT),
            call.prompt('host (hostname for the instance)', default=None, hide_input=False, type=None),
            call.prompt('port', default=5432, hide_input=False, type=click.INT),
            call.prompt('user (dev username)', default=None, hide_input=False, type=None),
            call.prompt('pass (dev password)', default=None, hide_input=True, type=None),
            call.prompt('dbname (default database that dbt will build objects in)', default=None, hide_input=False, type=None),
            call.prompt('schema (default schema that dbt will build objects in)', default=None, hide_input=False, type=None),
            call.prompt('threads (1 or more)', default=1, hide_input=False, type=click.INT)
        ])
        with open(os.path.join(project.profiles_dir, 'profiles.yml'), 'r') as f:
            content: str = f.read()
            assert content == (
                'test:\n  outputs:\n    dev:\n      dbname: test_db\n'
                '      host: localhost\n      pass: test_password\n'
                '      port: 5432\n      schema: test_schema\n'
                '      threads: 4\n      type: postgres\n      user: test_user\n'
                '  target: dev\n'
            )

    @mock.patch.object(Path, 'exists', autospec=True)
    def test_init_task_in_project_without_profile_yml_specifying_profile_errors(
        self,
        exists: Any,
        project: Any
    ) -> None:
        def exists_side_effect(path: Path) -> bool:
            return {'profiles.yml': False}.get(path.name, os.path.exists(path))
        exists.side_effect = exists_side_effect
        with pytest.raises(DbtRuntimeError) as error:
            run_dbt(['init', '--profile', 'test'], expect_pass=False)
            assert 'Could not find profile named test' in str(error.value)


class TestInitProjectWithoutExistingProfilesYmlOrTemplate:
    @mock.patch('dbt.task.init._get_adapter_plugin_names')
    @mock.patch('click.confirm')
    @mock.patch('click.prompt')
    @mock.patch.object(Path, 'exists', autospec=True)
    def test_init_task_in_project_without_existing_profiles_yml_or_profile_template(
        self,
        exists: Any,
        mock_prompt: Any,
        mock_confirm: Any,
        mock_get_adapter: Any,
        project: Any
    ) -> None:
        def exists_side_effect(path: Path) -> bool:
            return {'profiles.yml': False, 'profile_template.yml': False}.get(path.name, os.path.exists(path))
        exists.side_effect = exists_side_effect
        manager: Any = Mock()
        manager.attach_mock(mock_prompt, 'prompt')
        manager.attach_mock(mock_confirm, 'confirm')
        manager.prompt.side_effect = [1]
        mock_get_adapter.return_value = [project.adapter.type()]
        run_dbt(['init'])
        manager.assert_has_calls([
            call.prompt("Which database would you like to use?\n[1] postgres\n\n(Don't see the one you want? https://docs.getdbt.com/docs/available-adapters)\n\nEnter a number", type=click.INT)
        ])
        with open(os.path.join(project.profiles_dir, 'profiles.yml'), 'r') as f:
            content: str = f.read()
            assert content == (
                'test:\n  outputs:\n\n    dev:\n      type: postgres\n      threads: [1 or more]\n'
                '      host: [host]\n      port: [port]\n      user: [dev_username]\n'
                '      pass: [dev_password]\n      dbname: [dbname]\n      schema: [dev_schema]\n\n'
                '    prod:\n      type: postgres\n      threads: [1 or more]\n'
                '      host: [host]\n      port: [port]\n      user: [prod_username]\n'
                '      pass: [prod_password]\n      dbname: [dbname]\n      schema: [prod_schema]\n\n'
                '  target: dev\n'
            )


class TestInitProjectWithProfileTemplateWithoutExistingProfilesYml:
    @mock.patch('dbt.task.init._get_adapter_plugin_names')
    @mock.patch('click.confirm')
    @mock.patch('click.prompt')
    @mock.patch.object(Path, 'exists', autospec=True)
    def test_init_task_in_project_with_profile_template_without_existing_profiles_yml(
        self,
        exists: Any,
        mock_prompt: Any,
        mock_confirm: Any,
        mock_get_adapter: Any,
        project: Any
    ) -> None:
        def exists_side_effect(path: Path) -> bool:
            return {'profiles.yml': False}.get(path.name, os.path.exists(path))
        exists.side_effect = exists_side_effect
        with open('profile_template.yml', 'w') as f:
            f.write(
                "fixed:\n  type: postgres\n  threads: 4\n  host: localhost\n  dbname: my_db\n"
                "  schema: my_schema\n  target: my_target\nprompts:\n  target:\n    hint: 'The target name'\n    type: string\n"
                "  port:\n    hint: 'The port (for integer test purposes)'\n    type: int\n    default: 5432\n"
                "  user:\n    hint: 'Your username'\n  pass:\n    hint: 'Your password'\n    hide_input: true"
            )
        manager: Any = Mock()
        manager.attach_mock(mock_prompt, 'prompt')
        manager.attach_mock(mock_confirm, 'confirm')
        manager.prompt.side_effect = ['my_target', 5432, 'test_username', 'test_password']
        mock_get_adapter.return_value = [project.adapter.type()]
        run_dbt(['init'])
        manager.assert_has_calls([
            call.prompt('target (The target name)', default=None, hide_input=False, type=click.STRING),
            call.prompt('port (The port (for integer test purposes))', default=5432, hide_input=False, type=click.INT),
            call.prompt('user (Your username)', default=None, hide_input=False, type=None),
            call.prompt('pass (Your password)', default=None, hide_input=True, type=None)
        ])
        with open(os.path.join(project.profiles_dir, 'profiles.yml'), 'r') as f:
            content: str = f.read()
            assert content == (
                'test:\n  outputs:\n    my_target:\n      dbname: my_db\n      host: localhost\n'
                '      pass: test_password\n      port: 5432\n      schema: my_schema\n      threads: 4\n'
                '      type: postgres\n      user: test_username\n  target: my_target\n'
            )


class TestInitInvalidProfileTemplate:
    @mock.patch('dbt.task.init._get_adapter_plugin_names')
    @mock.patch('click.confirm')
    @mock.patch('click.prompt')
    def test_init_task_in_project_with_invalid_profile_template(
        self,
        mock_prompt: Any,
        mock_confirm: Any,
        mock_get_adapter: Any,
        project: Any
    ) -> None:
        # Test that when an invalid profile_template.yml is provided in the project,
        # init command falls back to the target's profile_template.yml
        with open(os.path.join(project.project_root, 'profile_template.yml'), 'w') as f:
            f.write('invalid template')
        manager: Any = Mock()
        manager.attach_mock(mock_prompt, 'prompt')
        manager.attach_mock(mock_confirm, 'confirm')
        manager.confirm.side_effect = ['y']
        manager.prompt.side_effect = [
            1,
            'localhost',
            5432,
            'test_username',
            'test_password',
            'test_db',
            'test_schema',
            4,
        ]
        mock_get_adapter.return_value = [project.adapter.type()]
        run_dbt(['init'])
        manager.assert_has_calls([
            call.confirm(f'The profile test already exists in {os.path.join(project.profiles_dir, "profiles.yml")}. Continue and overwrite it?'),
            call.prompt("Which database would you like to use?\n[1] postgres\n\n(Don't see the one you want? https://docs.getdbt.com/docs/available-adapters)\n\nEnter a number", type=click.INT),
            call.prompt('host (hostname for the instance)', default=None, hide_input=False, type=None),
            call.prompt('port', default=5432, hide_input=False, type=click.INT),
            call.prompt('user (dev username)', default=None, hide_input=False, type=None),
            call.prompt('pass (dev password)', default=None, hide_input=True, type=None),
            call.prompt('dbname (default database that dbt will build objects in)', default=None, hide_input=False, type=None),
            call.prompt('schema (default schema that dbt will build objects in)', default=None, hide_input=False, type=None),
            call.prompt('threads (1 or more)', default=1, hide_input=False, type=click.INT)
        ])
        with open(os.path.join(project.profiles_dir, 'profiles.yml'), 'r') as f:
            content: str = f.read()
            assert content == (
                'test:\n  outputs:\n    dev:\n      dbname: test_db\n      host: localhost\n'
                '      pass: test_password\n      port: 5432\n      schema: test_schema\n'
                '      threads: 4\n      type: postgres\n      user: test_username\n'
                '  target: dev\n'
            )


class TestInitInsideOfProjectBase:
    @pytest.fixture(scope='class')
    def project_name(self, unique_schema: Any) -> str:
        return f'my_project_{unique_schema}'


class TestInitOutsideOfProjectBase:
    @pytest.fixture(scope='class')
    def project_name(self, unique_schema: Any) -> str:
        return f'my_project_{unique_schema}'

    @pytest.fixture(scope='class', autouse=True)
    def setup(self, project: Any) -> None:
        os.remove(os.path.join(project.project_root, 'dbt_project.yml'))


class TestInitOutsideOfProject(TestInitOutsideOfProjectBase):
    @pytest.fixture(scope='class')
    def dbt_profile_data(self, unique_schema: Any) -> Any:
        return {
            'test': {
                'outputs': {
                    'default2': {
                        'type': 'postgres',
                        'threads': 4,
                        'host': 'localhost',
                        'port': int(os.getenv('POSTGRES_TEST_PORT', 5432)),
                        'user': os.getenv('POSTGRES_TEST_USER', 'root'),
                        'pass': os.getenv('POSTGRES_TEST_PASS', 'password'),
                        'dbname': os.getenv('POSTGRES_TEST_DATABASE', 'dbt'),
                        'schema': unique_schema,
                    },
                    'noaccess': {
                        'type': 'postgres',
                        'threads': 4,
                        'host': 'localhost',
                        'port': int(os.getenv('POSTGRES_TEST_PORT', 5432)),
                        'user': 'noaccess',
                        'pass': 'password',
                        'dbname': os.getenv('POSTGRES_TEST_DATABASE', 'dbt'),
                        'schema': unique_schema,
                    }
                },
                'target': 'default2'
            }
        }

    @mock.patch('dbt.task.init._get_adapter_plugin_names')
    @mock.patch('click.confirm')
    @mock.patch('click.prompt')
    def test_init_task_outside_of_project(
        self,
        mock_prompt: Any,
        mock_confirm: Any,
        mock_get_adapter: Any,
        project: Any,
        project_name: str,
        unique_schema: Any
    ) -> None:
        manager: Any = Mock()
        manager.attach_mock(mock_prompt, 'prompt')
        manager.attach_mock(mock_confirm, 'confirm')
        manager.prompt.side_effect = [
            project_name,
            1,
            'localhost',
            5432,
            'test_username',
            'test_password',
            'test_db',
            'test_schema',
            4,
        ]
        mock_get_adapter.return_value = [project.adapter.type()]
        run_dbt(['init'])
        manager.assert_has_calls([
            call.prompt('Enter a name for your project (letters, digits, underscore)'),
            call.prompt("Which database would you like to use?\n[1] postgres\n\n(Don't see the one you want? https://docs.getdbt.com/docs/available-adapters)\n\nEnter a number", type=click.INT),
            call.prompt('host (hostname for the instance)', default=None, hide_input=False, type=None),
            call.prompt('port', default=5432, hide_input=False, type=click.INT),
            call.prompt('user (dev username)', default=None, hide_input=False, type=None),
            call.prompt('pass (dev password)', default=None, hide_input=True, type=None),
            call.prompt('dbname (default database that dbt will build objects in)', default=None, hide_input=False, type=None),
            call.prompt('schema (default schema that dbt will build objects in)', default=None, hide_input=False, type=None),
            call.prompt('threads (1 or more)', default=1, hide_input=False, type=click.INT)
        ])
        with open(os.path.join(project.profiles_dir, 'profiles.yml'), 'r') as f:
            content: str = f.read()
            expected_profiles: str = (
                f'{project_name}:\n  outputs:\n    dev:\n      dbname: test_db\n'
                '      host: localhost\n      pass: test_password\n'
                '      port: 5432\n      schema: test_schema\n      threads: 4\n'
                '      type: postgres\n      user: test_username\n  target: dev\n'
                'test:\n  outputs:\n    default2:\n      dbname: dbt\n'
                '      host: localhost\n      pass: password\n      port: 5432\n'
                f'      schema: {unique_schema}\n      threads: 4\n      type: postgres\n'
                "      user: root\n    noaccess:\n      dbname: dbt\n      host: localhost\n"
                "      pass: password\n      port: 5432\n"
                f"      schema: {unique_schema}\n      threads: 4\n      type: postgres\n      user: noaccess\n"
                "  target: default2\n"
            )
            assert content == expected_profiles
        with open(os.path.join(project.project_root, project_name, 'dbt_project.yml'), 'r') as f:
            content_project: str = f.read()
            expected_project: str = (
                "\n# Name your project! Project names should contain only lowercase characters\n"
                "# and underscores. A good package name should reflect your organization's\n"
                "name or the intended use of these models\n"
                f"name: '{project_name}'\nversion: '1.0.0'\n\n"
                "# This setting configures which \"profile\" dbt uses for this project.\n"
                f"profile: '{project_name}'\n\n"
                "# These configurations specify where dbt should look for different types of files.\n"
                "# The `model-paths` config, for example, states that models in this project can be\n"
                "# found in the \"models/\" directory. You probably won't need to change these!\n"
                'model-paths: ["models"]\nanalysis-paths: ["analyses"]\n'
                'test-paths: ["tests"]\nseed-paths: ["seeds"]\nmacro-paths: ["macros"]\n'
                'snapshot-paths: ["snapshots"]\n\n'
                "clean-targets:         # directories to be removed by `dbt clean`\n"
                '  - "target"\n  - "dbt_packages"\n\n\n'
                "# Configuring models\n"
                "# Full documentation: https://docs.getdbt.com/docs/configuring-models\n\n"
                f"# In this example config, we tell dbt to build all models in the example/\n"
                "# directory as views. These settings can be overridden in the individual model\n"
                "# files using the `{{ config(...) }}` macro.\n"
                f"models:\n  {project_name}:\n    # Config indicated by + and applies to all files under models/example/\n"
                "    example:\n      +materialized: view\n"
            )
            assert content_project == expected_project


class TestInitInvalidProjectNameCLI(TestInitOutsideOfProjectBase):
    @mock.patch('dbt.task.init._get_adapter_plugin_names')
    @mock.patch('click.confirm')
    @mock.patch('click.prompt')
    def test_init_invalid_project_name_cli(
        self,
        mock_prompt: Any,
        mock_confirm: Any,
        mock_get_adapter: Any,
        project_name: str,
        project: Any
    ) -> None:
        manager: Any = Mock()
        manager.attach_mock(mock_prompt, 'prompt')
        manager.attach_mock(mock_confirm, 'confirm')
        invalid_name: str = 'name-with-hyphen'
        valid_name: str = project_name
        manager.prompt.side_effect = [valid_name]
        mock_get_adapter.return_value = [project.adapter.type()]
        run_dbt(['init', invalid_name, '--skip-profile-setup'])
        manager.assert_has_calls([
            call.prompt('Enter a name for your project (letters, digits, underscore)')
        ])


class TestInitInvalidProjectNamePrompt(TestInitOutsideOfProjectBase):
    @mock.patch('dbt.task.init._get_adapter_plugin_names')
    @mock.patch('click.confirm')
    @mock.patch('click.prompt')
    def test_init_invalid_project_name_prompt(
        self,
        mock_prompt: Any,
        mock_confirm: Any,
        mock_get_adapter: Any,
        project_name: str,
        project: Any
    ) -> None:
        manager: Any = Mock()
        manager.attach_mock(mock_prompt, 'prompt')
        manager.attach_mock(mock_confirm, 'confirm')
        invalid_name: str = 'name-with-hyphen'
        valid_name: str = project_name
        manager.prompt.side_effect = [invalid_name, valid_name]
        mock_get_adapter.return_value = [project.adapter.type()]
        run_dbt(['init', '--skip-profile-setup'])
        manager.assert_has_calls([
            call.prompt('Enter a name for your project (letters, digits, underscore)'),
            call.prompt('Enter a name for your project (letters, digits, underscore)')
        ])


class TestInitProvidedProjectNameAndSkipProfileSetup(TestInitOutsideOfProjectBase):
    @mock.patch('dbt.task.init._get_adapter_plugin_names')
    @mock.patch('click.confirm')
    @mock.patch('click.prompt')
    def test_init_provided_project_name_and_skip_profile_setup(
        self,
        mock_prompt: Any,
        mock_confirm: Any,
        mock_get: Any,
        project: Any,
        project_name: str
    ) -> None:
        manager: Any = Mock()
        manager.attach_mock(mock_prompt, 'prompt')
        manager.attach_mock(mock_confirm, 'confirm')
        manager.prompt.side_effect = [
            1,
            'localhost',
            5432,
            'test_username',
            'test_password',
            'test_db',
            'test_schema',
            4,
        ]
        mock_get.return_value = [project.adapter.type()]
        run_dbt(['init', project_name, '--skip-profile-setup'])
        assert len(manager.mock_calls) == 0
        with open(os.path.join(project.project_root, project_name, 'dbt_project.yml'), 'r') as f:
            content: str = f.read()
            expected_project: str = (
                "\n# Name your project! Project names should contain only lowercase characters\n"
                "# and underscores. A good package name should reflect your organization's\n"
                "name or the intended use of these models\n"
                f"name: '{project_name}'\nversion: '1.0.0'\n\n"
                "# This setting configures which \"profile\" dbt uses for this project.\n"
                f"profile: '{project_name}'\n\n"
                "# These configurations specify where dbt should look for different types of files.\n"
                "# The `model-paths` config, for example, states that models in this project can be\n"
                "# found in the \"models/\" directory. You probably won't need to change these!\n"
                'model-paths: ["models"]\nanalysis-paths: ["analyses"]\n'
                'test-paths: ["tests"]\nseed-paths: ["seeds"]\nmacro-paths: ["macros"]\n'
                'snapshot-paths: ["snapshots"]\n\n'
                "clean-targets:         # directories to be removed by `dbt clean`\n"
                '  - "target"\n  - "dbt_packages"\n\n\n'
                "# Configuring models\n"
                "# Full documentation: https://docs.getdbt.com/docs/configuring-models\n\n"
                f"# In this example config, we tell dbt to build all models in the example/\n"
                "# directory as views. These settings can be overridden in the individual model\n"
                "# files using the `{{ config(...) }}` macro.\n"
                f"models:\n  {project_name}:\n    # Config indicated by + and applies to all files under models/example/\n"
                "    example:\n      +materialized: view\n"
            )
            assert content == expected_project


class TestInitInsideProjectAndSkipProfileSetup(TestInitInsideOfProjectBase):
    @mock.patch('dbt.task.init._get_adapter_plugin_names')
    @mock.patch('click.confirm')
    @mock.patch('click.prompt')
    def test_init_inside_project_and_skip_profile_setup(
        self,
        mock_prompt: Any,
        mock_confirm: Any,
        mock_get: Any,
        project: Any,
        project_name: str
    ) -> None:
        manager: Any = Mock()
        manager.attach_mock(mock_prompt, 'prompt')
        manager.attach_mock(mock_confirm, 'confirm')
        assert Path('dbt_project.yml').exists()
        run_dbt(['init', '--skip-profile-setup'])
        assert len(manager.mock_calls) == 0


class TestInitOutsideOfProjectWithSpecifiedProfile(TestInitOutsideOfProjectBase):
    @mock.patch('dbt.task.init._get_adapter_plugin_names')
    @mock.patch('click.prompt')
    def test_init_task_outside_of_project_with_specified_profile(
        self,
        mock_prompt: Any,
        mock_get_adapter: Any,
        project: Any,
        project_name: str,
        unique_schema: Any,
        dbt_profile_data: Any
    ) -> None:
        manager: Any = Mock()
        manager.attach_mock(mock_prompt, 'prompt')
        manager.prompt.side_effect = [project_name]
        mock_get_adapter.return_value = [project.adapter.type()]
        run_dbt(['init', '--profile', 'test'])
        manager.assert_has_calls([
            call.prompt('Enter a name for your project (letters, digits, underscore)')
        ])
        with open(os.path.join(project.profiles_dir, 'profiles.yml'), 'r') as f:
            content: str = f.read()
            assert content == yaml.safe_dump(dbt_profile_data)
        with open(os.path.join(project.project_root, project_name, 'dbt_project.yml'), 'r') as f:
            content_project: str = f.read()
            expected_project: str = (
                "\n# Name your project! Project names should contain only lowercase characters\n"
                "# and underscores. A good package name should reflect your organization's\n"
                "name or the intended use of these models\n"
                f"name: '{project_name}'\nversion: '1.0.0'\n\n"
                "# This setting configures which \"profile\" dbt uses for this project.\n"
                "profile: 'test'\n\n"
                "# These configurations specify where dbt should look for different types of files.\n"
                "# The `model-paths` config, for example, states that models in this project can be\n"
                "# found in the \"models/\" directory. You probably won't need to change these!\n"
                'model-paths: ["models"]\nanalysis-paths: ["analyses"]\n'
                'test-paths: ["tests"]\nseed-paths: ["seeds"]\nmacro-paths: ["macros"]\n'
                'snapshot-paths: ["snapshots"]\n\n'
                "clean-targets:         # directories to be removed by `dbt clean`\n"
                '  - "target"\n  - "dbt_packages"\n\n\n'
                "# Configuring models\n"
                "# Full documentation: https://docs.getdbt.com/docs/configuring-models\n\n"
                f"# In this example config, we tell dbt to build all models in the example/\n"
                "# directory as views. These settings can be overridden in the individual model\n"
                "# files using the `{{ config(...) }}` macro.\n"
                f"models:\n  {project_name}:\n    # Config indicated by + and applies to all files under models/example/\n"
                "    example:\n      +materialized: view\n"
            )
            assert content_project == expected_project


class TestInitOutsideOfProjectSpecifyingInvalidProfile(TestInitOutsideOfProjectBase):
    @mock.patch('dbt.task.init._get_adapter_plugin_names')
    @mock.patch('click.prompt')
    def test_init_task_outside_project_specifying_invalid_profile_errors(
        self,
        mock_prompt: Any,
        mock_get_adapter: Any,
        project: Any,
        project_name: str
    ) -> None:
        manager: Any = Mock()
        manager.attach_mock(mock_prompt, 'prompt')
        manager.prompt.side_effect = [project_name]
        mock_get_adapter.return_value = [project.adapter.type()]
        with pytest.raises(DbtRuntimeError) as error:
            run_dbt(['init', '--profile', 'invalid'], expect_pass=False)
            assert 'Could not find profile named invalid' in str(error.value)
        manager.assert_has_calls([
            call.prompt('Enter a name for your project (letters, digits, underscore)')
        ])


class TestInitOutsideOfProjectSpecifyingProfileNoProfilesYml(TestInitOutsideOfProjectBase):
    @mock.patch('dbt.task.init._get_adapter_plugin_names')
    @mock.patch('click.prompt')
    def test_init_task_outside_project_specifying_profile_no_profiles_yml_errors(
        self,
        mock_prompt: Any,
        mock_get_adapter: Any,
        project: Any,
        project_name: str
    ) -> None:
        manager: Any = Mock()
        manager.attach_mock(mock_prompt, 'prompt')
        manager.prompt.side_effect = [project_name]
        mock_get_adapter.return_value = [project.adapter.type()]
        original_isfile = os.path.isfile
        with mock.patch('os.path.isfile', new=lambda path: {'profiles.yml': False}.get(os.path.basename(path), original_isfile(path))):
            with pytest.raises(DbtRuntimeError) as error:
                run_dbt(['init', '--profile', 'test'], expect_pass=False)
                assert 'Could not find profile named invalid' in str(error.value)
        manager.assert_has_calls([
            call.prompt('Enter a name for your project (letters, digits, underscore)')
        ])
