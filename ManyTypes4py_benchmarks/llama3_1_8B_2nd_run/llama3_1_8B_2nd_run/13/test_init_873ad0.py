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
    """Test init task in project with existing profiles.yml"""

    @mock.patch('dbt.task.init._get_adapter_plugin_names')
    @mock.patch('click.confirm')
    @mock.patch('click.prompt')
    def test_init_task_in_project_with_existing_profiles_yml(
        self,
        mock_prompt: mock.Mock,
        mock_confirm: mock.Mock,
        mock_get_adapter: mock.Mock,
        project: object,
    ) -> None:
        """Test init task in project with existing profiles.yml"""
        manager = Mock()
        manager.attach_mock(mock_prompt, 'prompt')
        manager.attach_mock(mock_confirm, 'confirm')
        manager.confirm.side_effect = ['y']
        manager.prompt.side_effect = [1, 'localhost', 5432, 'test_user', 'test_password', 'test_db', 'test_schema', 4]
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
            call.prompt('threads (1 or more)', default=1, hide_input=False, type=click.INT),
        ])
        with open(os.path.join(project.profiles_dir, 'profiles.yml'), 'r') as f:
            assert f.read() == 'test:\n  outputs:\n    dev:\n      dbname: test_db\n      host: localhost\n      pass: test_password\n      port: 5432\n      schema: test_schema\n      threads: 4\n      type: postgres\n      user: test_user\n  target: dev\n'

    def test_init_task_in_project_specifying_profile_errors(
        self,
    ) -> None:
        """Test init task in project specifying profile errors"""
        with pytest.raises(DbtRuntimeError) as error:
            run_dbt(['init', '--profile', 'test'], expect_pass=False)
            assert 'Can not init existing project with specified profile' in str(error)

class TestInitProjectWithoutExistingProfilesYml:
    """Test init task in project without existing profiles.yml"""

    @mock.patch('dbt.task.init._get_adapter_plugin_names')
    @mock.patch('click.prompt')
    @mock.patch.object(Path, 'exists', autospec=True)
    def test_init_task_in_project_without_existing_profiles_yml(
        self,
        exists: mock.Mock,
        mock_prompt: mock.Mock,
        mock_get_adapter: mock.Mock,
        project: object,
    ) -> None:
        """Test init task in project without existing profiles.yml"""
        def exists_side_effect(path: Path) -> bool:
            return {'profiles.yml': False}.get(path.name, os.path.exists(path))

        exists.side_effect = exists_side_effect
        manager = Mock()
        manager.attach_mock(mock_prompt, 'prompt')
        manager.prompt.side_effect = [1, 'localhost', 5432, 'test_user', 'test_password', 'test_db', 'test_schema', 4]
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
            call.prompt('threads (1 or more)', default=1, hide_input=False, type=click.INT),
        ])
        with open(os.path.join(project.profiles_dir, 'profiles.yml'), 'r') as f:
            assert f.read() == 'test:\n  outputs:\n    dev:\n      dbname: test_db\n      host: localhost\n      pass: test_password\n      port: 5432\n      schema: test_schema\n      threads: 4\n      type: postgres\n      user: test_user\n  target: dev\n'

    @mock.patch.object(Path, 'exists', autospec=True)
    def test_init_task_in_project_without_profile_yml_specifying_profile_errors(
        self,
        exists: mock.Mock,
    ) -> None:
        """Test init task in project without profile.yml specifying profile errors"""
        def exists_side_effect(path: Path) -> bool:
            return {'profiles.yml': False}.get(path.name, os.path.exists(path))

        exists.side_effect = exists_side_effect
        with pytest.raises(DbtRuntimeError) as error:
            run_dbt(['init', '--profile', 'test'], expect_pass=False)
            assert 'Could not find profile named test' in str(error)

class TestInitProjectWithoutExistingProfilesYmlOrTemplate:
    """Test init task in project without existing profiles.yml or profile template"""

    @mock.patch('dbt.task.init._get_adapter_plugin_names')
    @mock.patch('click.confirm')
    @mock.patch('click.prompt')
    @mock.patch.object(Path, 'exists', autospec=True)
    def test_init_task_in_project_without_existing_profiles_yml_or_profile_template(
        self,
        exists: mock.Mock,
        mock_prompt: mock.Mock,
        mock_confirm: mock.Mock,
        mock_get_adapter: mock.Mock,
        project: object,
    ) -> None:
        """Test init task in project without existing profiles.yml or profile template"""
        def exists_side_effect(path: Path) -> bool:
            return {'profiles.yml': False, 'profile_template.yml': False}.get(path.name, os.path.exists(path))

        exists.side_effect = exists_side_effect
        manager = Mock()
        manager.attach_mock(mock_prompt, 'prompt')
        manager.attach_mock(mock_confirm, 'confirm')
        manager.prompt.side_effect = [1]
        mock_get_adapter.return_value = [project.adapter.type()]
        run_dbt(['init'])
        manager.assert_has_calls([
            call.prompt("Which database would you like to use?\n[1] postgres\n\n(Don't see the one you want? https://docs.getdbt.com/docs/available-adapters)\n\nEnter a number", type=click.INT),
        ])
        with open(os.path.join(project.profiles_dir, 'profiles.yml'), 'r') as f:
            assert f.read() == 'test:\n  outputs:\n\n    dev:\n      type: postgres\n      threads: [1 or more]\n      host: [host]\n      port: [port]\n      user: [dev_username]\n      pass: [dev_password]\n      dbname: [dbname]\n      schema: [dev_schema]\n\n    prod:\n      type: postgres\n      threads: [1 or more]\n      host: [host]\n      port: [port]\n      user: [prod_username]\n      pass: [prod_password]\n      dbname: [dbname]\n      schema: [prod_schema]\n\n  target: dev\n'

class TestInitProjectWithProfileTemplateWithoutExistingProfilesYml:
    """Test init task in project with profile template without existing profiles.yml"""

    @mock.patch('dbt.task.init._get_adapter_plugin_names')
    @mock.patch('click.confirm')
    @mock.patch('click.prompt')
    @mock.patch.object(Path, 'exists', autospec=True)
    def test_init_task_in_project_with_profile_template_without_existing_profiles_yml(
        self,
        exists: mock.Mock,
        mock_prompt: mock.Mock,
        mock_confirm: mock.Mock,
        mock_get_adapter: mock.Mock,
        project: object,
    ) -> None:
        """Test init task in project with profile template without existing profiles.yml"""
        def exists_side_effect(path: Path) -> bool:
            return {'profiles.yml': False}.get(path.name, os.path.exists(path))

        exists.side_effect = exists_side_effect
        with open('profile_template.yml', 'w') as f:
            f.write(
                "fixed:\n  type: postgres\n  threads: 4\n  host: localhost\n  dbname: my_db\n  schema: my_schema\n  target: my_target\nprompts:\n  target:\n    hint: 'The target name'\n    type: string\n  port:\n    hint: 'The port (for integer test purposes)'\n    type: int\n    default: 5432\n  user:\n    hint: 'Your username'\n  pass:\n    hint: 'Your password'\n    hide_input: true"
            )
        manager = Mock()
        manager.attach_mock(mock_prompt, 'prompt')
        manager.attach_mock(mock_confirm, 'confirm')
        manager.prompt.side_effect = ['my_target', 5432, 'test_username', 'test_password']
        mock_get_adapter.return_value = [project.adapter.type()]
        run_dbt(['init'])
        manager.assert_has_calls([
            call.prompt('target (The target name)', default=None, hide_input=False, type=click.STRING),
            call.prompt('port (The port (for integer test purposes))', default=5432, hide_input=False, type=click.INT),
            call.prompt('user (Your username)', default=None, hide_input=False, type=None),
            call.prompt('pass (Your password)', default=None, hide_input=True, type=None),
        ])
        with open(os.path.join(project.profiles_dir, 'profiles.yml'), 'r') as f:
            assert f.read() == 'test:\n  outputs:\n    my_target:\n      dbname: my_db\n      host: localhost\n      pass: test_password\n      port: 5432\n      schema: my_schema\n      threads: 4\n      type: postgres\n      user: test_username\n  target: my_target\n'

class TestInitInvalidProfileTemplate:
    """Test init task in project with invalid profile template"""

    @mock.patch('dbt.task.init._get_adapter_plugin_names')
    @mock.patch('click.confirm')
    @mock.patch('click.prompt')
    def test_init_task_in_project_with_invalid_profile_template(
        self,
        mock_prompt: mock.Mock,
        mock_confirm: mock.Mock,
        mock_get_adapter: mock.Mock,
        project: object,
    ) -> None:
        """Test init task in project with invalid profile template"""
        with open(os.path.join(project.project_root, 'profile_template.yml'), 'w') as f:
            f.write('invalid template')
        manager = Mock()
        manager.attach_mock(mock_prompt, 'prompt')
        manager.attach_mock(mock_confirm, 'confirm')
        manager.confirm.side_effect = ['y']
        manager.prompt.side_effect = [1, 'localhost', 5432, 'test_username', 'test_password', 'test_db', 'test_schema', 4]
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
            call.prompt('threads (1 or more)', default=1, hide_input=False, type=click.INT),
        ])
        with open(os.path.join(project.profiles_dir, 'profiles.yml'), 'r') as f:
            assert f.read() == 'test:\n  outputs:\n    dev:\n      dbname: test_db\n      host: localhost\n      pass: test_password\n      port: 5432\n      schema: test_schema\n      threads: 4\n      type: postgres\n      user: test_username\n  target: dev\n'

class TestInitInsideOfProjectBase:
    """Test init inside of project base"""

    @pytest.fixture(scope='class')
    def project_name(self, unique_schema: str) -> str:
        return f'my_project_{unique_schema}'

class TestInitOutsideOfProjectBase:
    """Test init outside of project base"""

    @pytest.fixture(scope='class')
    def project_name(self, unique_schema: str) -> str:
        return f'my_project_{unique_schema}'

    @pytest.fixture(scope='class', autouse=True)
    def setup(self, project: object) -> None:
        os.remove(os.path.join(project.project_root, 'dbt_project.yml'))

class TestInitOutsideOfProject(TestInitOutsideOfProjectBase):
    """Test init outside of project"""

    @pytest.fixture(scope='class')
    def dbt_profile_data(self, unique_schema: str) -> dict:
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
                    },
                },
                'target': 'default2',
            },
        }

    @mock.patch('dbt.task.init._get_adapter_plugin_names')
    @mock.patch('click.confirm')
    @mock.patch('click.prompt')
    def test_init_task_outside_of_project(
        self,
        mock_prompt: mock.Mock,
        mock_confirm: mock.Mock,
        mock_get_adapter: mock.Mock,
        project: object,
        project_name: str,
        unique_schema: str,
    ) -> None:
        """Test init task outside of project"""
        manager = Mock()
        manager.attach_mock(mock_prompt, 'prompt')
        manager.attach_mock(mock_confirm, 'confirm')
        manager.prompt.side_effect = [project_name, 1, 'localhost', 5432, 'test_username', 'test_password', 'test_db', 'test_schema', 4]
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
            call.prompt('threads (1 or more)', default=1, hide_input=False, type=click.INT),
        ])
        with open(os.path.join(project.profiles_dir, 'profiles.yml'), 'r') as f:
            assert f.read() == f'{project_name}:\n  outputs:\n    dev:\n      dbname: test_db\n      host: localhost\n      pass: test_password\n      port: 5432\n      schema: test_schema\n      threads: 4\n      type: postgres\n      user: test_username\n  target: dev\ntest:\n  outputs:\n    default2:\n      dbname: dbt\n      host: localhost\n      pass: password\n      port: 5432\n      schema: {unique_schema}\n      threads: 4\n      type: postgres\n      user: root\n    noaccess:\n      dbname: dbt\n      host: localhost\n      pass: password\n      port: 5432\n      schema: {unique_schema}\n      threads: 4\n      type: postgres\n      user: noaccess\n  target: default2\n'
        with open(os.path.join(project.project_root, project_name, 'dbt_project.yml'), 'r') as f:
            assert f.read() == f"""\
# Name your project! Project names should contain only lowercase characters
# and underscores. A good package name should reflect your organization's
# name or the intended use of these models
name: '{project_name}'
version: '1.0.0'

# This setting configures which "profile" dbt uses for this project.
profile: '{project_name}'

# These configurations specify where dbt should look for different types of files.
# The `model-paths` config, for example, states that models in this project can be
# found in the "models/" directory. You probably won't need to change these!
model-paths: ["models"]
analysis-paths: ["analyses"]
test-paths: ["tests"]
seed-paths: ["seeds"]
macro-paths: ["macros"]
snapshot-paths: ["snapshots"]

clean-targets:         # directories to be removed by `dbt clean`
  - "target"
  - "dbt_packages"

# Configuring models
# Full documentation: https://docs.getdbt.com/docs/configuring-models

# In this example config, we tell dbt to build all models in the example/
# directory as views. These settings can be overridden in the individual model
# files using the `{{{{ config(...) }}}}` macro.
models:
  {project_name}:
    # Config indicated by + and applies to all files under models/example/
    example:
      +materialized: view
"""

class TestInitInvalidProjectNameCLI(TestInitOutsideOfProjectBase):
    """Test init invalid project name CLI"""

    @mock.patch('dbt.task.init._get_adapter_plugin_names')
    @mock.patch('click.confirm')
    @mock.patch('click.prompt')
    def test_init_invalid_project_name_cli(
        self,
        mock_prompt: mock.Mock,
        mock_confirm: mock.Mock,
        mock_get_adapter: mock.Mock,
        project_name: str,
        project: object,
    ) -> None:
        """Test init invalid project name CLI"""
        manager = Mock()
        manager.attach_mock(mock_prompt, 'prompt')
        manager.attach_mock(mock_confirm, 'confirm')
        invalid_name = 'name-with-hyphen'
        valid_name = project_name
        manager.prompt.side_effect = [valid_name]
        mock_get_adapter.return_value = [project.adapter.type()]
        run_dbt(['init', invalid_name, '--skip-profile-setup'])
        manager.assert_has_calls([call.prompt('Enter a name for your project (letters, digits, underscore)')])

class TestInitInvalidProjectNamePrompt(TestInitOutsideOfProjectBase):
    """Test init invalid project name prompt"""

    @mock.patch('dbt.task.init._get_adapter_plugin_names')
    @mock.patch('click.confirm')
    @mock.patch('click.prompt')
    def test_init_invalid_project_name_prompt(
        self,
        mock_prompt: mock.Mock,
        mock_confirm: mock.Mock,
        mock_get_adapter: mock.Mock,
        project_name: str,
        project: object,
    ) -> None:
        """Test init invalid project name prompt"""
        manager = Mock()
        manager.attach_mock(mock_prompt, 'prompt')
        manager.attach_mock(mock_confirm, 'confirm')
        invalid_name = 'name-with-hyphen'
        valid_name = project_name
        manager.prompt.side_effect = [invalid_name, valid_name]
        mock_get_adapter.return_value = [project.adapter.type()]
        run_dbt(['init', '--skip-profile-setup'])
        manager.assert_has_calls([call.prompt('Enter a name for your project (letters, digits, underscore)'), call.prompt('Enter a name for your project (letters, digits, underscore)')])

class TestInitProvidedProjectNameAndSkipProfileSetup(TestInitOutsideOfProjectBase):
    """Test init provided project name and skip profile setup"""

    @mock.patch('dbt.task.init._get_adapter_plugin_names')
    @mock.patch('click.confirm')
    @mock.patch('click.prompt')
    def test_init_provided_project_name_and_skip_profile_setup(
        self,
        mock_prompt: mock.Mock,
        mock_confirm: mock.Mock,
        mock_get: mock.Mock,
        project: object,
        project_name: str,
    ) -> None:
        """Test init provided project name and skip profile setup"""
        manager = mock.Mock()
        manager.attach_mock(mock_prompt, 'prompt')
        manager.attach_mock(mock_confirm, 'confirm')
        manager.prompt.side_effect = [1, 'localhost', 5432, 'test_username', 'test_password', 'test_db', 'test_schema', 4]
        mock_get.return_value = [project.adapter.type()]
        run_dbt(['init', project_name, '--skip-profile-setup'])
        assert len(manager.mock_calls) == 0
        with open(os.path.join(project.project_root, project_name, 'dbt_project.yml'), 'r') as f:
            assert f.read() == f"""\
# Name your project! Project names should contain only lowercase characters
# and underscores. A good package name should reflect your organization's
# name or the intended use of these models
name: '{project_name}'
version: '1.0.0'

# This setting configures which "profile" dbt uses for this project.
profile: '{project_name}'

# These configurations specify where dbt should look for different types of files.
# The `model-paths` config, for example, states that models in this project can be
# found in the "models/" directory. You probably won't need to change these!
model-paths: ["models"]
analysis-paths: ["analyses"]
test-paths: ["tests"]
seed-paths: ["seeds"]
macro-paths: ["macros"]
snapshot-paths: ["snapshots"]

clean-targets:         # directories to be removed by `dbt clean`
  - "target"
  - "dbt_packages"

# Configuring models
# Full documentation: https://docs.getdbt.com/docs/configuring-models

# In this example config, we tell dbt to build all models in the example/
# directory as views. These settings can be overridden in the individual model
# files using the `{{{{ config(...) }}}}` macro.
models:
  {project_name}:
    # Config indicated by + and applies to all files under models/example/
    example:
      +materialized: view
"""

class TestInitInsideProjectAndSkipProfileSetup(TestInitInsideOfProjectBase):
    """Test init inside project and skip profile setup"""

    @mock.patch('dbt.task.init._get_adapter_plugin_names')
    @mock.patch('click.confirm')
    @mock.patch('click.prompt')
    def test_init_inside_project_and_skip_profile_setup(
        self,
        mock_prompt: mock.Mock,
        mock_confirm: mock.Mock,
        mock_get: mock.Mock,
        project: object,
        project_name: str,
    ) -> None:
        """Test init inside project and skip profile setup"""
        manager = mock.Mock()
        manager.attach_mock(mock_prompt, 'prompt')
        manager.attach_mock(mock_confirm, 'confirm')
        assert Path('dbt_project.yml').exists()
        run_dbt(['init', '--skip-profile-setup'])
        assert len(manager.mock_calls) == 0

class TestInitOutsideOfProjectWithSpecifiedProfile(TestInitOutsideOfProjectBase):
    """Test init outside of project with specified profile"""

    @mock.patch('dbt.task.init._get_adapter_plugin_names')
    @mock.patch('click.prompt')
    def test_init_task_outside_of_project_with_specified_profile(
        self,
        mock_prompt: mock.Mock,
        mock_get_adapter: mock.Mock,
        project: object,
        project_name: str,
        unique_schema: str,
        dbt_profile_data: dict,
    ) -> None:
        """Test init task outside of project with specified profile"""
        manager = Mock()
        manager.attach_mock(mock_prompt, 'prompt')
        manager.prompt.side_effect = [project_name]
        mock_get_adapter.return_value = [project.adapter.type()]
        run_dbt(['init', '--profile', 'test'])
        manager.assert_has_calls([call.prompt('Enter a name for your project (letters, digits, underscore)')])
        with open(os.path.join(project.profiles_dir, 'profiles.yml'), 'r') as f:
            assert f.read() == yaml.safe_dump(dbt_profile_data)
        with open(os.path.join(project.project_root, project_name, 'dbt_project.yml'), 'r') as f:
            assert f.read() == f"""\
# Name your project! Project names should contain only lowercase characters
# and underscores. A good package name should reflect your organization's
# name or the intended use of these models
name: '{project_name}'
version: '1.0.0'

# This setting configures which "profile" dbt uses for this project.
profile: 'test'

# These configurations specify where dbt should look for different types of files.
# The `model-paths` config, for example, states that models in this project can be
# found in the "models/" directory. You probably won't need to change these!
model-paths: ["models"]
analysis-paths: ["analyses"]
test-paths: ["tests"]
seed-paths: ["seeds"]
macro-paths: ["macros"]
snapshot-paths: ["snapshots"]

clean-targets:         # directories to be removed by `dbt clean`
  - "target"
  - "dbt_packages"

# Configuring models
# Full documentation: https://docs.getdbt.com/docs/configuring-models

# In this example config, we tell dbt to build all models in the example/
# directory as views. These settings can be overridden in the individual model
# files using the `{{{{ config(...) }}}}` macro.
models:
  {project_name}:
    # Config indicated by + and applies to all files under models/example/
    example:
      +materialized: view
"""

class TestInitOutsideOfProjectSpecifyingInvalidProfile(TestInitOutsideOfProjectBase):
    """Test init outside of project specifying invalid profile"""

    @mock.patch('dbt.task.init._get_adapter_plugin_names')
    @mock.patch('click.prompt')
    def test_init_task_outside_project_specifying_invalid_profile_errors(
        self,
        mock_prompt: mock.Mock,
        mock_get_adapter: mock.Mock,
        project: object,
        project_name: str,
    ) -> None:
        """Test init task outside of project specifying invalid profile errors"""
        manager = Mock()
        manager.attach_mock(mock_prompt, 'prompt')
        manager.prompt.side_effect = [project_name]
        mock_get_adapter.return_value = [project.adapter.type()]
        with pytest.raises(DbtRuntimeError) as error:
            run_dbt(['init', '--profile', 'invalid'], expect_pass=False)
            assert 'Could not find profile named invalid' in str(error)
        manager.assert_has_calls([call.prompt('Enter a name for your project (letters, digits, underscore)')])

class TestInitOutsideOfProjectSpecifyingProfileNoProfilesYml(TestInitOutsideOfProjectBase):
    """Test init outside of project specifying profile no profiles.yml"""

    @mock.patch('dbt.task.init._get_adapter_plugin_names')
    @mock.patch('click.prompt')
    def test_init_task_outside_project_specifying_profile_no_profiles_yml_errors(
        self,
        mock_prompt: mock.Mock,
        mock_get_adapter: mock.Mock,
        project: object,
        project_name: str,
    ) -> None:
        """Test init task outside of project specifying profile no profiles.yml errors"""
        manager = Mock()
        manager.attach_mock(mock_prompt, 'prompt')
        manager.prompt.side_effect = [project_name]
        mock_get_adapter.return_value = [project.adapter.type()]
        original_isfile = os.path.isfile
        with mock.patch('os.path.isfile', new=lambda path: {'profiles.yml': False}.get(os.path.basename(path), original_isfile(path))):
            with pytest.raises(DbtRuntimeError) as error:
                run_dbt(['init', '--profile', 'test'], expect_pass=False)
                assert 'Could not find profile named invalid' in str(error)
        manager.assert_has_calls([call.prompt('Enter a name for your project (letters, digits, underscore)')])
