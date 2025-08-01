#!/usr/bin/env python3
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock
import pytest
import yaml
from dbt.artifacts.resources.v1.components import FreshnessThreshold, Time
from dbt.cli.main import DbtUsageException, dbtRunnerResult
from dbt.contracts.files import FileHash
from dbt.contracts.graph.nodes import ModelNode, SourceDefinition
from dbt.contracts.results import (
    FreshnessMetadata,
    FreshnessResult,
    RunExecutionResult,
    RunResult,
    SourceFreshnessResult,
)
from prefect_dbt.cli.commands import (
    DbtCoreOperation,
    run_dbt_build,
    run_dbt_model,
    run_dbt_seed,
    run_dbt_snapshot,
    run_dbt_source_freshness,
    run_dbt_test,
    trigger_dbt_cli_command,
)
from prefect_dbt.cli.credentials import DbtCliProfile
from prefect import flow
from prefect.artifacts import Artifact
from prefect.testing.utilities import AsyncMock

# Fixture type annotations

@pytest.fixture
async def mock_dbt_runner_model_success() -> dbtRunnerResult:
    return dbtRunnerResult(
        success=True,
        exception=None,
        result=RunExecutionResult(
            results=[
                RunResult(
                    status='pass',
                    timing=None,
                    thread_id="'Thread-1 (worker)'",
                    message='CREATE TABLE (1.0 rows, 0 processed)',
                    failures=None,
                    node=ModelNode(
                        database='test-123',
                        schema='prefect_dbt_example',
                        name='my_first_dbt_model',
                        resource_type='model',
                        package_name='prefect_dbt_bigquery',
                        path='example/my_first_dbt_model.sql',
                        original_file_path='models/example/my_first_dbt_model.sql',
                        unique_id='model.prefect_dbt_bigquery.my_first_dbt_model',
                        fqn=['prefect_dbt_bigquery', 'example', 'my_first_dbt_model'],
                        alias='my_first_dbt_model',
                        checksum=FileHash(name='sha256', checksum='123456789'),
                    ),
                    execution_time=0.0,
                    adapter_response=None,
                )
            ],
            elapsed_time=0.0,
        ),
    )


@pytest.fixture
async def mock_dbt_runner_model_error() -> dbtRunnerResult:
    return dbtRunnerResult(
        success=False,
        exception=None,
        result=RunExecutionResult(
            results=[
                RunResult(
                    status='error',
                    timing=None,
                    thread_id="'Thread-1 (worker)'",
                    message='Runtime Error',
                    failures=None,
                    node=ModelNode(
                        database='test-123',
                        schema='prefect_dbt_example',
                        name='my_first_dbt_model',
                        resource_type='model',
                        package_name='prefect_dbt_bigquery',
                        path='example/my_first_dbt_model.sql',
                        original_file_path='models/example/my_first_dbt_model.sql',
                        unique_id='model.prefect_dbt_bigquery.my_first_dbt_model',
                        fqn=['prefect_dbt_bigquery', 'example', 'my_first_dbt_model'],
                        alias='my_first_dbt_model',
                        checksum=FileHash(name='sha256', checksum='123456789'),
                    ),
                    execution_time=0.0,
                    adapter_response=None,
                )
            ],
            elapsed_time=0.0,
        ),
    )


@pytest.fixture
async def mock_dbt_runner_freshness_success() -> dbtRunnerResult:
    now = datetime.datetime.now()
    return dbtRunnerResult(
        success=True,
        exception=None,
        result=FreshnessResult(
            results=[
                SourceFreshnessResult(
                    status='pass',
                    thread_id='Thread-1 (worker)',
                    execution_time=0.0,
                    adapter_response={},
                    message=None,
                    failures=None,
                    max_loaded_at=now,
                    snapshotted_at=now,
                    timing=[],
                    node=SourceDefinition(
                        database='test-123',
                        schema='prefect_dbt_example',
                        name='my_first_dbt_model',
                        resource_type='source',
                        package_name='prefect_dbt_bigquery',
                        path='example/my_first_dbt_model.yml',
                        original_file_path='models/example/my_first_dbt_model.yml',
                        unique_id='source.prefect_dbt_bigquery.my_first_dbt_model',
                        fqn=['prefect_dbt_bigquery', 'example', 'my_first_dbt_model'],
                        source_name='prefect_dbt_source',
                        source_description='',
                        description='',
                        loader='my_loader',
                        identifier='my_identifier',
                        freshness=FreshnessThreshold(
                            warn_after=Time(count=12, period='hour'),
                            error_after=Time(count=24, period='hour'),
                            filter=None,
                        ),
                    ),
                    age=0.0,
                )
            ],
            elapsed_time=0.0,
            metadata=FreshnessMetadata(
                dbt_schema_version='https://schemas.getdbt.com/dbt/sources/v3.json',
                dbt_version='1.1.1',
                generated_at=now,
                invocation_id='invocation_id',
                env={},
            ),
        ),
    )


@pytest.fixture
async def mock_dbt_runner_freshness_error() -> dbtRunnerResult:
    now = datetime.datetime.now()
    return dbtRunnerResult(
        success=False,
        exception=None,
        result=FreshnessResult(
            results=[
                SourceFreshnessResult(
                    status='error',
                    thread_id='Thread-1 (worker)',
                    execution_time=0.0,
                    adapter_response={},
                    message=None,
                    failures=None,
                    max_loaded_at=now,
                    snapshotted_at=now,
                    timing=[],
                    node=SourceDefinition(
                        database='test-123',
                        schema='prefect_dbt_example',
                        name='my_first_dbt_model',
                        resource_type='source',
                        package_name='prefect_dbt_bigquery',
                        path='example/my_first_dbt_model.yml',
                        original_file_path='models/example/my_first_dbt_model.yml',
                        unique_id='source.prefect_dbt_bigquery.my_first_dbt_model',
                        fqn=['prefect_dbt_bigquery', 'example', 'my_first_dbt_model'],
                        source_name='my_first_dbt_source',
                        source_description='',
                        description='',
                        loader='my_loader',
                        identifier='my_identifier',
                        freshness=FreshnessThreshold(
                            warn_after=Time(count=12, period='hour'),
                            error_after=Time(count=24, period='hour'),
                            filter=None,
                        ),
                    ),
                    age=0.0,
                )
            ],
            elapsed_time=0.0,
            metadata=FreshnessMetadata(
                dbt_schema_version='https://schemas.getdbt.com/dbt/sources/v3.json',
                dbt_version='1.1.1',
                generated_at=now,
                invocation_id='invocation_id',
                env={},
            ),
        ),
    )


@pytest.fixture
async def mock_dbt_runner_ls_success() -> dbtRunnerResult:
    return dbtRunnerResult(success=True, exception=None, result=['example.example.test_model'])


@pytest.fixture
def dbt_runner_model_result(monkeypatch: Any, mock_dbt_runner_model_success: dbtRunnerResult) -> None:
    _mock_dbt_runner_invoke_success: MagicMock = MagicMock(return_value=mock_dbt_runner_model_success)
    monkeypatch.setattr('dbt.cli.main.dbtRunner.invoke', _mock_dbt_runner_invoke_success)


@pytest.fixture
def dbt_runner_ls_result(monkeypatch: Any, mock_dbt_runner_ls_success: dbtRunnerResult) -> None:
    _mock_dbt_runner_ls_result: MagicMock = MagicMock(return_value=mock_dbt_runner_ls_success)
    monkeypatch.setattr('dbt.cli.main.dbtRunner.invoke', _mock_dbt_runner_ls_result)


@pytest.fixture
def dbt_runner_freshness_error(monkeypatch: Any, mock_dbt_runner_freshness_error: dbtRunnerResult) -> None:
    _mock_dbt_runner_freshness_error: MagicMock = MagicMock(return_value=mock_dbt_runner_freshness_error)
    monkeypatch.setattr('dbt.cli.main.dbtRunner.invoke', _mock_dbt_runner_freshness_error)


@pytest.fixture
def dbt_runner_freshness_success(monkeypatch: Any, mock_dbt_runner_freshness_success: dbtRunnerResult) -> MagicMock:
    _mock_dbt_runner_freshness_success: MagicMock = MagicMock(return_value=mock_dbt_runner_freshness_success)
    monkeypatch.setattr('dbt.cli.main.dbtRunner.invoke', _mock_dbt_runner_freshness_success)
    return _mock_dbt_runner_freshness_success


@pytest.fixture
def dbt_runner_failed_result(monkeypatch: Any) -> None:
    _mock_dbt_runner_invoke_failed: MagicMock = MagicMock(
        return_value=dbtRunnerResult(
            success=False,
            exception=DbtUsageException("No such command 'weeeeeee'."),
            result=None,
        )
    )
    monkeypatch.setattr('dbt.cli.main.dbtRunner.invoke', _mock_dbt_runner_invoke_failed)


@pytest.fixture
def profiles_dir(tmp_path: Path) -> str:
    return str(tmp_path) + '/.dbt'


@pytest.mark.usefixtures('dbt_runner_ls_result')
def test_trigger_dbt_cli_command(profiles_dir: str, dbt_cli_profile_bare: DbtCliProfile) -> None:
    @flow
    def test_flow() -> dbtRunnerResult:
        return trigger_dbt_cli_command(
            command='dbt ls',
            profiles_dir=profiles_dir,
            dbt_cli_profile=dbt_cli_profile_bare,
        )
    result: dbtRunnerResult = test_flow()
    assert isinstance(result, dbtRunnerResult)


def test_trigger_dbt_cli_command_cli_argument_list(
    profiles_dir: str, dbt_cli_profile_bare: DbtCliProfile, dbt_runner_freshness_success: MagicMock
) -> None:
    @flow
    def test_flow() -> dbtRunnerResult:
        return trigger_dbt_cli_command(
            command='dbt source freshness',
            profiles_dir=profiles_dir,
            dbt_cli_profile=dbt_cli_profile_bare,
        )
    test_flow()
    dbt_runner_freshness_success.assert_called_with(['source', 'freshness', '--profiles-dir', profiles_dir])


@pytest.mark.usefixtures('dbt_runner_freshness_error')
def test_trigger_dbt_cli_command_failed(profiles_dir: str, dbt_cli_profile_bare: DbtCliProfile) -> None:
    @flow
    def test_flow() -> dbtRunnerResult:
        return trigger_dbt_cli_command(
            command='dbt source freshness',
            profiles_dir=profiles_dir,
            dbt_cli_profile=dbt_cli_profile_bare,
        )
    with pytest.raises(Exception, match='dbt task result success: False with exception: None'):
        test_flow()


@pytest.mark.usefixtures('dbt_runner_ls_result')
def test_trigger_dbt_cli_command_run_twice_overwrite(
    profiles_dir: str, dbt_cli_profile: DbtCliProfile, dbt_cli_profile_bare: DbtCliProfile
) -> None:
    @flow
    def test_flow() -> dbtRunnerResult:
        trigger_dbt_cli_command(
            command='dbt ls',
            profiles_dir=profiles_dir,
            dbt_cli_profile=dbt_cli_profile,
        )
        run_two: dbtRunnerResult = trigger_dbt_cli_command(
            command='dbt ls',
            profiles_dir=profiles_dir,
            dbt_cli_profile=dbt_cli_profile_bare,
            overwrite_profiles=True,
        )
        return run_two
    result: dbtRunnerResult = test_flow()
    assert isinstance(result, dbtRunnerResult)
    with open(profiles_dir + '/profiles.yml', 'r') as f:
        actual: Dict[str, Any] = yaml.safe_load(f)
    expected: Dict[str, Any] = {
        'config': {},
        'prefecto': {
            'target': 'testing',
            'outputs': {'testing': {'type': 'custom', 'schema': 'my_schema', 'threads': 4, 'account': 'fake'}},
        },
    }
    assert actual == expected


@pytest.mark.usefixtures('dbt_runner_ls_result')
def test_trigger_dbt_cli_command_run_twice_exists(
    profiles_dir: str, dbt_cli_profile: DbtCliProfile, dbt_cli_profile_bare: DbtCliProfile
) -> None:
    @flow
    def test_flow() -> dbtRunnerResult:
        trigger_dbt_cli_command('dbt ls', profiles_dir=profiles_dir, dbt_cli_profile=dbt_cli_profile)
        run_two: dbtRunnerResult = trigger_dbt_cli_command(
            'dbt ls', profiles_dir=profiles_dir, dbt_cli_profile=dbt_cli_profile_bare
        )
        return run_two
    with pytest.raises(ValueError, match='Since overwrite_profiles is False'):
        test_flow()


@pytest.mark.usefixtures('dbt_runner_ls_result')
def test_trigger_dbt_cli_command_missing_profile(profiles_dir: str) -> None:
    @flow
    def test_flow() -> dbtRunnerResult:
        return trigger_dbt_cli_command('dbt ls', profiles_dir=profiles_dir)
    with pytest.raises(ValueError, match='Profile not found. Provide `dbt_cli_profile` or'):
        test_flow()


@pytest.mark.usefixtures('dbt_runner_ls_result')
def test_trigger_dbt_cli_command_find_home(dbt_cli_profile_bare: Optional[DbtCliProfile]) -> None:
    home_dbt_dir: Path = Path.home() / '.dbt'
    if (home_dbt_dir / 'profiles.yml').exists():
        dbt_cli_profile: Optional[DbtCliProfile] = None
    else:
        dbt_cli_profile = dbt_cli_profile_bare

    @flow
    def test_flow() -> dbtRunnerResult:
        return trigger_dbt_cli_command('ls', dbt_cli_profile=dbt_cli_profile, overwrite_profiles=False)
    result: dbtRunnerResult = test_flow()
    assert isinstance(result, dbtRunnerResult)


@pytest.mark.usefixtures('dbt_runner_ls_result')
def test_trigger_dbt_cli_command_find_env(
    profiles_dir: str, dbt_cli_profile_bare: DbtCliProfile, monkeypatch: Any
) -> None:
    @flow
    def test_flow() -> dbtRunnerResult:
        return trigger_dbt_cli_command('ls', dbt_cli_profile=dbt_cli_profile_bare)
    monkeypatch.setenv('DBT_PROFILES_DIR', str(profiles_dir))
    result: dbtRunnerResult = test_flow()
    assert isinstance(result, dbtRunnerResult)


@pytest.mark.usefixtures('dbt_runner_ls_result')
def test_trigger_dbt_cli_command_project_dir(profiles_dir: str, dbt_cli_profile_bare: DbtCliProfile) -> None:
    @flow
    def test_flow() -> dbtRunnerResult:
        return trigger_dbt_cli_command(
            'dbt ls',
            profiles_dir=profiles_dir,
            dbt_cli_profile=dbt_cli_profile_bare,
            project_dir='project',
        )
    result: dbtRunnerResult = test_flow()
    assert isinstance(result, dbtRunnerResult)


@pytest.mark.usefixtures('dbt_runner_ls_result')
def test_trigger_dbt_cli_command_extra_command_args(
    profiles_dir: str, dbt_cli_profile_bare: DbtCliProfile
) -> None:
    @flow
    def test_flow() -> dbtRunnerResult:
        return trigger_dbt_cli_command(
            'dbt ls',
            profiles_dir=profiles_dir,
            dbt_cli_profile=dbt_cli_profile_bare,
            extra_command_args=['--return_all', 'True'],
        )
    result: dbtRunnerResult = test_flow()
    assert isinstance(result, dbtRunnerResult)


class TestDbtCoreOperation:
    @pytest.fixture
    def mock_open_process(self, monkeypatch: Any) -> MagicMock:
        open_process: MagicMock = MagicMock(name='open_process')
        open_process.return_value = AsyncMock(name='returned open_process')
        monkeypatch.setattr('prefect_shell.commands.open_process', open_process)
        return open_process

    @pytest.fixture
    def mock_shell_process(self, monkeypatch: Any) -> MagicMock:
        shell_process: MagicMock = MagicMock()
        opened_shell_process = AsyncMock()
        shell_process.return_value = opened_shell_process
        monkeypatch.setattr('prefect_shell.commands.ShellProcess', shell_process)
        return shell_process

    @pytest.fixture
    def dbt_cli_profile(self) -> DbtCliProfile:
        return DbtCliProfile(
            name='my_name',
            target='my_target',
            target_configs={'type': 'my_type', 'threads': 4, 'schema': 'my_schema'},
        )

    def test_find_valid_profiles_dir_default_env(
        self, tmp_path: Path, mock_open_process: MagicMock, mock_shell_process: MagicMock, monkeypatch: Any
    ) -> None:
        monkeypatch.setenv('DBT_PROFILES_DIR', str(tmp_path))
        (tmp_path / 'profiles.yml').write_text('test')
        DbtCoreOperation(commands=['dbt debug']).run()
        actual: str = str(mock_open_process.call_args_list[0][1]['env']['DBT_PROFILES_DIR'])
        expected: str = str(tmp_path)
        assert actual == expected

    def test_find_valid_profiles_dir_input_env(
        self, tmp_path: Path, mock_open_process: MagicMock, mock_shell_process: MagicMock
    ) -> None:
        (tmp_path / 'profiles.yml').write_text('test')
        DbtCoreOperation(commands=['dbt debug'], env={'DBT_PROFILES_DIR': str(tmp_path)}).run()
        actual: str = str(mock_open_process.call_args_list[0][1]['env']['DBT_PROFILES_DIR'])
        expected: str = str(tmp_path)
        assert actual == expected

    def test_find_valid_profiles_dir_overwrite_without_profile(
        self, tmp_path: Path, mock_open_process: MagicMock, mock_shell_process: MagicMock
    ) -> None:
        with pytest.raises(ValueError, match='Since overwrite_profiles is True'):
            DbtCoreOperation(commands=['dbt debug'], profiles_dir=tmp_path, overwrite_profiles=True).run()

    def test_find_valid_profiles_dir_overwrite_with_profile(
        self, tmp_path: Path, dbt_cli_profile: DbtCliProfile, mock_open_process: MagicMock, mock_shell_process: MagicMock
    ) -> None:
        DbtCoreOperation(
            commands=['dbt debug'],
            profiles_dir=tmp_path,
            overwrite_profiles=True,
            dbt_cli_profile=dbt_cli_profile,
        ).run()
        assert (tmp_path / 'profiles.yml').exists()

    def test_find_valid_profiles_dir_not_overwrite_with_profile(
        self, tmp_path: Path, dbt_cli_profile: DbtCliProfile, mock_open_process: MagicMock, mock_shell_process: MagicMock
    ) -> None:
        (tmp_path / 'profiles.yml').write_text('test')
        with pytest.raises(ValueError, match='Since overwrite_profiles is False'):
            DbtCoreOperation(
                commands=['dbt debug'],
                profiles_dir=tmp_path,
                overwrite_profiles=False,
                dbt_cli_profile=dbt_cli_profile,
            ).run()

    def test_find_valid_profiles_dir_path_without_profile(self) -> None:
        with pytest.raises(ValueError, match='Since overwrite_profiles is True'):
            DbtCoreOperation(commands=['dbt debug'], profiles_dir=Path('fake')).run()

    def test_append_dirs_to_commands(
        self,
        tmp_path: Path,
        dbt_cli_profile: DbtCliProfile,
        mock_open_process: MagicMock,
        mock_shell_process: MagicMock,
        monkeypatch: Any,
    ) -> None:
        mock_named_temporary_file: MagicMock = MagicMock(name='tempfile')
        monkeypatch.setattr('tempfile.NamedTemporaryFile', mock_named_temporary_file)
        try:
            with DbtCoreOperation(
                commands=['dbt debug'],
                profiles_dir=tmp_path,
                project_dir=tmp_path,
                dbt_cli_profile=dbt_cli_profile,
            ) as op:
                op.run()
        except (FileNotFoundError, TypeError):
            pass
        mock_write: MagicMock = mock_named_temporary_file.return_value.write
        assert mock_write.call_args_list[0][0][0] == f'dbt debug --profiles-dir {tmp_path} --project-dir {tmp_path}'.encode()


@pytest.mark.usefixtures('dbt_runner_freshness_success')
def test_sync_dbt_cli_command_creates_artifact(profiles_dir: str, dbt_cli_profile: DbtCliProfile) -> None:
    @flow
    def test_flow() -> None:
        trigger_dbt_cli_command(
            command='dbt source freshness',
            profiles_dir=profiles_dir,
            dbt_cli_profile=dbt_cli_profile,
            summary_artifact_key='foo',
            create_summary_artifact=True,
        )
    test_flow()
    a: Artifact = Artifact.get(key='foo')
    assert a
    assert a.type == 'markdown'
    assert isinstance(a.data, str) and a.data.startswith('#  dbt source freshness Task Summary')
    assert 'my_first_dbt_model' in a.data
    assert 'Successful Nodes' in a.data


@pytest.mark.usefixtures('dbt_runner_model_result')
async def test_run_dbt_build_creates_artifact(profiles_dir: str, dbt_cli_profile_bare: DbtCliProfile) -> None:
    @flow
    async def test_flow() -> dbtRunnerResult:
        return await run_dbt_build(
            profiles_dir=profiles_dir,
            dbt_cli_profile=dbt_cli_profile_bare,
            summary_artifact_key='foo',
            create_summary_artifact=True,
        )
    await test_flow()
    a: Artifact = await Artifact.get(key='foo')
    assert a
    assert a.type == 'markdown'
    assert a.data.startswith('# dbt build Task Summary')
    assert 'my_first_dbt_model' in a.data
    assert 'Successful Nodes' in a.data


@pytest.mark.usefixtures('dbt_runner_model_result')
async def test_run_dbt_test_creates_artifact(profiles_dir: str, dbt_cli_profile_bare: DbtCliProfile) -> None:
    @flow
    async def test_flow() -> dbtRunnerResult:
        return await run_dbt_test(
            profiles_dir=profiles_dir,
            dbt_cli_profile=dbt_cli_profile_bare,
            summary_artifact_key='foo',
            create_summary_artifact=True,
        )
    await test_flow()
    a: Artifact = await Artifact.get(key='foo')
    assert a
    assert a.type == 'markdown'
    assert a.data.startswith('# dbt test Task Summary')
    assert 'my_first_dbt_model' in a.data
    assert 'Successful Nodes' in a.data


@pytest.mark.usefixtures('dbt_runner_model_result')
async def test_run_dbt_snapshot_creates_artifact(profiles_dir: str, dbt_cli_profile_bare: DbtCliProfile) -> None:
    @flow
    async def test_flow() -> dbtRunnerResult:
        return await run_dbt_snapshot(
            profiles_dir=profiles_dir,
            dbt_cli_profile=dbt_cli_profile_bare,
            summary_artifact_key='foo',
            create_summary_artifact=True,
        )
    await test_flow()
    a: Artifact = await Artifact.get(key='foo')
    assert a
    assert a.type == 'markdown'
    assert a.data.startswith('# dbt snapshot Task Summary')
    assert 'my_first_dbt_model' in a.data
    assert 'Successful Nodes' in a.data


@pytest.mark.usefixtures('dbt_runner_model_result')
async def test_run_dbt_seed_creates_artifact(profiles_dir: str, dbt_cli_profile_bare: DbtCliProfile) -> None:
    @flow
    async def test_flow() -> dbtRunnerResult:
        return await run_dbt_seed(
            profiles_dir=profiles_dir,
            dbt_cli_profile=dbt_cli_profile_bare,
            summary_artifact_key='foo',
            create_summary_artifact=True,
        )
    await test_flow()
    a: Artifact = await Artifact.get(key='foo')
    assert a
    assert a.type == 'markdown'
    assert a.data.startswith('# dbt seed Task Summary')
    assert 'my_first_dbt_model' in a.data
    assert 'Successful Nodes' in a.data


@pytest.mark.usefixtures('dbt_runner_model_result')
async def test_run_dbt_model_creates_artifact(profiles_dir: str, dbt_cli_profile_bare: DbtCliProfile) -> None:
    @flow
    async def test_flow() -> dbtRunnerResult:
        return await run_dbt_model(
            profiles_dir=profiles_dir,
            dbt_cli_profile=dbt_cli_profile_bare,
            summary_artifact_key='foo',
            create_summary_artifact=True,
        )
    await test_flow()
    a: Artifact = await Artifact.get(key='foo')
    assert a
    assert a.type == 'markdown'
    assert a.data.startswith('# dbt run Task Summary')
    assert 'my_first_dbt_model' in a.data
    assert 'Successful Nodes' in a.data


@pytest.mark.usefixtures('dbt_runner_freshness_success')
async def test_run_dbt_source_freshness_creates_artifact(profiles_dir: str, dbt_cli_profile_bare: DbtCliProfile) -> None:
    @flow
    async def test_flow() -> dbtRunnerResult:
        return await run_dbt_source_freshness(
            profiles_dir=profiles_dir,
            dbt_cli_profile=dbt_cli_profile_bare,
            summary_artifact_key='foo',
            create_summary_artifact=True,
        )
    await test_flow()
    a: Artifact = await Artifact.get(key='foo')
    assert a
    assert a.type == 'markdown'
    assert a.data.startswith('# dbt source freshness Task Summary')
    assert 'my_first_dbt_model' in a.data
    assert 'Successful Nodes' in a.data


@pytest.fixture
def dbt_runner_model_error(monkeypatch: Any, mock_dbt_runner_model_error: dbtRunnerResult) -> None:
    _mock_dbt_runner_invoke_error: MagicMock = MagicMock(return_value=mock_dbt_runner_model_error)
    monkeypatch.setattr('dbt.cli.main.dbtRunner.invoke', _mock_dbt_runner_invoke_error)


@pytest.mark.usefixtures('dbt_runner_model_error')
async def test_run_dbt_model_creates_unsuccessful_artifact(profiles_dir: str, dbt_cli_profile_bare: DbtCliProfile) -> None:
    @flow
    async def test_flow() -> dbtRunnerResult:
        return await run_dbt_model(
            profiles_dir=profiles_dir,
            dbt_cli_profile=dbt_cli_profile_bare,
            summary_artifact_key='foo',
            create_summary_artifact=True,
        )
    with pytest.raises(Exception, match='dbt task result success: False with exception: None'):
        await test_flow()
    a: Artifact = await Artifact.get(key='foo')
    assert a
    assert a.type == 'markdown'
    assert a.data.startswith('# dbt run Task Summary')
    assert 'my_first_dbt_model' in a.data
    assert 'Unsuccessful Nodes' in a.data


@pytest.mark.usefixtures('dbt_runner_freshness_error')
async def test_run_dbt_source_freshness_creates_unsuccessful_artifact(profiles_dir: str, dbt_cli_profile_bare: DbtCliProfile) -> None:
    @flow
    async def test_flow() -> dbtRunnerResult:
        return await run_dbt_source_freshness(
            profiles_dir=profiles_dir,
            dbt_cli_profile=dbt_cli_profile_bare,
            summary_artifact_key='foo',
            create_summary_artifact=True,
        )
    with pytest.raises(Exception, match='dbt task result success: False with exception: None'):
        await test_flow()
    a: Artifact = await Artifact.get(key='foo')
    assert a
    assert a.type == 'markdown'
    assert a.data.startswith('# dbt source freshness Task Summary')
    assert 'my_first_dbt_model' in a.data
    assert 'Unsuccessful Nodes' in a.data


@pytest.mark.usefixtures('dbt_runner_failed_result')
async def test_run_dbt_model_throws_error(profiles_dir: str, dbt_cli_profile_bare: DbtCliProfile) -> None:
    @flow
    async def test_flow() -> dbtRunnerResult:
        return await run_dbt_model(
            profiles_dir=profiles_dir,
            dbt_cli_profile=dbt_cli_profile_bare,
            summary_artifact_key='foo',
            create_summary_artifact=True,
        )
    with pytest.raises(DbtUsageException, match="No such command 'weeeeeee'."):
        await test_flow()