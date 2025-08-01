import pytest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from prefect.blocks.abstract import (
    CredentialsBlock,
    DatabaseBlock,
    JobBlock,
    JobRun,
    NotificationBlock,
    ObjectStorageBlock,
    SecretBlock,
)
from prefect.exceptions import PrefectException


class TestCredentialsBlock:

    def test_credentials_block_is_abstract(self) -> None:
        with pytest.raises(TypeError, match="Can't instantiate abstract class CredentialsBlock"):
            CredentialsBlock()

    def test_credentials_block_implementation(self, caplog: pytest.LogCaptureFixture) -> None:

        class ACredentialsBlock(CredentialsBlock):

            def get_client(self) -> str:
                self.logger.info('Got client.')
                return 'client'

        a_credentials_block = ACredentialsBlock()
        assert a_credentials_block.get_client() == 'client'
        assert hasattr(a_credentials_block, 'logger')
        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.name == 'prefect.ACredentialsBlock'
        assert record.msg == 'Got client.'


class TestNotificationBlock:

    def test_notification_block_is_abstract(self) -> None:
        with pytest.raises(TypeError, match="Can't instantiate abstract class NotificationBlock"):
            NotificationBlock()

    def test_notification_block_implementation(self, caplog: pytest.LogCaptureFixture) -> None:

        class ANotificationBlock(NotificationBlock):

            def notify(self, body: str, subject: Optional[str] = None) -> None:
                self.logger.info(f'Notification sent with {body} {subject}.')

        a_notification_block = ANotificationBlock()
        a_notification_block.notify('body', 'subject')
        assert hasattr(a_notification_block, 'logger')
        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.name == 'prefect.ANotificationBlock'
        assert record.msg == 'Notification sent with body subject.'


class JobRunIsRunning(PrefectException):
    """Raised when a job run is still running."""


class TestJobBlock:

    def test_job_block_is_abstract(self) -> None:
        with pytest.raises(TypeError, match="Can't instantiate abstract class JobBlock"):
            JobBlock()

    def test_job_block_implementation(self, caplog: pytest.LogCaptureFixture) -> None:

        class AJobRun(JobRun):
            _status: str

            def __init__(self) -> None:
                super().__init__()
                self._status = 'running'

            @property
            def status(self) -> str:
                return self._status

            @status.setter
            def status(self, value: str) -> None:
                self._status = value

            def wait_for_completion(self) -> None:
                self.status = 'completed'
                self.logger.info('Job run completed.')

            def fetch_result(self) -> str:
                if self.status != 'completed':
                    raise JobRunIsRunning('Job run is still running.')
                return 'results'

        class AJobBlock(JobBlock):

            def trigger(self) -> AJobRun:
                self.logger.info('Job run triggered.')
                return AJobRun()

        a_job_block = AJobBlock()
        a_job_run = a_job_block.trigger()
        with pytest.raises(JobRunIsRunning, match='Job run is still running.'):
            a_job_run.fetch_result()
        assert a_job_run.wait_for_completion() is None
        assert a_job_run.fetch_result() == 'results'
        assert hasattr(a_job_block, 'logger')
        assert hasattr(a_job_run, 'logger')
        assert len(caplog.records) == 2
        record_1 = caplog.records[0]
        assert record_1.name == 'prefect.AJobBlock'
        assert record_1.msg == 'Job run triggered.'
        record_2 = caplog.records[1]
        assert record_2.name == 'prefect.AJobRun'
        assert record_2.msg == 'Job run completed.'


class TestDatabaseBlock:

    def test_database_block_is_abstract(self) -> None:
        with pytest.raises(TypeError, match="Can't instantiate abstract class DatabaseBlock"):
            DatabaseBlock()

    async def test_database_block_implementation(self, caplog: pytest.LogCaptureFixture) -> None:

        class ADatabaseBlock(DatabaseBlock):
            _results: Tuple[Tuple[str, int, bool], ...]
            _engine: Optional[bool]

            def __init__(self) -> None:
                super().__init__()
                self._results = tuple(zip(['apple', 'banana', 'cherry'], [1, 2, 3], [True, False, True]))
                self._engine = None

            def fetch_one(self, operation: str, parameters: Optional[Dict[str, Any]] = None, **execution_kwargs: Any) -> Tuple[str, int, bool]:
                self.logger.info(f'Fetching one result using {parameters}.')
                return self._results[0]

            def fetch_many(
                self,
                operation: str,
                parameters: Optional[Dict[str, Any]] = None,
                size: Optional[int] = None,
                **execution_kwargs: Any
            ) -> Tuple[Tuple[str, int, bool], ...]:
                self.logger.info(f'Fetching {size} results using {parameters}.')
                return self._results[:size]

            def fetch_all(
                self,
                operation: str,
                parameters: Optional[Dict[str, Any]] = None,
                **execution_kwargs: Any
            ) -> Tuple[Tuple[str, int, bool], ...]:
                self.logger.info(f'Fetching all results using {parameters}.')
                return self._results

            def execute(
                self,
                operation: str,
                parameters: Optional[Dict[str, Any]] = None,
                **execution_kwargs: Any
            ) -> None:
                self.logger.info(f'Executing operation using {parameters}.')

            def execute_many(
                self,
                operation: str,
                seq_of_parameters: List[Dict[str, Any]],
                **execution_kwargs: Any
            ) -> None:
                self.logger.info(f'Executing many operations using {seq_of_parameters}.')

            def __enter__(self) -> 'ADatabaseBlock':
                self._engine = True
                return self

            def __exit__(self, exc_type: Optional[type], exc_val: Optional[BaseException], exc_tb: Optional[Any]) -> None:
                self._engine = None

        a_database_block = ADatabaseBlock()
        parameters: Dict[str, str] = {'a': 'b'}
        assert a_database_block.fetch_one('SELECT * FROM table', parameters=parameters) == ('apple', 1, True)
        assert a_database_block.fetch_many('SELECT * FROM table', size=2, parameters=parameters) == (
            ('apple', 1, True),
            ('banana', 2, False),
        )
        assert a_database_block.fetch_all('SELECT * FROM table', parameters=parameters) == (
            ('apple', 1, True),
            ('banana', 2, False),
            ('cherry', 3, True),
        )
        assert a_database_block.execute('INSERT INTO table VALUES (1, 2, 3)', parameters=parameters) is None
        assert a_database_block.execute_many(
            'INSERT INTO table VALUES (1, 2, 3)', seq_of_parameters=[parameters, parameters], parameters=parameters
        ) is None
        records = caplog.records
        for record in records:
            assert record.name == 'prefect.ADatabaseBlock'
        assert records[0].message == "Fetching one result using {'a': 'b'}."
        assert records[1].message == "Fetching 2 results using {'a': 'b'}."
        assert records[2].message == "Fetching all results using {'a': 'b'}."
        assert records[3].message == "Executing operation using {'a': 'b'}."
        assert records[4].message == "Executing many operations using [{'a': 'b'}, {'a': 'b'}]."
        with a_database_block as db:
            assert db._engine is True
        assert a_database_block._engine is None
        match = 'ADatabaseBlock does not support async context management.'
        with pytest.raises(NotImplementedError, match=match):
            async with a_database_block:
                pass


class TestObjectStorageBlock:

    def test_object_storage_block_is_abstract(self) -> None:
        with pytest.raises(TypeError, match="Can't instantiate abstract class ObjectStorageBlock"):
            ObjectStorageBlock()

    def test_object_storage_block_implementation(
        self, caplog: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:

        class AObjectStorageBlock(ObjectStorageBlock):
            _storage: Dict[str, str]

            def __init__(self) -> None:
                super().__init__()
                self._storage = {}

            def download_object_to_path(
                self, from_path: str, to_path: Path, **download_kwargs: Any
            ) -> Path:
                with open(to_path, 'w') as f:
                    f.write(self._storage[from_path])
                return to_path

            def download_object_to_file_object(
                self, from_path: str, to_file_object: Any, **download_kwargs: Any
            ) -> Any:
                to_file_object.write(self._storage[from_path])
                return to_file_object

            def download_folder_to_path(
                self, from_folder: str, to_folder: Path, **download_kwargs: Any
            ) -> None:
                self.logger.info(f'downloaded from {from_folder} to {to_folder}')

            def upload_from_path(
                self, from_path: Path, to_path: str, **upload_kwargs: Any
            ) -> str:
                with open(from_path, 'r') as f:
                    self._storage[to_path] = f.read()
                return to_path

            def upload_from_file_object(
                self, from_file_object: Any, to_path: str, **upload_kwargs: Any
            ) -> str:
                self._storage[to_path] = from_file_object.read()
                return to_path

            def upload_from_folder(
                self, from_folder: Path, to_folder: str, **upload_kwargs: Any
            ) -> None:
                self.logger.info(f'uploaded from {from_folder} to {to_folder}')

        a_object_storage_block = AObjectStorageBlock()
        a_file_path: Path = tmp_path / 'a_file.txt'
        a_file_path.write_text('hello')
        a_object_storage_block.upload_from_path(from_path=a_file_path, to_path='uploaded_from_path.txt')
        assert a_object_storage_block._storage['uploaded_from_path.txt'] == 'hello'
        with open(a_file_path, 'r') as f:
            a_object_storage_block.upload_from_file_object(from_file_object=f, to_path='uploaded_from_file_object.txt')
        assert a_object_storage_block._storage['uploaded_from_file_object.txt'] == 'hello'
        a_object_storage_block.upload_from_folder(from_folder=tmp_path, to_folder='uploaded_from_folder')
        assert caplog.records[0].message == f'uploaded from {tmp_path} to uploaded_from_folder'
        a_object_storage_block.download_object_to_path(
            from_path='uploaded_from_path.txt', to_path=tmp_path / 'downloaded_to_path.txt'
        )
        assert (tmp_path / 'downloaded_to_path.txt').exists()
        with open(tmp_path / 'downloaded_to_file_object.txt', 'w') as f:
            a_object_storage_block.download_object_to_file_object(
                from_path='uploaded_from_file_object.txt', to_file_object=f
            )
        with open(tmp_path / 'downloaded_to_file_object.txt', 'r') as f:
            assert f.read() == 'hello'
        a_object_storage_block.download_folder_to_path(
            from_folder='uploaded_from_folder', to_folder=Path('downloaded_to_folder')
        )
        assert caplog.records[1].message == 'downloaded from uploaded_from_folder to downloaded_to_folder'


class TestSecretBlock:

    def test_secret_block_is_abstract(self) -> None:
        with pytest.raises(TypeError, match="Can't instantiate abstract class SecretBlock"):
            SecretBlock()

    def test_secret_block_implementation(self, caplog: pytest.LogCaptureFixture) -> None:

        class ASecretBlock(SecretBlock):
            _secrets: Dict[str, str]

            def __init__(self, secret_name: str) -> None:
                super().__init__(secret_name=secret_name)
                self._secrets = {}

            def read_secret(self) -> str:
                if self.secret_name not in self._secrets:
                    raise KeyError('Secret does not exist')
                return self._secrets[self.secret_name]

            def write_secret(self, secret_value: str) -> None:
                if self.secret_name in self._secrets:
                    raise ValueError('Secret already exists')
                self._secrets[self.secret_name] = secret_value

            def update_secret(self, secret_value: str) -> None:
                self._secrets[self.secret_name] = secret_value

            def delete_secret(self) -> None:
                del self._secrets[self.secret_name]

        a_secret_block = ASecretBlock(secret_name='secret_name')
        a_secret_block.write_secret('hello')
        assert a_secret_block.read_secret() == 'hello'
        with pytest.raises(ValueError, match='Secret already exists'):
            a_secret_block.write_secret('hello again')
        a_secret_block.update_secret('hello again')
        assert a_secret_block.read_secret() == 'hello again'
        a_secret_block.delete_secret()
        with pytest.raises(KeyError, match='Secret does not exist'):
            a_secret_block.read_secret()
