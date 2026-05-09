import pytest
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
    def test_credentials_block_is_abstract(self) -> None: ...
    def test_credentials_block_implementation(self, caplog) -> None: ...

class TestNotificationBlock:
    def test_notification_block_is_abstract(self) -> None: ...
    def test_notification_block_implementation(self, caplog) -> None: ...

class JobRunIsRunning(PrefectException):
    def __init__(self, message: str) -> None: ...

class TestJobBlock:
    def test_job_block_is_abstract(self) -> None: ...
    def test_job_block_implementation(self, caplog) -> None: ...

class TestDatabaseBlock:
    async def test_database_block_implementation(
        self, caplog
    ) -> None: ...

class TestObjectStorageBlock:
    def test_object_storage_block_is_abstract(self) -> None: ...
    def test_object_storage_block_implementation(
        self, caplog, tmp_path
    ) -> None: ...

class TestSecretBlock:
    def test_secret_block_is_abstract(self) -> None: ...
    def test_secret_block_implementation(self, caplog) -> None: ...