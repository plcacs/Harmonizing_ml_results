from pathlib import Path
from typing import Any
from unittest import mock
import pytest
from homeassistant import backup_restore

def restore_result_file_content() -> Any:
    ...

def test_reading_the_instruction_contents(side_effect: Any, content: str, expected: Any) -> None:
    ...

def test_restoring_backup_that_does_not_exist() -> None:
    ...

def test_restoring_backup_when_instructions_can_not_be_read() -> None:
    ...

def test_restoring_backup_that_is_not_a_file() -> None:
    ...

def test_aborting_for_older_versions() -> None:
    ...

def test_removal_of_current_configuration_when_restoring(restore_backup_content: backup_restore.RestoreBackupFileContent, expected_removed_files: tuple, expected_removed_directories: tuple, expected_copied_files: tuple, expected_copied_trees: tuple) -> None:
    ...

def test_extracting_the_contents_of_a_backup_file() -> None:
    ...

def test_remove_backup_file_after_restore(remove_after_restore: bool, unlink_calls: int) -> None:
    ...

def test_pw_to_key(password: str, expected: bytes) -> None:
    ...

def test_pw_to_key_none() -> None:
    ...
