import logging
import re
import subprocess
import sys
import textwrap
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union, List, Callable, Generator
from unittest.mock import create_autospec, Mock

import pytest
import toml
import yaml
from omegaconf import OmegaConf

from kedro import __version__ as kedro_version
from kedro.config import AbstractConfigLoader, OmegaConfigLoader
from kedro.framework.cli.utils import _split_params
from kedro.framework.context import KedroContext
from kedro.framework.project import (
    LOGGING,
    ValidationError,
    Validator,
    _HasSharedParentClassValidator,
    _ProjectSettings,
)
from kedro.framework.session import KedroSession
from kedro.framework.session.session import KedroSessionError
from kedro.framework.session.store import BaseSessionStore
from kedro.utils import _has_rich_handler

_FAKE_PROJECT_NAME: str = "fake_project"
_FAKE_PIPELINE_NAME: str = "fake_pipeline"


class BadStore:
    """
    Store class that doesn't subclass `BaseSessionStore`, for testing only.
    """


class BadConfigLoader:
    """
    ConfigLoader class that doesn't subclass `AbstractConfigLoader`, for testing only.
    """


ATTRS_ATTRIBUTE: str = "__attrs_attrs__"

NEW_TYPING: bool = sys.version_info[:3] >= (3, 7, 0)  # PEP 560


def create_attrs_autospec(spec: type, spec_set: bool = True) -> Any:
    """Creates a mock of an attr class (creates mocks recursively on all attributes).
    https://github.com/python-attrs/attrs/issues/462#issuecomment-1134656377

    :param spec: the spec to mock
    :param spec_set: if True, AttributeError will be raised if an attribute that is not in the spec is set.
    """

    if not hasattr(spec, ATTRS_ATTRIBUTE):
        raise TypeError(f"{spec!r} is not an attrs class")
    mock: Any = create_autospec(spec, spec_set=spec_set)
    for attribute in getattr(spec, ATTRS_ATTRIBUTE):
        attribute_type: Any = attribute.type
        if NEW_TYPING:
            # A[T] does not get a copy of __dict__ from A(Generic[T]) anymore, use __origin__ to get it
            while hasattr(attribute_type, "__origin__"):
                attribute_type = attribute_type.__origin__
        if hasattr(attribute_type, ATTRS_ATTRIBUTE):
            mock_attribute: Any = create_attrs_autospec(attribute_type, spec_set)
        else:
            mock_attribute = create_autospec(attribute_type, spec_set=spec_set)
        object.__setattr__(mock, attribute.name, mock_attribute)
    return mock


@pytest.fixture
def mock_runner(mocker: Any) -> Mock:
    mock_runner: Mock = mocker.patch(
        "kedro.runner.sequential_runner.SequentialRunner",
        autospec=True,
    )
    mock_runner.__name__ = "MockRunner"
    return mock_runner


@pytest.fixture
def mock_thread_runner(mocker: Any) -> Mock:
    mock_runner: Mock = mocker.patch(
        "kedro.runner.thread_runner.ThreadRunner",
        autospec=True,
    )
    mock_runner.__name__ = "MockThreadRunner`"
    return mock_runner


@pytest.fixture
def mock_context_class(mocker: Any) -> Mock:
    mock_cls: Any = create_attrs_autospec(KedroContext)
    return mocker.patch(
        "kedro.framework.context.KedroContext", autospec=True, return_value=mock_cls
    )


def _mock_imported_settings_paths(mocker: Any, mock_settings: _ProjectSettings) -> _ProjectSettings:
    for path in [
        "kedro.framework.project.settings",
        "kedro.framework.session.session.settings",
    ]:
        mocker.patch(path, mock_settings)
    return mock_settings


@pytest.fixture
def mock_settings(mocker: Any) -> _ProjectSettings:
    return _mock_imported_settings_paths(mocker, _ProjectSettings())


@pytest.fixture
def mock_settings_context_class(mocker: Any, mock_context_class: Mock) -> _ProjectSettings:
    class MockSettings(_ProjectSettings):
        # dynaconf automatically deleted some attribute when the class is MagicMock
        _CONTEXT_CLASS: Validator = Validator(
            "CONTEXT_CLASS", default=lambda *_: mock_context_class
        )

    return _mock_imported_settings_paths(mocker, MockSettings())


@pytest.fixture
def mock_settings_custom_context_class(mocker: Any) -> _ProjectSettings:
    class MyContext(KedroContext):
        pass

    class MockSettings(_ProjectSettings):
        _CONTEXT_CLASS: Validator = Validator("CONTEXT_CLASS", default=lambda *_: MyContext)

    return _mock_imported_settings_paths(mocker, MockSettings())


@pytest.fixture
def mock_settings_custom_config_loader_class(mocker: Any) -> _ProjectSettings:
    class MyConfigLoader(AbstractConfigLoader):
        pass

    class MockSettings(_ProjectSettings):
        _CONFIG_LOADER_CLASS: _HasSharedParentClassValidator = _HasSharedParentClassValidator(
            "CONFIG_LOADER_CLASS", default=lambda *_: MyConfigLoader
        )

    return _mock_imported_settings_paths(mocker, MockSettings())


@pytest.fixture
def mock_settings_omega_config_loader_class(mocker: Any) -> _ProjectSettings:
    class MockSettings(_ProjectSettings):
        _CONFIG_LOADER_CLASS: _HasSharedParentClassValidator = _HasSharedParentClassValidator(
            "CONFIG_LOADER_CLASS", default=lambda *_: OmegaConfigLoader
        )

    return _mock_imported_settings_paths(mocker, MockSettings())


@pytest.fixture
def mock_settings_config_loader_args(mocker: Any) -> _ProjectSettings:
    class MockSettings(_ProjectSettings):
        _CONFIG_LOADER_ARGS: Validator = Validator(
            "CONFIG_LOADER_ARGS",
            default={"config_patterns": {"spark": ["spark/*"]}},
        )

    return _mock_imported_settings_paths(mocker, MockSettings())


@pytest.fixture
def mock_settings_config_loader_args_env(mocker: Any) -> _ProjectSettings:
    class MockSettings(_ProjectSettings):
        _CONFIG_LOADER_ARGS: Validator = Validator(
            "CONFIG_LOADER_ARGS",
            default={"base_env": "something_new"},
        )

    return _mock_imported_settings_paths(mocker, MockSettings())


@pytest.fixture
def mock_settings_file_bad_config_loader_class(tmpdir: Any) -> Path:
    mock_settings_file: Path = Path(tmpdir.join("mock_settings_file.py"))
    mock_settings_file.write_text(
        textwrap.dedent(
            f"""
            from {__name__} import BadConfigLoader
            CONFIG_LOADER_CLASS = BadConfigLoader
            """
        )
    )
    return mock_settings_file


@pytest.fixture
def mock_settings_file_bad_session_store_class(tmpdir: Any) -> Path:
    mock_settings_file: Path = Path(tmpdir.join("mock_settings_file.py"))
    mock_settings_file.write_text(
        textwrap.dedent(
            f"""
            from {__name__} import BadStore
            SESSION_STORE_CLASS = BadStore
            """
        )
    )
    return mock_settings_file


@pytest.fixture
def mock_settings_bad_session_store_args(mocker: Any) -> _ProjectSettings:
    class MockSettings(_ProjectSettings):
        _SESSION_STORE_ARGS: Validator = Validator(
            "SESSION_STORE_ARGS", default={"wrong_arg": "O_o"}
        )

    return _mock_imported_settings_paths(mocker, MockSettings())


@pytest.fixture
def mock_settings_uncaught_session_store_exception(mocker: Any) -> Mock:
    class MockSettings(_ProjectSettings):
        _SESSION_STORE_ARGS: Validator = Validator("SESSION_STORE_ARGS", default={"path": "path"})

    _mock_imported_settings_paths(mocker, MockSettings())
    return mocker.patch.object(
        BaseSessionStore, "__init__", side_effect=Exception("Fake")
    )


@pytest.fixture
def fake_session_id(mocker: Any) -> str:
    session_id: str = "fake_session_id"
    mocker.patch(
        "kedro.framework.session.session.generate_timestamp", return_value=session_id
    )
    return session_id


@pytest.fixture
def fake_project(tmp_path: Path, mock_package_name: str) -> Path:
    fake_project_dir: Path = Path(tmp_path) / "fake_project"
    (fake_project_dir / "src").mkdir(parents=True)

    pyproject_toml_path: Path = fake_project_dir / "pyproject.toml"
    payload: Dict[str, Any] = {
        "tool": {
            "kedro": {
                "kedro_init_version": kedro_version,
                "project_name": _FAKE_PROJECT_NAME,
                "package_name": mock_package_name,
            }
        }
    }
    toml_str: str = toml.dumps(payload)
    pyproject_toml_path.write_text(toml_str, encoding="utf-8")

    (fake_project_dir / "conf" / "base").mkdir(parents=True)
    (fake_project_dir / "conf" / "local").mkdir()
    return fake_project_dir


@pytest.fixture
def fake_username(mocker: Any) -> str:
    username: str = "user1"
    mocker.patch(
        "kedro.framework.session.session.getpass.getuser", return_value=username
    )
    return username


class FakeException(Exception):
    """Fake exception class for testing purposes"""


SESSION_LOGGER_NAME: str = "kedro.framework.session.session"
STORE_LOGGER_NAME: str = "kedro.framework.session.store"


class TestKedroSession:
    @pytest.mark.usefixtures("mock_settings_context_class")
    @pytest.mark.parametrize("env", [None, "env1"])
    @pytest.mark.parametrize("extra_params", [None, {"key": "val"}])
    def test_create(
        self,
        fake_project: Path,
        mock_context_class: Mock,
        fake_session_id: str,
        mocker: Any,
        env: Optional[str],
        extra_params: Optional[Dict[str, Any]],
        fake_username: str,
    ) -> None:
        mock_click_ctx: Any = mocker.patch("click.get_current_context").return_value
        mocker.patch("sys.argv", ["kedro", "run", "--params=x"])
        session: KedroSession = KedroSession.create(fake_project, env=env, extra_params=extra_params)

        expected_cli_data: Dict[str, Any] = {
            "args": mock_click_ctx.args,
            "params": mock_click_ctx.params,
            "command_name": mock_click_ctx.command.name,
            "command_path": "kedro run --params=x",
        }
        expected_store: Dict[str, Any] = {
            "project_path": fake_project,
            "session_id": fake_session_id,
            "cli": expected_cli_data,
        }
        if env:
            expected_store["env"] = env
        if extra_params:
            expected_store["extra_params"] = extra_params

        expected_store["username"] = fake_username

        mocker.patch(
            "kedro_telemetry.plugin._check_for_telemetry_consent",
            return_value=False,
        )

        assert session.store == expected_store
        assert session.load_context() is mock_context_class.return_value
        assert isinstance(session._get_config_loader(), OmegaConfigLoader)

    @pytest.mark.usefixtures("mock_settings")
    def test_create_multiple_sessions(self, fake_project: Path) -> None:
        with KedroSession.create(fake_project):
            with KedroSession.create(fake_project):
                pass

    @pytest.mark.usefixtures("mock_settings_context_class")
    def test_create_no_env_extra_params(
        self,
        fake_project: Path,
        mock_context_class: Mock,
        fake_session_id: str,
        mocker: Any,
        fake_username: str,
    ) -> None:
        mock_click_ctx: Any = mocker.patch("click.get_current_context").return_value
        mocker.patch("sys.argv", ["kedro", "run", "--params=x"])
        session: KedroSession = KedroSession.create(fake_project)

        expected_cli_data: Dict[str, Any] = {
            "args": mock_click_ctx.args,
            "params": mock_click_ctx.params,
            "command_name": mock_click_ctx.command.name,
            "command_path": "kedro run --params=x",
        }
        expected_store: Dict[str, Any] = {
            "project_path": fake_project,
            "session_id": fake_session_id,
            "cli": expected_cli_data,
        }

        expected_store["username"] = fake_username

        mocker.patch(
            "kedro_telemetry.plugin._check_for_telemetry_consent",
            return_value=False,
        )

        assert session.store == expected_store
        assert session.load_context() is mock_context_class.return_value
        assert isinstance(session._get_config_loader(), OmegaConfigLoader)

    @pytest.mark.usefixtures("mock_settings")
    def test_load_context_with_envvar(self, fake_project: Path, monkeypatch: Any, mocker: Any) -> None:
        monkeypatch.setenv("KEDRO_ENV", "my_fake_env")

        session: KedroSession = KedroSession.create(fake_project)
        result: KedroContext = session.load_context()

        assert isinstance(result, KedroContext)
        assert result.__class__.__name__ == "KedroContext"
        assert result.env == "my_fake_env"

    @pytest.mark.usefixtures("mock_settings")
    def test_load_config_loader_with_envvar(self, fake_project: Path, monkeypatch: Any) -> None:
        monkeypatch.setenv("KEDRO_ENV", "my_fake_env")

        session: KedroSession = KedroSession.create(fake_project)
        result: AbstractConfigLoader = session._get_config_loader()

        assert isinstance(result, OmegaConfigLoader)
        assert result.__class__.__name__ == "OmegaConfigLoader"
        assert result.env == "my_fake_env"

    @pytest.mark.usefixtures("mock_settings_custom_context_class")
    def test_load_context_custom_context_class(self, fake_project: Path) -> None:
        session: KedroSession = KedroSession.create(fake_project)
        result: KedroContext = session.load_context()

        assert isinstance(result, KedroContext)
        assert result.__class__.__name__ == "MyContext"

    @pytest.mark.usefixtures("mock_settings_custom_config_loader_class")
    def test_load_config_loader_custom_config_loader_class(self, fake_project: Path) -> None:
        session: KedroSession = KedroSession.create(fake_project)
        result: AbstractConfigLoader = session._get_config_loader()

        assert isinstance(result, AbstractConfigLoader)
        assert result.__class__.__name__ == "MyConfigLoader"

    @pytest.mark.usefixtures("mock_settings_config_loader_args")
    def test_load_config_loader_args(self, fake_project: Path, mocker: Any) -> None:
        session: KedroSession = KedroSession.create(fake_project)
        result: OmegaConfigLoader = session._get_config_loader()

        assert isinstance(result, OmegaConfigLoader)
        assert result.config_patterns["catalog"] == [
            "catalog*",
            "catalog*/**",
            "**/catalog*",
        ]
        assert result.config_patterns["spark"] == ["spark/*"]
        mocker.patch(
            "kedro.config.OmegaConfigLoader.__getitem__",
            return_value=["spark/*"],
        )
        assert result["spark"] == ["spark/*"]

    @pytest.mark.usefixtures("mock_settings_config_loader_args")
    def test_config_loader_args_no_env_overwrites_env(self, fake_project: Path, mocker: Any) -> None:
        session: KedroSession = KedroSession.create(fake_project)
        result: OmegaConfigLoader = session._get_config_loader()

        assert isinstance(result, OmegaConfigLoader)
        assert result.base_env == ""
        assert result.default_run_env == ""

    @pytest.mark.usefixtures("mock_settings_config_loader_args_env")
    def test_config_loader_args_overwrite_env(self, fake_project: Path, mocker: Any) -> None:
        session: KedroSession = KedroSession.create(fake_project)
        result: OmegaConfigLoader = session._get_config_loader()

        assert isinstance(result, OmegaConfigLoader)
        assert result.base_env == "something_new"
        assert result.default_run_env == ""

    def test_broken_config_loader(self, mock_settings_file_bad_config_loader_class: Path) -> None:
        pattern: str = (
            "Invalid value 'tests.framework.session.test_session.BadConfigLoader' received "
            "for setting 'CONFIG_LOADER_CLASS'. "
            "It must be a subclass of 'kedro.config.abstract_config.AbstractConfigLoader'."
        )
        mock_settings: _ProjectSettings = _ProjectSettings(
            settings_file=str(mock_settings_file_bad_config_loader_class)
        )
        with pytest.raises(ValidationError, match=re.escape(pattern)):
            assert mock_settings.CONFIG_LOADER_CLASS

    def test_logging_is_not_reconfigure(self, fake_project: Path, caplog: Any, mocker: Any) -> None:
        caplog.set_level(logging.DEBUG, logger="kedro")

        mock_logging: Any = mocker.patch.object(LOGGING, "configure")
        session: KedroSession = KedroSession.create(fake_project)
        session.close()

        mock_logging.assert_not_called()

    @pytest.mark.usefixtures("mock_settings_context_class")
