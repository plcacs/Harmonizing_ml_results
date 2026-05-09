from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from unittest.mock import MagicMock

import pytest
from IPython.core.interactiveshell import InteractiveShell
from IPython.core.magic import Magic_argv
from IPython.core.magic_arguments import ArgumentParser
from pytest_mock import MockFixture

from kedro.ipython import (
    _find_node,
    _format_node_inputs_text,
    _get_node_bound_arguments,
    _load_node,
    _prepare_function_body,
    _prepare_imports,
    _prepare_node_inputs,
    _resolve_project_path,
    load_ipython_extension,
    magic_load_node,
    reload_kedro,
)
from kedro.pipeline.modular_pipeline import ModularPipeline

class TestLoadKedroObjects:
    def test_ipython_load_entry_points(
        self,
        mocker: MockFixture,
        fake_metadata: Any,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        ...

    def test_ipython_lazy_load_pipeline(self, mocker: MockFixture) -> None:
        ...

    def test_ipython_load_objects(
        self,
        mocker: MockFixture,
        ipython: InteractiveShell,
    ) -> None:
        ...

    def test_ipython_load_objects_with_args(
        self,
        mocker: MockFixture,
        fake_metadata: Any,
        ipython: InteractiveShell,
    ) -> None:
        ...

class TestLoadIPythonExtension:
    def test_load_ipython_extension(self, ipython: InteractiveShell) -> None:
        ...

    def test_load_extension_missing_dependency(self, mocker: MockFixture) -> None:
        ...

    def test_load_extension_not_in_kedro_project(
        self,
        mocker: MockFixture,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        ...

    def test_load_extension_register_line_magic(
        self,
        mocker: MockFixture,
        ipython: InteractiveShell,
    ) -> None:
        ...

    @pytest.mark.parametrize('args', ['', '.', '. --env=base', '--env=base', '-e base', '. --env=base --params=key=val', '--conf-source=new_conf'])
    def test_line_magic_with_valid_arguments(
        self,
        mocker: MockFixture,
        args: str,
        ipython: InteractiveShell,
    ) -> None:
        ...

    def test_line_magic_with_invalid_arguments(
        self,
        mocker: MockFixture,
        ipython: InteractiveShell,
    ) -> None:
        ...

    def test_ipython_kedro_extension_alias(
        self,
        mocker: MockFixture,
        ipython: InteractiveShell,
    ) -> None:
        ...

class TestProjectPathResolution:
    def test_only_path_specified(self) -> Path:
        ...

    def test_only_local_namespace_specified(self) -> Path:
        ...

    def test_no_path_no_local_namespace_specified(
        self,
        mocker: MockFixture,
    ) -> Path:
        ...

    def test_project_path_unresolvable(self, mocker: MockFixture) -> None:
        ...

    def test_project_path_unresolvable_warning(
        self,
        mocker: MockFixture,
        caplog: pytest.LogCaptureFixture,
        ipython: InteractiveShell,
    ) -> None:
        ...

    def test_project_path_update(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        ...

class TestLoadNodeMagic:
    def test_load_node_magic(
        self,
        mocker: MockFixture,
        dummy_module_literal: str,
        dummy_pipelines: Dict[str, ModularPipeline],
    ) -> None:
        ...

    def test_load_node(
        self,
        mocker: MockFixture,
        dummy_function_defintion: str,
        dummy_pipelines: Dict[str, ModularPipeline],
    ) -> None:
        ...

    def test_find_node(
        self,
        dummy_pipelines: Dict[str, ModularPipeline],
        dummy_node: Any,
    ) -> Any:
        ...

    def test_node_not_found(
        self,
        dummy_pipelines: Dict[str, ModularPipeline],
    ) -> None:
        ...

    def test_prepare_imports(self, mocker: MockFixture) -> str:
        ...

    def test_prepare_imports_multiline(self, mocker: MockFixture) -> str:
        ...

    def test_prepare_imports_func_not_found(self, mocker: MockFixture) -> None:
        ...

    def test_prepare_node_inputs(self, dummy_node: Any) -> Dict[str, str]:
        ...

    def test_prepare_node_inputs_when_input_is_empty(
        self,
        dummy_node_empty_input: Any,
    ) -> Dict[str, str]:
        ...

    def test_prepare_node_inputs_with_dict_input(
        self,
        dummy_node_dict_input: Any,
    ) -> Dict[str, str]:
        ...

    def test_prepare_node_inputs_with_variable_length_args(
        self,
        dummy_node_with_variable_length: Any,
    ) -> Dict[str, str]:
        ...

    def test_prepare_function_body(self, dummy_function_defintion: str) -> str:
        ...

    @pytest.mark.skip('lambda function is not supported yet.')
    def test_get_lambda_function_body(self, lambda_node: Any) -> str:
        ...

    def test_get_nested_function_body(
        self,
        dummy_nested_function_literal: str,
    ) -> str:
        ...

    def test_get_function_with_loop_body(
        self,
        dummy_function_with_loop_literal: str,
    ) -> str:
        ...

    def test_load_node_magic_with_valid_arguments(
        self,
        mocker: MockFixture,
        ipython: InteractiveShell,
    ) -> None:
        ...

    def test_load_node_with_invalid_arguments(
        self,
        mocker: MockFixture,
        ipython: InteractiveShell,
    ) -> None:
        ...

    def test_load_node_with_jupyter(
        self,
        mocker: MockFixture,
        ipython: InteractiveShell,
    ) -> None:
        ...

    @pytest.mark.parametrize('run_env', ['ipython', 'vscode'])
    def test_load_node_with_ipython(
        self,
        mocker: MockFixture,
        ipython: InteractiveShell,
        run_env: str,
    ) -> None:
        ...

    @pytest.mark.parametrize('run_env, rich_installed', [('databricks', True), ('databricks', False), ('colab', True), ('colab', False), ('dummy', True), ('dummy', False)])
    def test_load_node_with_other(
        self,
        mocker: MockFixture,
        ipython: InteractiveShell,
        run_env: str,
        rich_installed: bool,
    ) -> None:
        ...

class TestFormatNodeInputsText:
    def test_format_node_inputs_text_empty_input(self) -> None:
        ...

    def test_format_node_inputs_text_single_input(self) -> str:
        ...

    def test_format_node_inputs_text_multiple_inputs(self) -> str:
        ...

    def test_format_node_inputs_text_no_catalog_load(self) -> None:
        ...