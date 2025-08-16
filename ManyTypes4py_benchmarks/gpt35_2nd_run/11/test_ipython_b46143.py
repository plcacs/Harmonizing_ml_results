from pathlib import Path
import pytest
from IPython.core.error import UsageError
import kedro.ipython
from kedro.framework.project import pipelines
from kedro.ipython import _find_node, _format_node_inputs_text, _get_node_bound_arguments, _load_node, _prepare_function_body, _prepare_imports, _prepare_node_inputs, _resolve_project_path, load_ipython_extension, magic_load_node, reload_kedro
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline
from .conftest import dummy_function, dummy_function_with_loop, dummy_multiline_import_function, dummy_nested_function

class TestLoadKedroObjects:

    def test_ipython_load_entry_points(self, mocker: pytest.Mock, fake_metadata: pytest.Mock, caplog: pytest.Mock) -> None:
        ...

    def test_ipython_lazy_load_pipeline(self, mocker: pytest.Mock) -> None:
        ...

    def test_ipython_load_objects(self, mocker: pytest.Mock, ipython: pytest.Mock) -> None:
        ...

    def test_ipython_load_objects_with_args(self, mocker: pytest.Mock, fake_metadata: pytest.Mock, ipython: pytest.Mock) -> None:
        ...

class TestLoadIPythonExtension:

    def test_load_ipython_extension(self, ipython: pytest.Mock) -> None:
        ...

    def test_load_extension_missing_dependency(self, mocker: pytest.Mock) -> None:
        ...

    def test_load_extension_not_in_kedro_project(self, mocker: pytest.Mock, caplog: pytest.Mock) -> None:
        ...

    def test_load_extension_register_line_magic(self, mocker: pytest.Mock, ipython: pytest.Mock) -> None:
        ...

    def test_line_magic_with_valid_arguments(self, mocker: pytest.Mock, args: str, ipython: pytest.Mock) -> None:
        ...

    def test_line_magic_with_invalid_arguments(self, mocker: pytest.Mock, ipython: pytest.Mock) -> None:
        ...

    def test_ipython_kedro_extension_alias(self, mocker: pytest.Mock, ipython: pytest.Mock) -> None:
        ...

class TestProjectPathResolution:

    def test_only_path_specified(self) -> None:
        ...

    def test_only_local_namespace_specified(self) -> None:
        ...

    def test_no_path_no_local_namespace_specified(self, mocker: pytest.Mock) -> None:
        ...

    def test_project_path_unresolvable(self, mocker: pytest.Mock) -> None:
        ...

    def test_project_path_unresolvable_warning(self, mocker: pytest.Mock, caplog: pytest.Mock, ipython: pytest.Mock) -> None:
        ...

    def test_project_path_update(self, caplog: pytest.Mock) -> None:
        ...

class TestLoadNodeMagic:

    def test_load_node_magic(self, mocker: pytest.Mock, dummy_module_literal: str, dummy_pipelines: dict) -> None:
        ...

    def test_load_node(self, mocker: pytest.Mock, dummy_function_defintion: str, dummy_pipelines: dict) -> None:
        ...

    def test_find_node(self, dummy_pipelines: dict, dummy_node: str) -> None:
        ...

    def test_node_not_found(self, dummy_pipelines: dict) -> None:
        ...

    def test_prepare_imports(self, mocker: pytest.Mock) -> None:
        ...

    def test_prepare_imports_multiline(self, mocker: pytest.Mock) -> None:
        ...

    def test_prepare_node_inputs(self, dummy_node: str) -> None:
        ...

    def test_prepare_node_inputs_when_input_is_empty(self, dummy_node_empty_input: str) -> None:
        ...

    def test_prepare_node_inputs_with_dict_input(self, dummy_node_dict_input: str) -> None:
        ...

    def test_prepare_node_inputs_with_variable_length_args(self, dummy_node_with_variable_length: str) -> None:
        ...

    def test_prepare_function_body(self, dummy_function_defintion: str) -> None:
        ...

    def test_get_nested_function_body(self, dummy_nested_function_literal: str) -> None:
        ...

    def test_get_function_with_loop_body(self, dummy_function_with_loop_literal: str) -> None:
        ...

    def test_load_node_magic_with_valid_arguments(self, mocker: pytest.Mock, ipython: pytest.Mock) -> None:
        ...

    def test_load_node_with_invalid_arguments(self, mocker: pytest.Mock, ipython: pytest.Mock) -> None:
        ...

    def test_load_node_with_jupyter(self, mocker: pytest.Mock, ipython: pytest.Mock) -> None:
        ...

    def test_load_node_with_ipython(self, mocker: pytest.Mock, ipython: pytest.Mock, run_env: str) -> None:
        ...

    def test_load_node_with_other(self, mocker: pytest.Mock, ipython: pytest.Mock, run_env: str, rich_installed: bool) -> None:
        ...

class TestFormatNodeInputsText:

    def test_format_node_inputs_text_empty_input(self) -> None:
        ...

    def test_format_node_inputs_text_single_input(self) -> None:
        ...

    def test_format_node_inputs_text_multiple_inputs(self) -> None:
        ...

    def test_format_node_inputs_text_no_catalog_load(self) -> None:
        ...
