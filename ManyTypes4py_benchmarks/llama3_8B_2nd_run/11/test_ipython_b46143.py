from pathlib import Path
import pytest
from IPython.core.error import UsageError
import kedro.ipython
from kedro.framework.project import pipelines
from kedro.ipython import _find_node, _format_node_inputs_text, _get_node_bound_arguments, _load_node, _prepare_function_body, _prepare_imports, _prepare_node_inputs, _resolve_project_path, load_ipython_extension, magic_load_node, reload_kedro
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline
from .conftest import dummy_function, dummy_function_with_loop, dummy_multiline_import_function, dummy_nested_function

class TestLoadKedroObjects:
    def test_ipython_load_entry_points(self, mocker: MockerFixture, fake_metadata: object, caplog: LogCaptureFixture) -> None:
        # ... (rest of the function)

    def test_ipython_lazy_load_pipeline(self, mocker: MockerFixture) -> None:
        # ... (rest of the function)

    def test_ipython_load_objects(self, mocker: MockerFixture, ipython: object) -> None:
        # ... (rest of the function)

    def test_ipython_load_objects_with_args(self, mocker: MockerFixture, fake_metadata: object, ipython: object) -> None:
        # ... (rest of the function)

class TestLoadIPythonExtension:
    def test_load_ipython_extension(self, ipython: object) -> None:
        # ... (rest of the function)

    def test_load_extension_missing_dependency(self, mocker: MockerFixture, ipython: object) -> None:
        # ... (rest of the function)

    def test_load_extension_not_in_kedro_project(self, mocker: MockerFixture, caplog: LogCaptureFixture) -> None:
        # ... (rest of the function)

    def test_load_extension_register_line_magic(self, mocker: MockerFixture, ipython: object) -> None:
        # ... (rest of the function)

class TestProjectPathResolution:
    def test_only_path_specified(self) -> None:
        # ... (rest of the function)

    def test_only_local_namespace_specified(self) -> None:
        # ... (rest of the function)

    def test_no_path_no_local_namespace_specified(self, mocker: MockerFixture) -> None:
        # ... (rest of the function)

    def test_project_path_unresolvable(self, mocker: MockerFixture, caplog: LogCaptureFixture) -> None:
        # ... (rest of the function)

    def test_project_path_update(self, caplog: LogCaptureFixture) -> None:
        # ... (rest of the function)

class TestLoadNodeMagic:
    def test_load_node_magic(self, mocker: MockerFixture, dummy_module_literal: object, dummy_pipelines: object) -> None:
        # ... (rest of the function)

    def test_load_node(self, mocker: MockerFixture, dummy_function_defintion: object, dummy_pipelines: object) -> None:
        # ... (rest of the function)

    def test_find_node(self, dummy_pipelines: object, dummy_node: object) -> None:
        # ... (rest of the function)

    def test_prepare_imports(self, mocker: MockerFixture) -> None:
        # ... (rest of the function)

    def test_prepare_imports_multiline(self, mocker: MockerFixture) -> None:
        # ... (rest of the function)

    def test_prepare_node_inputs(self, dummy_node: object) -> None:
        # ... (rest of the function)

    def test_prepare_node_inputs_when_input_is_empty(self, dummy_node_empty_input: object) -> None:
        # ... (rest of the function)

    def test_prepare_node_inputs_with_dict_input(self, dummy_node_dict_input: object) -> None:
        # ... (rest of the function)

    def test_prepare_node_inputs_with_variable_length_args(self, dummy_node_with_variable_length: object) -> None:
        # ... (rest of the function)

    def test_prepare_function_body(self, dummy_function_defintion: object) -> None:
        # ... (rest of the function)

    def test_load_node_magic_with_valid_arguments(self, mocker: MockerFixture, ipython: object) -> None:
        # ... (rest of the function)

    def test_load_node_with_invalid_arguments(self, mocker: MockerFixture, ipython: object) -> None:
        # ... (rest of the function)

    def test_load_node_with_jupyter(self, mocker: MockerFixture, ipython: object) -> None:
        # ... (rest of the function)

    def test_load_node_with_ipython(self, mocker: MockerFixture, ipython: object, run_env: str) -> None:
        # ... (rest of the function)

    def test_load_node_with_other(self, mocker: MockerFixture, ipython: object, run_env: str, rich_installed: bool) -> None:
        # ... (rest of the function)

class TestFormatNodeInputsText:
    def test_format_node_inputs_text_empty_input(self) -> None:
        # ... (rest of the function)

    def test_format_node_inputs_text_single_input(self) -> None:
        # ... (rest of the function)

    def test_format_node_inputs_text_multiple_inputs(self) -> None:
        # ... (rest of the function)

    def test_format_node_inputs_text_no_catalog_load(self) -> None:
        # ... (rest of the function)
