#!/usr/bin/env python
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from IPython.core.error import UsageError

import kedro.ipython
from kedro.framework.project import pipelines
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
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline

from .conftest import (
    dummy_function,
    dummy_function_with_loop,
    dummy_multiline_import_function,
    dummy_nested_function,
)


class TestLoadKedroObjects:
    def test_ipython_load_entry_points(
        self, mocker: Any, fake_metadata: Any, caplog: Any
    ) -> None:
        mock_line_magic = mocker.MagicMock()
        mock_line_magic_name: str = "abc"
        mock_line_magic.__name__ = mock_line_magic_name
        mock_line_magic.__qualname__ = mock_line_magic_name  # Required by IPython

        mocker.patch("kedro.ipython.load_entry_points", return_value=[mock_line_magic])
        expected_message: str = f"Registered line magic '{mock_line_magic_name}'"

        reload_kedro(fake_metadata.project_path)

        log_messages: List[str] = [record.getMessage() for record in caplog.records]
        assert expected_message in log_messages

    def test_ipython_lazy_load_pipeline(self, mocker: Any) -> None:
        pipelines.configure("dummy_pipeline")  # Setup the pipelines

        my_pipelines: Dict[str, Any] = {"ds": modular_pipeline([])}

        def my_register_pipeline() -> Dict[str, Any]:
            return my_pipelines

        mocker.patch.object(
            pipelines,
            "_get_pipelines_registry_callable",
            return_value=my_register_pipeline,
        )
        reload_kedro()

        assert pipelines._content == {}  # Check if it is lazy loaded
        pipelines._load_data()  # Trigger data load
        assert pipelines._content == my_pipelines

    def test_ipython_load_objects(self, mocker: Any, ipython: Any) -> None:
        mock_session_create = mocker.patch("kedro.ipython.KedroSession.create")
        pipelines.configure("dummy_pipeline")  # Setup the pipelines

        my_pipelines: Dict[str, Any] = {"ds": modular_pipeline([])}

        def my_register_pipeline() -> Dict[str, Any]:
            return my_pipelines

        mocker.patch.object(
            pipelines,
            "_get_pipelines_registry_callable",
            return_value=my_register_pipeline,
        )
        ipython_spy = mocker.spy(ipython, "push")

        reload_kedro()

        mock_session_create.assert_called_once_with(
            None,
            env=None,
            extra_params=None,
            conf_source=None,
        )
        _, kwargs = ipython_spy.call_args_list[0]
        variables: Dict[str, Any] = kwargs["variables"]

        assert variables["context"] == mock_session_create().load_context()
        assert variables["catalog"] == mock_session_create().load_context().catalog
        assert variables["session"] == mock_session_create()
        assert variables["pipelines"] == my_pipelines

    def test_ipython_load_objects_with_args(
        self, mocker: Any, fake_metadata: Any, ipython: Any
    ) -> None:
        mock_session_create = mocker.patch("kedro.ipython.KedroSession.create")
        pipelines.configure("dummy_pipeline")  # Setup the pipelines

        my_pipelines: Dict[str, Any] = {"ds": modular_pipeline([])}

        def my_register_pipeline() -> Dict[str, Any]:
            return my_pipelines

        mocker.patch.object(
            pipelines,
            "_get_pipelines_registry_callable",
            return_value=my_register_pipeline,
        )
        ipython_spy = mocker.spy(ipython, "push")
        dummy_env: str = "env"
        dummy_dict: Dict[str, str] = {"key": "value"}
        dummy_conf_source: str = "conf/"

        reload_kedro(
            fake_metadata.project_path, "env", {"key": "value"}, conf_source="conf/"
        )

        mock_session_create.assert_called_once_with(
            fake_metadata.project_path,
            env=dummy_env,
            extra_params=dummy_dict,
            conf_source=dummy_conf_source,
        )
        _, kwargs = ipython_spy.call_args_list[0]
        variables: Dict[str, Any] = kwargs["variables"]

        assert variables["context"] == mock_session_create().load_context()
        assert variables["catalog"] == mock_session_create().load_context().catalog
        assert variables["session"] == mock_session_create()
        assert variables["pipelines"] == my_pipelines


class TestLoadIPythonExtension:
    def test_load_ipython_extension(self, ipython: Any) -> None:
        ipython.magic("load_ext kedro.ipython")

    def test_load_extension_missing_dependency(self, mocker: Any) -> None:
        mocker.patch("kedro.ipython.reload_kedro", side_effect=ImportError)
        mocker.patch(
            "kedro.ipython._find_kedro_project",
            return_value=mocker.Mock(),
        )
        mocker.patch("IPython.core.magic.register_line_magic")
        mocker.patch("IPython.core.magic_arguments.magic_arguments")
        mocker.patch("IPython.core.magic_arguments.argument")
        mock_ipython = mocker.patch("IPython.get_ipython")

        with pytest.raises(ImportError):
            load_ipython_extension(mocker.Mock())

        assert not mock_ipython().called
        assert not mock_ipython().push.called

    def test_load_extension_not_in_kedro_project(self, mocker: Any, caplog: Any) -> None:
        mocker.patch("kedro.ipython._find_kedro_project", return_value=None)
        mocker.patch("IPython.core.magic.register_line_magic")
        mocker.patch("IPython.core.magic_arguments.magic_arguments")
        mocker.patch("IPython.core.magic_arguments.argument")
        mock_ipython = mocker.patch("IPython.get_ipython")

        load_ipython_extension(mocker.Mock())

        assert not mock_ipython().called
        assert not mock_ipython().push.called

        log_messages: List[str] = [record.getMessage() for record in caplog.records]
        expected_message: str = (
            "Kedro extension was registered but couldn't find a Kedro project. "
            "Make sure you run '%reload_kedro <project_root>'."
        )
        assert expected_message in log_messages

    def test_load_extension_register_line_magic(self, mocker: Any, ipython: Any) -> None:
        mocker.patch("kedro.ipython._find_kedro_project")
        mock_reload_kedro = mocker.patch("kedro.ipython.reload_kedro")
        load_ipython_extension(ipython)
        mock_reload_kedro.assert_called_once()

        # Calling the line magic to check if the line magic is available
        ipython.magic("reload_kedro")
        assert mock_reload_kedro.call_count == 2

    @pytest.mark.parametrize(
        "args",
        [
            "",
            ".",
            ". --env=base",
            "--env=base",
            "-e base",
            ". --env=base --params=key=val",
            "--conf-source=new_conf",
        ],
    )
    def test_line_magic_with_valid_arguments(self, mocker: Any, args: str, ipython: Any) -> None:
        mocker.patch("kedro.ipython._find_kedro_project")
        mocker.patch("kedro.ipython.reload_kedro")

        ipython.magic(f"reload_kedro {args}")

    def test_line_magic_with_invalid_arguments(self, mocker: Any, ipython: Any) -> None:
        mocker.patch("kedro.ipython._find_kedro_project")
        mocker.patch("kedro.ipython.reload_kedro")
        load_ipython_extension(ipython)

        with pytest.raises(
            UsageError, match=r"unrecognized arguments: --invalid_arg=dummy"
        ):
            ipython.magic("reload_kedro --invalid_arg=dummy")

    def test_ipython_kedro_extension_alias(self, mocker: Any, ipython: Any) -> None:
        mock_ipython_extension = mocker.patch(
            "kedro.ipython.load_ipython_extension", autospec=True
        )
        # Ensure that `kedro` is not loaded initially
        assert "kedro" not in ipython.extension_manager.loaded
        ipython.magic("load_ext kedro")
        mock_ipython_extension.assert_called_once_with(ipython)
        # Ensure that `kedro` extension has been loaded
        assert "kedro" in ipython.extension_manager.loaded


class TestProjectPathResolution:
    def test_only_path_specified(self) -> None:
        result: Optional[Path] = _resolve_project_path(path="/test")
        expected: Path = Path("/test").resolve()
        assert result == expected

    def test_only_local_namespace_specified(self) -> None:
        class MockKedroContext:
            # A dummy stand-in for KedroContext sufficient for this test
            project_path: Path = Path("/test").resolve()

        result: Optional[Path] = _resolve_project_path(local_namespace={"context": MockKedroContext()})
        expected: Path = Path("/test").resolve()
        assert result == expected

    def test_no_path_no_local_namespace_specified(self, mocker: Any) -> None:
        mocker.patch(
            "kedro.ipython._find_kedro_project", return_value=Path("/test").resolve()
        )
        result: Optional[Path] = _resolve_project_path()
        expected: Path = Path("/test").resolve()
        assert result == expected

    def test_project_path_unresolvable(self, mocker: Any) -> None:
        mocker.patch("kedro.ipython._find_kedro_project", return_value=None)
        result: Optional[Path] = _resolve_project_path()
        expected: None = None
        assert result == expected

    def test_project_path_unresolvable_warning(self, mocker: Any, caplog: Any, ipython: Any) -> None:
        mocker.patch("kedro.ipython._find_kedro_project", return_value=None)
        ipython.magic("reload_ext kedro.ipython")
        log_messages: List[str] = [record.getMessage() for record in caplog.records]
        expected_message: str = (
            "Kedro extension was registered but couldn't find a Kedro project. "
            "Make sure you run '%reload_kedro <project_root>'."
        )
        assert expected_message in log_messages

    def test_project_path_update(self, caplog: Any) -> None:
        class MockKedroContext:
            # A dummy stand-in for KedroContext sufficient for this test
            project_path: Path = Path("/test").resolve()

        local_namespace: Dict[str, Any] = {"context": MockKedroContext()}
        updated_path: Path = Path("/updated_path").resolve()
        _resolve_project_path(path=updated_path, local_namespace=local_namespace)

        log_messages: List[str] = [record.getMessage() for record in caplog.records]
        expected_message: str = f"Updating path to Kedro project: {updated_path}..."
        assert expected_message in log_messages


class TestLoadNodeMagic:
    def test_load_node_magic(
        self, mocker: Any, dummy_module_literal: Any, dummy_pipelines: Any
    ) -> None:
        # Reimport `pipelines` from `kedro.framework.project` to ensure that
        # it was not removed by prior tests.
        from kedro.framework.project import pipelines

        # Mocking setup
        mock_jupyter_console = mocker.MagicMock()
        mocker.patch("ipylab.JupyterFrontEnd", mock_jupyter_console)
        mock_pipeline_values = dummy_pipelines.values()
        mocker.patch.object(pipelines, "values", return_value=mock_pipeline_values)

        node_to_load: str = "dummy_node"
        magic_load_node(node_to_load)

    def test_load_node(
        self, mocker: Any, dummy_function_defintion: str, dummy_pipelines: Any
    ) -> None:
        # wraps all the other functions
        mock_pipeline_values = dummy_pipelines.values()
        mocker.patch.object(pipelines, "values", return_value=mock_pipeline_values)

        node_inputs: str = (
            "# Prepare necessary inputs for debugging\n"
            "# All debugging inputs must be defined in your project catalog\n"
            'dummy_input = catalog.load("dummy_input")\n'
            'my_input = catalog.load("extra_input")'
        )

        node_imports: str = (
            "import logging  # noqa\n"
            "from logging import config  # noqa\n"
            "import logging as dummy_logging  # noqa\n"
            "import logging.config  # noqa Dummy import"
        )

        node_func_definition: str = dummy_function_defintion
        node_func_call: str = "dummy_function(dummy_input, my_input)"

        expected_cells: List[str] = [
            node_inputs,
            node_imports,
            node_func_definition,
            node_func_call,
        ]

        node_to_load: str = "dummy_node"
        cells_list: List[str] = _load_node(node_to_load, pipelines)

        for cell, expected_cell in zip(cells_list, expected_cells):
            assert cell == expected_cell

    def test_find_node(self, dummy_pipelines: Any, dummy_node: Any) -> None:
        node_to_find: str = "dummy_node"
        result: Any = _find_node(node_to_find, dummy_pipelines)
        assert result == dummy_node

    def test_node_not_found(self, dummy_pipelines: Any) -> None:
        node_to_find: str = "not_a_node"
        dummy_registered_pipelines: Any = dummy_pipelines
        with pytest.raises(ValueError) as excinfo:
            _find_node(node_to_find, dummy_registered_pipelines)

        assert (
            f"Node with name='{node_to_find}' not found in any pipelines. Remember to specify the node name, not the node function."
            in str(excinfo.value)
        )

    def test_prepare_imports(self, mocker: Any) -> None:
        func_imports: str = (
            "import logging  # noqa\n"
            "from logging import config  # noqa\n"
            "import logging as dummy_logging  # noqa\n"
            "import logging.config  # noqa Dummy import"
        )

        result: str = _prepare_imports(dummy_function)
        assert result == func_imports

    def test_prepare_imports_multiline(self, mocker: Any) -> None:
        func_imports: str = """from logging import (
INFO,
DEBUG,
WARN,
ERROR,
)"""

        result: str = _prepare_imports(dummy_multiline_import_function)
        assert result == func_imports

    def test_prepare_imports_func_not_found(self, mocker: Any) -> None:
        mocker.patch("inspect.getsourcefile", return_value=None)

        with pytest.raises(FileNotFoundError) as excinfo:
            _prepare_imports(dummy_function)

        assert f"Could not find {dummy_function.__name__}" in str(excinfo.value)

    def test_prepare_node_inputs(self, dummy_node: Any) -> None:
        expected: Dict[str, str] = {"dummy_input": "dummy_input", "my_input": "extra_input"}

        node_bound_arguments: Any = _get_node_bound_arguments(dummy_node)
        result: Dict[str, str] = _prepare_node_inputs(node_bound_arguments)
        assert result == expected

    def test_prepare_node_inputs_when_input_is_empty(
        self, dummy_node_empty_input: Any
    ) -> None:
        expected: Dict[str, str] = {"dummy_input": "", "my_input": ""}

        node_bound_arguments: Any = _get_node_bound_arguments(dummy_node_empty_input)
        result: Dict[str, str] = _prepare_node_inputs(node_bound_arguments)
        assert result == expected

    def test_prepare_node_inputs_with_dict_input(
        self, dummy_node_dict_input: Any
    ) -> None:
        expected: Dict[str, str] = {"dummy_input": "dummy_input", "my_input": "extra_input"}

        node_bound_arguments: Any = _get_node_bound_arguments(dummy_node_dict_input)
        result: Dict[str, str] = _prepare_node_inputs(node_bound_arguments)
        assert result == expected

    def test_prepare_node_inputs_with_variable_length_args(
        self, dummy_node_with_variable_length: Any
    ) -> None:
        expected: Dict[str, str] = {
            "dummy_input": "dummy_input",
            "my_input": "extra_input",
            "first": "first",
            "second": "second",
        }

        node_bound_arguments: Any = _get_node_bound_arguments(
            dummy_node_with_variable_length
        )
        result: Dict[str, str] = _prepare_node_inputs(node_bound_arguments)
        assert result == expected

    def test_prepare_function_body(self, dummy_function_defintion: str) -> None:
        result: str = _prepare_function_body(dummy_function)
        assert result == dummy_function_defintion

    @pytest.mark.skip("lambda function is not supported yet.")
    def test_get_lambda_function_body(self, lambda_node: Any) -> None:
        result: str = _prepare_function_body(lambda_node.func)
        assert result == "func=lambda x: x\n"

    def test_get_nested_function_body(self, dummy_nested_function_literal: str) -> None:
        result: str = _prepare_function_body(dummy_nested_function)
        assert result == dummy_nested_function_literal

    def test_get_function_with_loop_body(self, dummy_function_with_loop_literal: str) -> None:
        result: str = _prepare_function_body(dummy_function_with_loop)
        assert result == dummy_function_with_loop_literal

    def test_load_node_magic_with_valid_arguments(self, mocker: Any, ipython: Any) -> None:
        mocker.patch("kedro.ipython._find_kedro_project")
        mocker.patch("kedro.ipython._load_node")
        ipython.magic("load_node dummy_node")

    def test_load_node_with_invalid_arguments(self, mocker: Any, ipython: Any) -> None:
        mocker.patch("kedro.ipython._find_kedro_project")
        mocker.patch("kedro.ipython._load_node")
        load_ipython_extension(ipython)

        with pytest.raises(
            UsageError, match=r"unrecognized arguments: --invalid_arg=dummy_node"
        ):
            ipython.magic("load_node --invalid_arg=dummy_node")

    def test_load_node_with_jupyter(self, mocker: Any, ipython: Any) -> None:
        mocker.patch("kedro.ipython._find_kedro_project")
        mocker.patch("kedro.ipython._load_node", return_value=["cell1", "cell2"])
        mocker.patch("kedro.ipython._guess_run_environment", return_value="jupyter")
        spy = mocker.spy(kedro.ipython, "_create_cell_with_text")
        call = mocker.call

        load_ipython_extension(ipython)
        ipython.magic("load_node dummy_node")
        calls = [call("cell1", is_jupyter=True), call("cell2", is_jupyter=True)]
        spy.assert_has_calls(calls)

    @pytest.mark.parametrize("run_env", ["ipython", "vscode"])
    def test_load_node_with_ipython(self, mocker: Any, ipython: Any, run_env: str) -> None:
        mocker.patch("kedro.ipython._find_kedro_project")
        mocker.patch("kedro.ipython._load_node", return_value=["cell1", "cell2"])
        mocker.patch("kedro.ipython._guess_run_environment", return_value=run_env)
        spy = mocker.spy(kedro.ipython, "_create_cell_with_text")

        load_ipython_extension(ipython)
        ipython.magic("load_node dummy_node")
        spy.assert_called_once()

    @pytest.mark.parametrize(
        "run_env, rich_installed",
        [
            ("databricks", True),
            ("databricks", False),
            ("colab", True),
            ("colab", False),
            ("dummy", True),
            ("dummy", False),
        ],
    )
    def test_load_node_with_other(
        self, mocker: Any, ipython: Any, run_env: str, rich_installed: bool
    ) -> None:
        mocker.patch("kedro.ipython._find_kedro_project")
        mocker.patch("kedro.ipython.RICH_INSTALLED", rich_installed)
        mocker.patch("kedro.ipython._load_node", return_value=["cell1", "cell2"])
        mocker.patch("kedro.ipython._guess_run_environment", return_value=run_env)
        spy = mocker.spy(kedro.ipython, "_print_cells")

        load_ipython_extension(ipython)
        ipython.magic("load_node dummy_node")
        spy.assert_called_once()


class TestFormatNodeInputsText:
    def test_format_node_inputs_text_empty_input(self) -> None:
        # Test with empty input_params_dict
        input_params_dict: Dict[str, str] = {}
        expected_output: Optional[str] = None
        assert _format_node_inputs_text(input_params_dict) == expected_output

    def test_format_node_inputs_text_single_input(self) -> None:
        # Test with a single input
        input_params_dict: Dict[str, str] = {"input1": "dataset1"}
        expected_output: str = (
            "# Prepare necessary inputs for debugging\n"
            "# All debugging inputs must be defined in your project catalog\n"
            'input1 = catalog.load("dataset1")'
        )
        assert _format_node_inputs_text(input_params_dict) == expected_output

    def test_format_node_inputs_text_multiple_inputs(self) -> None:
        # Test with multiple inputs
        input_params_dict: Dict[str, str] = {
            "input1": "dataset1",
            "input2": "dataset2",
            "input3": "dataset3",
        }
        expected_output: str = (
            "# Prepare necessary inputs for debugging\n"
            "# All debugging inputs must be defined in your project catalog\n"
            'input1 = catalog.load("dataset1")\n'
            'input2 = catalog.load("dataset2")\n'
            'input3 = catalog.load("dataset3")'
        )
        assert _format_node_inputs_text(input_params_dict) == expected_output

    def test_format_node_inputs_text_no_catalog_load(self) -> None:
        # Test with no catalog.load() statements if input_params_dict is None
        assert _format_node_inputs_text(None) is None
