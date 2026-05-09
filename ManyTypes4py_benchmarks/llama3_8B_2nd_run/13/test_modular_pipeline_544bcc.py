import pytest
from kedro.pipeline import node, pipeline
from kedro.pipeline.modular_pipeline import ModularPipelineError
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline

def constant_output() -> str:
    return 'output'

def identity(input1: str) -> str:
    return input1

def biconcat(input1: str, input2: str) -> str:
    return input1 + input2

def triconcat(input1: str, input2: str, input3: str) -> str:
    return input1 + input2 + input3

class TestPipelineHelper:
    def test_transform_dataset_names(self) -> None:
        # ... (rest of the code remains the same)

    def test_prefix_dataset_names(self) -> None:
        # ... (rest of the code remains the same)

    def test_prefixing_and_renaming(self) -> None:
        # ... (rest of the code remains the same)

    def test_prefix_exclude_free_inputs(self, inputs: str, outputs: str) -> None:
        # ... (rest of the code remains the same)

    def test_transform_params_prefix_and_parameters(self) -> None:
        # ... (rest of the code remains the same)

    def test_dataset_transcoding_mapping_base_name(self) -> None:
        # ... (rest of the code remains the same)

    def test_dataset_transcoding_mapping_full_dataset(self) -> None:
        # ... (rest of the code remains the same)

    def test_empty_input(self) -> None:
        # ... (rest of the code remains the same)

    def test_empty_output(self) -> None:
        # ... (rest of the code remains the same)

    @pytest.mark.parametrize('func, inputs, outputs, inputs_map, outputs_map, expected_missing, expected_suggestion', [
        # ... (rest of the code remains the same)
    ])
    def test_missing_dataset_name_no_suggestion(self, func: callable, inputs: str, outputs: str, inputs_map: dict, outputs_map: dict, expected_missing: list, expected_suggestion: list) -> None:
        # ... (rest of the code remains the same)

    @pytest.mark.parametrize('func, inputs, outputs, inputs_map, outputs_map, expected_missing, expected_suggestion', [
        # ... (rest of the code remains the same)
    ])
    def test_missing_dataset_with_suggestion(self, func: callable, inputs: str, outputs: str, inputs_map: dict, outputs_map: dict, expected_missing: list, expected_suggestion: list) -> None:
        # ... (rest of the code remains the same)

    def test_node_properties_preserved(self) -> None:
        # ... (rest of the code remains the same)

    def test_default_node_name_is_namespaced(self) -> None:
        # ... (rest of the code remains the same)

    def test_expose_intermediate_output(self) -> None:
        # ... (rest of the code remains the same)

    def test_parameters_left_intact_when_defined_as_str(self) -> None:
        # ... (rest of the code remains the same)

    @pytest.mark.parametrize('parameters', ['params:x', {'params:x'}, {'params:x': 'params:x'}])
    def test_parameters_left_intact_when_defined_as_(self, parameters: str) -> None:
        # ... (rest of the code remains the same)

    def test_parameters_updated_with_dict(self) -> None:
        # ... (rest of the code remains the same)

    def test_parameters_defined_with_params_prefix(self) -> None:
        # ... (rest of the code remains the same)

    def test_parameters_specified_under_inputs(self) -> None:
        # ... (rest of the code remains the same)

    def test_non_existent_parameters_mapped(self) -> None:
        # ... (rest of the code remains the same)

    def test_bad_inputs_mapping(self) -> None:
        # ... (rest of the code remains the same)

    def test_bad_outputs_mapping(self) -> None:
        # ... (rest of the code remains the same)

    def test_pipeline_always_copies(self) -> None:
        # ... (rest of the code remains the same)

    def test_pipeline_tags(self) -> None:
        # ... (rest of the code remains the same)
