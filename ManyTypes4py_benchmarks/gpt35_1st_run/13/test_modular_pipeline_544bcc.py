from typing import List, Dict, Union

def constant_output() -> str:
    return 'output'

def identity(input1: str) -> str:
    return input1

def biconcat(input1: Union[str, List[str]], input2: Union[str, List[str]]) -> str:
    return input1 + input2

def triconcat(input1: str, input2: str, input3: str) -> str:
    return input1 + input2 + input3

class TestPipelineHelper:

    def test_transform_dataset_names(self) -> None:
        ...

    def test_prefix_dataset_names(self) -> None:
        ...

    def test_prefixing_and_renaming(self) -> None:
        ...

    def test_prefix_exclude_free_inputs(self, inputs: Union[str, List[str], Dict[str, str]], outputs: Union[str, List[str], Dict[str, str]]) -> None:
        ...

    def test_transform_params_prefix_and_parameters(self) -> None:
        ...

    def test_dataset_transcoding_mapping_base_name(self) -> None:
        ...

    def test_dataset_transcoding_mapping_full_dataset(self) -> None:
        ...

    def test_empty_input(self) -> None:
        ...

    def test_empty_output(self) -> None:
        ...

    def test_missing_dataset_name_no_suggestion(self, func, inputs, outputs, inputs_map, outputs_map, expected_missing) -> None:
        ...

    def test_missing_dataset_with_suggestion(self, func, inputs, outputs, inputs_map, outputs_map, expected_missing, expected_suggestion) -> None:
        ...

    def test_node_properties_preserved(self) -> None:
        ...

    def test_default_node_name_is_namespaced(self) -> None:
        ...

    def test_expose_intermediate_output(self) -> None:
        ...

    def test_parameters_left_intact_when_defined_as_str(self) -> None:
        ...

    def test_parameters_left_intact_when_defined_as_(self, parameters: Union[str, Dict[str, str]]) -> None:
        ...

    def test_parameters_updated_with_dict(self) -> None:
        ...

    def test_parameters_defined_with_params_prefix(self) -> None:
        ...

    def test_parameters_specified_under_inputs(self) -> None:
        ...

    def test_non_existent_parameters_mapped(self) -> None:
        ...

    def test_bad_inputs_mapping(self) -> None:
        ...

    def test_bad_outputs_mapping(self) -> None:
        ...

    def test_pipeline_always_copies(self) -> None:
        ...

    def test_pipeline_tags(self) -> None:
        ...
