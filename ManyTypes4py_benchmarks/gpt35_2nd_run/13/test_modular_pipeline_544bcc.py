from typing import List, Dict, Union

def constant_output() -> str:
    return 'output'

def identity(input1: str) -> str:
    return input1

def biconcat(input1: Union[str, List[str]], input2: Union[str, List[str]]) -> str:
    return input1 + input2

def triconcat(input1: str, input2: str, input3: str) -> str:
    return input1 + input2 + input3

def test_transform_dataset_names() -> None:
    ...

def test_prefix_dataset_names() -> None:
    ...

def test_prefixing_and_renaming() -> None:
    ...

def test_prefix_exclude_free_inputs(inputs: Union[str, List[str], Dict[str, str]], outputs: Union[str, List[str], Dict[str, str]) -> None:
    ...

def test_transform_params_prefix_and_parameters() -> None:
    ...

def test_dataset_transcoding_mapping_base_name() -> None:
    ...

def test_dataset_transcoding_mapping_full_dataset() -> None:
    ...

def test_empty_input() -> None:
    ...

def test_empty_output() -> None:
    ...

def test_missing_dataset_name_no_suggestion(func, inputs, outputs, inputs_map, outputs_map, expected_missing) -> None:
    ...

def test_missing_dataset_with_suggestion(func, inputs, outputs, inputs_map, outputs_map, expected_missing, expected_suggestion) -> None:
    ...

def test_node_properties_preserved() -> None:
    ...

def test_default_node_name_is_namespaced() -> None:
    ...

def test_expose_intermediate_output() -> None:
    ...

def test_parameters_left_intact_when_defined_as_str() -> None:
    ...

def test_parameters_left_intact_when_defined_as_(parameters: Union[str, Dict[str, str]]) -> None:
    ...

def test_parameters_updated_with_dict() -> None:
    ...

def test_parameters_defined_with_params_prefix() -> None:
    ...

def test_parameters_specified_under_inputs() -> None:
    ...

def test_non_existent_parameters_mapped() -> None:
    ...

def test_bad_inputs_mapping() -> None:
    ...

def test_bad_outputs_mapping() -> None:
    ...

def test_pipeline_always_copies() -> None:
    ...

def test_pipeline_tags() -> None:
    ...
