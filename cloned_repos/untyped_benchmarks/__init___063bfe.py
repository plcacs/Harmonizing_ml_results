"""
Utilities and helpers for writing tests.
"""
from typing import Dict, Any, Optional, Union, Tuple, List
import torch
from torch.testing import assert_allclose
import pytest
from allennlp.common.testing.test_case import AllenNlpTestCase
from allennlp.common.testing.model_test_case import ModelTestCase
from allennlp.common.testing.distributed_test import run_distributed_test
from allennlp.modules.transformer import TransformerModule
from allennlp.training.metrics import Metric
_available_devices = ['cuda:0'] if torch.cuda.is_available() else ['cpu']

def multi_device(test_method):
    """
    Decorator that provides an argument `device` of type `str` to a test function.

    If you have a CUDA capable GPU available, device will be "cuda:0", otherwise the device will
    be "cpu".

    !!! Note
        If you have a CUDA capable GPU available, but you want to run the test using CPU only,
        just set the environment variable "CUDA_CAPABLE_DEVICES=''" before running pytest.
    """
    return pytest.mark.parametrize('device', _available_devices)(pytest.mark.gpu(test_method))

def requires_gpu(test_method):
    """
    Decorator to indicate that a test requires a GPU device.
    """
    return pytest.mark.gpu(pytest.mark.skipif(not torch.cuda.is_available(), reason='No CUDA device registered.')(test_method))

def requires_multi_gpu(test_method):
    """
    Decorator to indicate that a test requires multiple GPU devices.
    """
    return pytest.mark.gpu(pytest.mark.skipif(torch.cuda.device_count() < 2, reason='2 or more GPUs required.')(test_method))

def cpu_or_gpu(test_method):
    """
    Decorator to indicate that a test should run on both CPU and GPU
    """
    return pytest.mark.gpu(test_method)

def assert_metrics_values(metrics, desired_values, rtol=0.0001, atol=1e-05):
    for key in metrics:
        if isinstance(metrics[key], Dict) and isinstance(desired_values[key], Dict):
            for subkey in metrics[key]:
                assert_allclose(metrics[key][subkey], desired_values[key][subkey], rtol=rtol, atol=atol)
        else:
            assert_allclose(metrics[key], desired_values[key], rtol=rtol, atol=atol)

def global_distributed_metric(global_rank, world_size, gpu_id, metric, metric_kwargs, desired_values, exact=True, number_of_runs=1):
    kwargs = {}
    for argname in metric_kwargs:
        kwargs[argname] = metric_kwargs[argname][global_rank]
    for _ in range(number_of_runs):
        metric(**kwargs)
    metrics = metric.get_metric(False)
    if not isinstance(metrics, Dict) and (not isinstance(desired_values, Dict)):
        metrics = {'metric_value': metrics}
        desired_values = {'metric_value': desired_values}
    if isinstance(exact, bool):
        if exact:
            rtol = 0.0
            atol = 0.0
        else:
            rtol = 0.0001
            atol = 1e-05
    else:
        rtol = exact[0]
        atol = exact[1]
    assert_metrics_values(metrics, desired_values, rtol, atol)

def assert_equal_parameters(old_module, new_module, ignore_missing=False, mapping=None):
    """
    Tests if the parameters present in the `new_module` are equal to the ones in `old_module`.
    Note that any parameters present in the `old_module` that are not present in `new_module`
    are ignored.
    """
    mapping = mapping or {}
    old_parameters = dict(old_module.named_parameters())
    present_only_in_old = set(old_parameters.keys())
    for name, parameter in new_module.named_parameters():
        for key, val in mapping.items():
            name = name.replace(key, val)
        if ignore_missing:
            if name not in old_parameters:
                continue
        present_only_in_old.remove(name)
        assert torch.allclose(old_parameters[name], parameter)
    return present_only_in_old