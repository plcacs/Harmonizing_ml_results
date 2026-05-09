from dataclasses import dataclass, field
import os
import time
import shutil
import argparse
from pathlib import Path
import sys
import json
from typing import Iterable, AnyStr, Any, Callable
import traceback
from collections import namedtuple
from ruamel.yaml import YAML
from filelock import FileLock
from snappy import compress
from pathos.multiprocessing import ProcessingPool as Pool
from eth_utils import encode_hex
from eth2spec.test import context
from eth2spec.test.exceptions import SkippedTest
from .gen_typing import TestProvider
from .settings import GENERATOR_MODE, MODE_MULTIPROCESSING, MODE_SINGLE_PROCESS, NUM_PROCESS, TIME_THRESHOLD_TO_PRINT
context.is_pytest = False

@dataclass
class Diagnostics(object):
    collected_test_count: int
    generated_test_count: int
    skipped_test_count: int
    test_identifiers: list[str]

TestCaseParams = namedtuple('TestCaseParams', ['test_case', 'case_dir', 'log_file', 'file_mode'])

def worker_function(item: TestCaseParams) -> Any:
    return generate_test_vector(*item)

def get_default_yaml() -> YAML:
    yaml = YAML(pure=True)
    yaml.default_flow_style = None

    def _represent_none(self, _):
        return self.represent_scalar('tag:yaml.org,2002:null', 'null')

    def _represent_str(self, data: str):
        if data.startswith('0x'):
            return self.represent_scalar('tag:yaml.org,2002:str', data, style="'")
        return self.represent_str(data)
    yaml.representer.add_representer(type(None), _represent_none)
    yaml.representer.add_representer(str, _represent_str)
    return yaml

def get_cfg_yaml() -> YAML:
    cfg_yaml = YAML(pure=True)
    cfg_yaml.default_flow_style = False

    def cfg_represent_bytes(self, data: bytes):
        return self.represent_int(encode_hex(data))
    cfg_yaml.representer.add_representer(bytes, cfg_represent_bytes)

    def cfg_represent_quoted_str(self, data: str):
        return self.represent_scalar(u'tag:yaml.org,2002:str', data, style="'")
    cfg_yaml.representer.add_representer(context.quoted_str, cfg_represent_quoted_str)
    return cfg_yaml

def validate_output_dir(path_str: str) -> Path:
    path = Path(path_str)
    if not path.exists():
        raise argparse.ArgumentTypeError('Output directory must exist')
    if not path.is_dir():
        raise argparse.ArgumentTypeError('Output path must lead to a directory')
    return path

def get_test_case_dir(test_case: Any, output_dir: Path) -> Path:
    return Path(output_dir) / Path(test_case.preset_name) / Path(test_case.fork_name) / Path(test_case.runner_name) / Path(test_case.handler_name) / Path(test_case.suite_name) / Path(test_case.case_name)

def get_test_identifier(test_case: Any) -> str:
    return '::'.join([test_case.preset_name, test_case.fork_name, test_case.runner_name, test_case.handler_name, test_case.suite_name, test_case.case_name])

def get_incomplete_tag_file(case_dir: Path) -> Path:
    return case_dir / 'INCOMPLETE'

def should_skip_case_dir(case_dir: Path, is_force: bool, diagnostics_obj: Diagnostics) -> tuple[bool, Diagnostics]:
    is_skip = False
    incomplete_tag_file = get_incomplete_tag_file(case_dir)
    if case_dir.exists():
        if not is_force and (not incomplete_tag_file.exists()):
            diagnostics_obj.skipped_test_count += 1
            print(f'Skipping already existing test: {case_dir}')
            is_skip = True
        else:
            print(f'Warning, output directory {case_dir} already exist,  old files will be deleted and it will generate test vector files with the latest version')
            shutil.rmtree(case_dir)
    return (is_skip, diagnostics_obj)

def run_generator(generator_name: str, test_providers: Iterable[TestProvider]) -> None:
    # ...

def generate_test_vector(test_case: Any, case_dir: Path, log_file: Path, file_mode: str) -> str:
    # ...

def write_result_into_diagnostics_obj(result: Any, diagnostics_obj: Diagnostics) -> None:
    # ...

def dump_yaml_fn(data: Any, name: str, file_mode: str, yaml_encoder: YAML) -> Callable[[Path], None]:
    # ...

def output_part(case_dir: Path, log_file: Path, out_kind: str, name: str, fn: Callable[[Path], None]) -> None:
    # ...

def execute_test(test_case: Any, case_dir: Path, meta: dict, log_file: Path, file_mode: str, cfg_yaml: YAML, yaml: YAML) -> tuple[bool, dict]:
    # ...

def dump_ssz_fn(data: Any, name: str, file_mode: str) -> Callable[[Path], None]:
    # ...
