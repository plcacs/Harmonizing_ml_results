from dataclasses import dataclass, field
import os
import time
import shutil
import argparse
from pathlib import Path
import sys
import json
from typing import Iterable, AnyStr, Any, Callable, Tuple, Union, List, Dict
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

# Constants (assumed to be defined in settings)
from .settings import GENERATOR_MODE, MODE_MULTIPROCESSING, MODE_SINGLE_PROCESS, NUM_PROCESS, TIME_THRESHOLD_TO_PRINT

context.is_pytest = False

@dataclass
class Diagnostics:
    collected_test_count: int = 0
    generated_test_count: int = 0
    skipped_test_count: int = 0
    test_identifiers: List[str] = field(default_factory=list)

TestCaseParams = namedtuple('TestCaseParams', ['test_case', 'case_dir', 'log_file', 'file_mode'])

def worker_function(item: TestCaseParams) -> Union[int, str]:
    return generate_test_vector(*item)

def get_default_yaml() -> YAML:
    yaml = YAML(pure=True)
    yaml.default_flow_style = None

    def _represent_none(self: YAML, _: Any) -> Any:
        return self.represent_scalar('tag:yaml.org,2002:null', 'null')

    def _represent_str(self: YAML, data: str) -> Any:
        if data.startswith('0x'):
            return self.represent_scalar('tag:yaml.org,2002:str', data, style="'")
        return self.represent_str(data)
    yaml.representer.add_representer(type(None), _represent_none)
    yaml.representer.add_representer(str, _represent_str)
    return yaml

def get_cfg_yaml() -> YAML:
    cfg_yaml = YAML(pure=True)
    cfg_yaml.default_flow_style = False

    def cfg_represent_bytes(self: YAML, data: bytes) -> Any:
        return self.represent_int(encode_hex(data))
    cfg_yaml.representer.add_representer(bytes, cfg_represent_bytes)

    def cfg_represent_quoted_str(self: YAML, data: context.quoted_str) -> Any:
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

def should_skip_case_dir(case_dir: Path, is_force: bool, diagnostics_obj: Diagnostics) -> Tuple[bool, Diagnostics]:
    is_skip: bool = False
    incomplete_tag_file: Path = get_incomplete_tag_file(case_dir)
    if case_dir.exists():
        if not is_force and (not incomplete_tag_file.exists()):
            diagnostics_obj.skipped_test_count += 1
            print(f'Skipping already existing test: {case_dir}')
            is_skip = True
        else:
            print(f'Warning, output directory {case_dir} already exist,  old files will be deleted and it will generate test vector files with the latest version')
            shutil.rmtree(case_dir)
    return (is_skip, diagnostics_obj)

def run_generator(generator_name: str, test_providers: List[TestProvider]) -> None:
    parser = argparse.ArgumentParser(prog='gen-' + generator_name, description=f'Generate YAML test suite files for {generator_name}')
    parser.add_argument('-o', '--output-dir', dest='output_dir', required=True, type=validate_output_dir, help='directory into which the generated YAML files will be dumped')
    parser.add_argument('-f', '--force', action='store_true', default=False, help='if set re-generate and overwrite test files if they already exist')
    parser.add_argument('--preset-list', dest='preset_list', nargs='*', type=str, required=False, help='specify presets to run with. Allows all if no preset names are specified.')
    parser.add_argument('--fork-list', dest='fork_list', nargs='*', type=str, required=False, help='specify forks to run with. Allows all if no fork names are specified.')
    parser.add_argument('--modcheck', action='store_true', default=False, help='check generator modules, do not run any tests.')
    args = parser.parse_args()
    if args.modcheck:
        return
    output_dir: Path = args.output_dir
    if not args.force:
        file_mode: str = 'x'
    else:
        file_mode = 'w'
    log_file: Path = Path(output_dir) / 'testgen_error_log.txt'
    print(f'Generating tests into {output_dir}')
    print(f'Error log file: {log_file}')
    presets: List[str] = args.preset_list if args.preset_list is not None else []
    if len(presets) != 0:
        print(f'Filtering test-generator runs to only include presets: {", ".join(presets)}')
    forks: List[str] = args.fork_list if args.fork_list is not None else []
    if len(forks) != 0:
        print(f'Filtering test-generator runs to only include forks: {", ".join(forks)}')
    diagnostics_obj: Diagnostics = Diagnostics()
    provider_start: float = time.time()
    if GENERATOR_MODE == MODE_MULTIPROCESSING:
        all_test_case_params: List[TestCaseParams] = []
    for tprov in test_providers:
        tprov.prepare()
        for test_case in tprov.make_cases():
            if len(presets) != 0 and test_case.preset_name not in presets:
                continue
            if len(forks) != 0 and test_case.fork_name not in forks:
                continue
            case_dir: Path = get_test_case_dir(test_case, output_dir)
            print(f'Collected test at: {case_dir}')
            diagnostics_obj.collected_test_count += 1
            is_skip, diagnostics_obj = should_skip_case_dir(case_dir, args.force, diagnostics_obj)
            if is_skip:
                continue
            if GENERATOR_MODE == MODE_SINGLE_PROCESS:
                result: Union[int, str] = generate_test_vector(test_case, case_dir, log_file, file_mode)
                write_result_into_diagnostics_obj(result, diagnostics_obj)
            elif GENERATOR_MODE == MODE_MULTIPROCESSING:
                item: TestCaseParams = TestCaseParams(test_case, case_dir, log_file, file_mode)
                all_test_case_params.append(item)
    if GENERATOR_MODE == MODE_MULTIPROCESSING:
        with Pool(processes=NUM_PROCESS) as pool:
            results: List[Union[int, str]] = pool.map(worker_function, iter(all_test_case_params))
        for result in results:
            write_result_into_diagnostics_obj(result, diagnostics_obj)
    provider_end: float = time.time()
    span: float = round(provider_end - provider_start, 2)
    summary_message: str = f'completed generation of {generator_name} with {diagnostics_obj.generated_test_count} tests'
    summary_message += f' ({diagnostics_obj.skipped_test_count} skipped tests)'
    if span > TIME_THRESHOLD_TO_PRINT:
        summary_message += f' in {span} seconds'
    print(summary_message)
    diagnostics_output: Dict[str, Any] = {
        'collected_test_count': diagnostics_obj.collected_test_count,
        'generated_test_count': diagnostics_obj.generated_test_count,
        'skipped_test_count': diagnostics_obj.skipped_test_count,
        'test_identifiers': diagnostics_obj.test_identifiers,
        'durations': [f'{span} seconds']
    }
    diagnostics_path: Path = Path(os.path.join(output_dir, 'diagnostics_obj.json'))
    diagnostics_lock: FileLock = FileLock(os.path.join(output_dir, 'diagnostics_obj.json.lock'))
    with diagnostics_lock:
        diagnostics_path.touch(exist_ok=True)
        if os.path.getsize(diagnostics_path) == 0:
            with open(diagnostics_path, 'w+') as f:
                json.dump(diagnostics_output, f)
        else:
            with open(diagnostics_path, 'r+') as f:
                existing_diagnostics: Dict[str, Any] = json.load(f)
                for k, v in diagnostics_output.items():
                    existing_diagnostics[k] += v
            with open(diagnostics_path, 'w+') as f:
                json.dump(existing_diagnostics, f)
        print(f'wrote diagnostics_obj to {diagnostics_path}')

def generate_test_vector(test_case: Any, case_dir: Path, log_file: Path, file_mode: str) -> Union[int, str]:
    cfg_yaml: YAML = get_cfg_yaml()
    yaml: YAML = get_default_yaml()
    written_part: bool = False
    print(f'Generating test: {case_dir}')
    test_start: float = time.time()
    incomplete_tag_file: Path = get_incomplete_tag_file(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    with incomplete_tag_file.open('w') as f:
        f.write('\n')
    result: Union[int, str, None] = None
    try:
        meta: Dict[str, Any] = dict()
        try:
            written_part, meta = execute_test(test_case, case_dir, meta, log_file, file_mode, cfg_yaml, yaml)
        except SkippedTest as e:
            result = 0
            print(e)
            shutil.rmtree(case_dir)
            return result
        if len(meta) != 0:
            written_part = True
            output_part(case_dir, log_file, 'data', 'meta', dump_yaml_fn(meta, 'meta', file_mode, yaml))
    except Exception as e:
        result = -1
        error_message: str = f'[ERROR] failed to generate vector(s) for test {case_dir}: {e}'
        with log_file.open('a+') as f:
            f.write(error_message)
            traceback.print_exc(file=f)
            f.write('\n')
        print(error_message)
        traceback.print_exc()
    else:
        if not written_part:
            print(f'[Error] test case {case_dir} did not produce any written_part')
            shutil.rmtree(case_dir)
            result = -1
        else:
            result = get_test_identifier(test_case)
            os.remove(incomplete_tag_file)
    test_end: float = time.time()
    span: float = round(test_end - test_start, 2)
    if span > TIME_THRESHOLD_TO_PRINT:
        print(f'- generated in {span} seconds')
    return result

def write_result_into_diagnostics_obj(result: Union[int, str, None], diagnostics_obj: Diagnostics) -> None:
    if result == -1:
        pass
    elif result == 0:
        diagnostics_obj.skipped_test_count += 1
    elif result is not None:
        diagnostics_obj.generated_test_count += 1
        diagnostics_obj.test_identifiers.append(result)
    else:
        raise Exception(f'Unexpected result: {result}')

def dump_yaml_fn(data: Any, name: str, file_mode: str, yaml_encoder: YAML) -> Callable[[Path], None]:
    def dump(case_path: Path) -> None:
        out_path: Path = case_path / Path(name + '.yaml')
        with out_path.open(file_mode) as f:
            yaml_encoder.dump(data, f)
            f.close()
    return dump

def output_part(case_dir: Path, log_file: Path, out_kind: str, name: str, fn: Callable[[Path], None]) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    try:
        fn(case_dir)
    except (IOError, ValueError) as e:
        error_message: str = f'[Error] error when dumping test "{case_dir}", part "{name}", kind "{out_kind}": {e}'
        with log_file.open('a+') as f:
            f.write(error_message)
            traceback.print_exc(file=f)
            f.write('\n')
        print(error_message)
        sys.exit(error_message)

def execute_test(test_case: Any, case_dir: Path, meta: Dict[str, Any], log_file: Path, file_mode: str, cfg_yaml: YAML, yaml: YAML) -> Tuple[bool, Dict[str, Any]]:
    result = test_case.case_fn()
    written_part: bool = False
    for name, out_kind, data in result:
        written_part = True
        if out_kind == 'meta':
            meta[name] = data
        elif out_kind == 'cfg':
            output_part(case_dir, log_file, out_kind, name, dump_yaml_fn(data, name, file_mode, cfg_yaml))
        elif out_kind == 'data':
            output_part(case_dir, log_file, out_kind, name, dump_yaml_fn(data, name, file_mode, yaml))
        elif out_kind == 'ssz':
            output_part(case_dir, log_file, out_kind, name, dump_ssz_fn(data, name, file_mode))
        else:
            raise ValueError('Unknown out_kind %s' % out_kind)
    return (written_part, meta)

def dump_ssz_fn(data: bytes, name: str, file_mode: str) -> Callable[[Path], None]:
    def dump(case_path: Path) -> None:
        out_path: Path = case_path / Path(name + '.ssz_snappy')
        compressed: bytes = compress(data)
        with out_path.open(file_mode + 'b') as f:
            f.write(compressed)
    return dump