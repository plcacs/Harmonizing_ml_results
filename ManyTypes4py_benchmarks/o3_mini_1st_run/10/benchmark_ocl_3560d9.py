import tempfile
from pathlib import Path
from typing import Any, Dict
import yaml
from kedro.config import OmegaConfigLoader

def _generate_catalog(
    start_range: int,
    end_range: int,
    is_local: bool = False,
    is_versioned: bool = False,
    add_interpolation: bool = False
) -> Dict[str, Dict[str, Any]]:
    catalog: Dict[str, Dict[str, Any]] = {}
    for i in range(start_range, end_range + 1):
        catalog[f'dataset_{i}'] = {
            'type': 'pandas.CSVDataset',
            'filepath': f'data{i}{("_local" if is_local else "")}.csv'
        }
        if is_versioned:
            catalog[f'dataset_{i}']['versioned'] = True
        if add_interpolation:
            catalog[f'dataset_{i}']['filepath'] = '${_basepath}' + catalog[f'dataset_{i}']['filepath']
    return catalog

def _generate_params(
    start_range: int,
    end_range: int,
    is_local: bool = False,
    add_globals: bool = False
) -> Dict[str, str]:
    if add_globals:
        params: Dict[str, str] = {f'param_{i}': f'${{globals:global_{i}}}' for i in range(start_range, end_range + 1)}
    else:
        params = {f'param_{i}': f'value_{i}{("_local" if is_local else "")}' for i in range(start_range, end_range + 1)}
    return params

def _generate_globals(
    start_range: int,
    end_range: int,
    is_local: bool = False
) -> Dict[str, str]:
    globals_dict: Dict[str, str] = {f'global_{i}': f'value{i}{("_local" if is_local else "")}' for i in range(start_range, end_range + 1)}
    return globals_dict

def _create_config_file(
    conf_source: Path,
    env: str,
    file_name: str,
    data: Dict[str, Any]
) -> None:
    env_path: Path = conf_source / env
    env_path.mkdir(parents=True, exist_ok=True)
    file_path: Path = env_path / file_name
    with open(file_path, 'w') as f:
        yaml.dump(data, f)

base_catalog: Dict[str, Dict[str, Any]] = _generate_catalog(1, 1000, is_versioned=True)
local_catalog: Dict[str, Dict[str, Any]] = _generate_catalog(501, 1500, is_local=True)
base_params: Dict[str, str] = _generate_params(1, 1000)
local_params: Dict[str, str] = _generate_params(501, 1500, is_local=True)
base_globals: Dict[str, str] = _generate_globals(1, 1000)
local_globals: Dict[str, str] = _generate_globals(501, 1500, is_local=True)
base_catalog_with_interpolations: Dict[str, Dict[str, Any]] = _generate_catalog(1, 1000, is_versioned=True, add_interpolation=True)
base_catalog_with_interpolations.update({'_basepath': '/path/to/data'})
local_catalog_with_interpolations: Dict[str, Dict[str, Any]] = _generate_catalog(501, 1500, is_local=True, add_interpolation=True)
local_catalog_with_interpolations.update({'_basepath': '/path/to/data'})
base_params_with_globals: Dict[str, str] = _generate_params(1, 100, add_globals=True)

class TimeOmegaConfigLoader:
    temp_dir: tempfile.TemporaryDirectory
    conf_source: Path
    loader: OmegaConfigLoader

    def setup(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.conf_source = Path(self.temp_dir.name)
        _create_config_file(self.conf_source, 'base', 'catalog.yml', base_catalog)
        _create_config_file(self.conf_source, 'local', 'catalog.yml', local_catalog)
        _create_config_file(self.conf_source, 'base', 'parameters.yml', base_params)
        _create_config_file(self.conf_source, 'local', 'parameters.yml', local_params)
        _create_config_file(self.conf_source, 'base', 'globals.yml', base_globals)
        _create_config_file(self.conf_source, 'local', 'globals.yml', local_globals)
        self.loader = OmegaConfigLoader(conf_source=self.conf_source, base_env='base', default_run_env='local')

    def teardown(self) -> None:
        self.temp_dir.cleanup()

    def time_loading_catalog(self) -> None:
        """Benchmark the time to load the catalog"""
        _ = self.loader['catalog']

    def time_loading_parameters(self) -> None:
        """Benchmark the time to load the parameters"""
        _ = self.loader['parameters']

    def time_loading_globals(self) -> None:
        """Benchmark the time to load global configuration"""
        _ = self.loader['globals']

    def time_loading_parameters_runtime(self) -> None:
        """Benchmark the time to load parameters with runtime configuration"""
        self.loader.runtime_params = _generate_params(2001, 2002)
        _ = self.loader['parameters']

    def time_merge_soft_strategy(self) -> None:
        """Benchmark the time to load and soft-merge configurations"""
        self.loader.merge_strategy = {'catalog': 'soft'}
        _ = self.loader['catalog']

class TimeOmegaConfigLoaderAdvanced:
    temp_dir: tempfile.TemporaryDirectory
    conf_source: Path
    loader: OmegaConfigLoader

    def setup(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.conf_source = Path(self.temp_dir.name)
        _create_config_file(self.conf_source, 'base', 'catalog.yml', base_catalog_with_interpolations)
        _create_config_file(self.conf_source, 'local', 'catalog.yml', local_catalog_with_interpolations)
        _create_config_file(self.conf_source, 'base', 'parameters.yml', base_params_with_globals)
        _create_config_file(self.conf_source, 'base', 'globals.yml', base_globals)
        self.loader = OmegaConfigLoader(conf_source=self.conf_source, base_env='base', default_run_env='local')

    def teardown(self) -> None:
        self.temp_dir.cleanup()

    def time_loading_catalog(self) -> None:
        """Benchmark the time to load the catalog"""
        _ = self.loader['catalog']

    def time_loading_parameters(self) -> None:
        """Benchmark the time to load parameters with global interpolation"""
        _ = self.loader['parameters']