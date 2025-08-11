from __future__ import annotations
import os
import sys
import json
from typing import Dict, Any, Optional, List, Union
from chalice import __version__ as current_chalice_version
from chalice.app import Chalice
from chalice.constants import DEFAULT_STAGE_NAME
from chalice.constants import DEFAULT_HANDLER_NAME
StrMap = Dict[str, Any]

class Config(object):
    """Configuration information for a chalice app.

    Configuration values for a chalice app can come from
    a number of locations, files on disk, CLI params, default
    values, etc.  This object is an abstraction that normalizes
    these values.

    In general, there's a precedence for looking up
    config values:

        * User specified params
        * Config file values
        * Default values

    A user specified parameter would mean values explicitly
    specified by a user.  Generally these come from command
    line parameters (e.g ``--profile prod``), but for the purposes
    of this object would also mean values passed explicitly to
    this config object when instantiated.

    Additionally, there are some configurations that can vary
    per chalice stage (note that a chalice stage is different
    from an api gateway stage).  For config values loaded from
    disk, we allow values to be specified for all stages or
    for a specific stage.  For example, take ``environment_variables``.
    You can set this as a top level key to specify env vars
    to set for all stages, or you can set this value per chalice
    stage to set stage-specific environment variables.  Consider
    this config file::

        {
          "environment_variables": {
            "TABLE": "foo"
          },
          "stages": {
            "dev": {
              "environment_variables": {
                "S3BUCKET": "devbucket"
              }
            },
            "prod": {
              "environment_variables": {
                "S3BUCKET": "prodbucket",
                "TABLE": "prodtable"
              }
            }
          }
        }

    If the currently configured chalice stage is "dev", then
    the config.environment_variables would be::

        {"TABLE": "foo", "S3BUCKET": "devbucket"}

    The "prod" stage would be::

        {"TABLE": "prodtable", "S3BUCKET": "prodbucket"}

    """

    def __init__(self, chalice_stage=DEFAULT_STAGE_NAME, function_name=DEFAULT_HANDLER_NAME, user_provided_params=None, config_from_disk=None, default_params=None, layers=None) -> None:
        self.chalice_stage = chalice_stage
        self.function_name = function_name
        if user_provided_params is None:
            user_provided_params = {}
        self._user_provided_params = user_provided_params
        if config_from_disk is None:
            config_from_disk = {}
        self._config_from_disk = config_from_disk
        if default_params is None:
            default_params = {}
        self._default_params = default_params
        self._chalice_app = None
        self._layers = layers

    @classmethod
    def create(cls: Union[str, typing.NamedTuple, typing.Callable], chalice_stage: Any=DEFAULT_STAGE_NAME, function_name: Any=DEFAULT_HANDLER_NAME, **kwargs):
        return cls(chalice_stage=chalice_stage, user_provided_params=kwargs.copy())

    @property
    def profile(self) -> Union[dict, str, None, bytes]:
        return self._chain_lookup('profile')

    @property
    def app_name(self) -> Union[str, list[str], None, bool]:
        return self._chain_lookup('app_name')

    @property
    def project_dir(self) -> str:
        return self._chain_lookup('project_dir')

    @property
    def chalice_app(self) -> Union[Chalice, flask.app.Flask, lemon.app.Lemon, asyncworker.app.App]:
        v = self._chain_lookup('chalice_app')
        if isinstance(v, Chalice):
            return v
        elif self._chalice_app is not None:
            return self._chalice_app
        elif not callable(v):
            raise TypeError('Unable to load chalice app, lazy loader is not callable: %s' % v)
        app = v()
        self._chalice_app = app
        return app

    @property
    def config_from_disk(self):
        return self._config_from_disk

    @property
    def lambda_python_version(self) -> typing.Text:
        major, minor = (sys.version_info[0], sys.version_info[1])
        if (major, minor) < (3, 8):
            return 'python3.8'
        elif (major, minor) <= (3, 11):
            return 'python%s.%s' % (major, minor)
        return 'python3.12'

    @property
    def log_retention_in_days(self) -> int:
        return self._chain_lookup('log_retention_in_days', varies_per_chalice_stage=True, varies_per_function=True)

    @property
    def layers(self) -> Union[Atom, list]:
        return self._chain_lookup('layers', varies_per_chalice_stage=True, varies_per_function=True)

    @property
    def api_gateway_custom_domain(self) -> list[str]:
        return self._chain_lookup('api_gateway_custom_domain', varies_per_chalice_stage=True)

    @property
    def websocket_api_custom_domain(self) -> typing.Mapping:
        return self._chain_lookup('websocket_api_custom_domain', varies_per_chalice_stage=True)

    def _chain_lookup(self, name: Union[str, typing.AbstractSet, None, int], varies_per_chalice_stage: bool=False, varies_per_function: bool=False):
        search_dicts = [self._user_provided_params]
        if varies_per_chalice_stage:
            search_dicts.append(self._config_from_disk.get('stages', {}).get(self.chalice_stage, {}))
        if varies_per_function:
            search_dicts.insert(0, self._config_from_disk.get('stages', {}).get(self.chalice_stage, {}).get('lambda_functions', {}).get(self.function_name, {}))
            search_dicts.append(self._config_from_disk.get('lambda_functions', {}).get(self.function_name, {}))
        search_dicts.extend([self._config_from_disk, self._default_params])
        for cfg_dict in search_dicts:
            if isinstance(cfg_dict, dict) and cfg_dict.get(name) is not None:
                return cfg_dict[name]

    def _chain_merge(self, name: Union[str, None]) -> dict:
        search_dicts = [self._default_params, self._config_from_disk, self._config_from_disk.get('stages', {}).get(self.chalice_stage, {}), self._config_from_disk.get('stages', {}).get(self.chalice_stage, {}).get('lambda_functions', {}).get(self.function_name, {}), self._user_provided_params]
        final = {}
        for cfg_dict in search_dicts:
            value = cfg_dict.get(name, {})
            if isinstance(value, dict):
                final.update(value)
        return final

    @property
    def config_file_version(self) -> Union[bool, str, typing.Iterable[str]]:
        return self._config_from_disk.get('version', '1.0')

    @property
    def api_gateway_stage(self) -> Union[dict, dict[str, typing.Any], solo.config.app.Config]:
        return self._chain_lookup('api_gateway_stage', varies_per_chalice_stage=True)

    @property
    def api_gateway_endpoint_type(self) -> Union[str, typing.Callable, None]:
        return self._chain_lookup('api_gateway_endpoint_type', varies_per_chalice_stage=True)

    @property
    def api_gateway_endpoint_vpce(self) -> Union[typing.Callable, str, dict[str, str]]:
        return self._chain_lookup('api_gateway_endpoint_vpce', varies_per_chalice_stage=True)

    @property
    def api_gateway_policy_file(self) -> Union[str, dict, bool]:
        return self._chain_lookup('api_gateway_policy_file', varies_per_chalice_stage=True)

    @property
    def minimum_compression_size(self) -> Union[int, str, typing.Callable[float, None]]:
        return self._chain_lookup('minimum_compression_size', varies_per_chalice_stage=True)

    @property
    def iam_policy_file(self) -> Union[str, list[str], list[dict[str, str]]]:
        return self._chain_lookup('iam_policy_file', varies_per_chalice_stage=True, varies_per_function=True)

    @property
    def lambda_memory_size(self) -> Union[str, int, typing.Callable[float, None]]:
        return self._chain_lookup('lambda_memory_size', varies_per_chalice_stage=True, varies_per_function=True)

    @property
    def lambda_timeout(self) -> Union[int, typing.Sequence[dict[str, typing.Any]]]:
        return self._chain_lookup('lambda_timeout', varies_per_chalice_stage=True, varies_per_function=True)

    @property
    def automatic_layer(self) -> Union[bool, int, float, list[str], None]:
        v = self._chain_lookup('automatic_layer', varies_per_chalice_stage=True, varies_per_function=False)
        if v is None:
            return False
        return v

    @property
    def iam_role_arn(self) -> Union[tuple[typing.Union[float,int]], str, bool]:
        return self._chain_lookup('iam_role_arn', varies_per_chalice_stage=True, varies_per_function=True)

    @property
    def manage_iam_role(self) -> Union[bool, list, dict[str, int], int, None]:
        result = self._chain_lookup('manage_iam_role', varies_per_chalice_stage=True, varies_per_function=True)
        if result is None:
            return True
        return result

    @property
    def autogen_policy(self) -> Union[dict[bytes, int], tuple[bytes]]:
        return self._chain_lookup('autogen_policy', varies_per_chalice_stage=True, varies_per_function=True)

    @property
    def xray_enabled(self) -> Union[bool, str, None]:
        return self._chain_lookup('xray', varies_per_chalice_stage=True, varies_per_function=True)

    @property
    def environment_variables(self) -> Union[str, dict[str, typing.Any], dict]:
        return self._chain_merge('environment_variables')

    @property
    def tags(self):
        tags = self._chain_merge('tags')
        tags['aws-chalice'] = 'version=%s:stage=%s:app=%s' % (current_chalice_version, self.chalice_stage, self.app_name)
        return tags

    @property
    def security_group_ids(self) -> Union[str, tuple[typing.Union[float,int]], list[str]]:
        return self._chain_lookup('security_group_ids', varies_per_chalice_stage=True, varies_per_function=True)

    @property
    def subnet_ids(self) -> Union[str, bytes, int]:
        return self._chain_lookup('subnet_ids', varies_per_chalice_stage=True, varies_per_function=True)

    @property
    def reserved_concurrency(self) -> Union[str, typing.Callable[None,None, bytes]]:
        return self._chain_lookup('reserved_concurrency', varies_per_chalice_stage=True, varies_per_function=True)

    def scope(self, chalice_stage: Union[str, typing.Mapping, typing.Callable[int, None]], function_name: Union[str, typing.Mapping, typing.Callable[int, None]]):
        clone = self.__class__(chalice_stage=chalice_stage, function_name=function_name, user_provided_params=self._user_provided_params, config_from_disk=self._config_from_disk, default_params=self._default_params)
        return clone

    def deployed_resources(self, chalice_stage_name: str) -> Union[DeployedResources, str, typing.Callable[str, None], bool]:
        """Return resources associated with a given stage.

        If a deployment to a given stage has never happened,
        this method will return a value of None.

        """
        deployed_file = os.path.join(self.project_dir, '.chalice', 'deployed', '%s.json' % chalice_stage_name)
        data = self._load_json_file(deployed_file)
        if data is not None:
            schema_version = data.get('schema_version', '1.0')
            if schema_version != '2.0':
                raise ValueError('Unsupported schema version (%s) in file: %s' % (schema_version, deployed_file))
            return DeployedResources(data)
        return self._try_old_deployer_values(chalice_stage_name)

    def _try_old_deployer_values(self, chalice_stage_name: Union[str, None, dict[str, str]]) -> Union[bool, str]:
        old_deployed_file = os.path.join(self.project_dir, '.chalice', 'deployed.json')
        data = self._load_json_file(old_deployed_file)
        if data is None or chalice_stage_name not in data:
            return DeployedResources.empty()
        return self._upgrade_deployed_values(chalice_stage_name, data)

    def _load_json_file(self, deployed_file: str) -> None:
        if not os.path.isfile(deployed_file):
            return None
        with open(deployed_file, 'r') as f:
            return json.load(f)

    def _upgrade_deployed_values(self, chalice_stage_name: Union[str, None, dict[str, typing.Any]], data: Union[str, dict[str, str]]) -> DeployedResources:
        deployed = data[chalice_stage_name]
        prefix = '%s-%s-' % (self.app_name, chalice_stage_name)
        resources = []
        self._upgrade_lambda_functions(resources, deployed, prefix)
        self._upgrade_rest_api(resources, deployed)
        return DeployedResources({'resources': resources, 'schema_version': '2.0'})

    def _upgrade_lambda_functions(self, resources: list[str], deployed: Union[dict, raiden.constants.Environment], prefix: Union[str, list[str]]) -> None:
        lambda_functions = deployed.get('lambda_functions', {})
        is_pre_10_format = not all((isinstance(v, dict) for v in lambda_functions.values()))
        if is_pre_10_format:
            lambda_functions = {k: {'type': 'authorizer', 'arn': v} for k, v in lambda_functions.items()}
        for name, values in lambda_functions.items():
            short_name = name[len(prefix):]
            current = {'resource_type': 'lambda_function', 'lambda_arn': values['arn'], 'name': short_name}
            resources.append(current)

    def _upgrade_rest_api(self, resources: Union[list[str], list, list[dict[str, typing.Any]]], deployed: Union[list[str], list, list[dict[str, typing.Any]]]) -> None:
        resources.extend([{'name': 'api_handler', 'resource_type': 'lambda_function', 'lambda_arn': deployed['api_handler_arn']}, {'name': 'rest_api', 'resource_type': 'rest_api', 'rest_api_id': deployed['rest_api_id']}])

class DeployedResources(object):

    def __init__(self, deployed_values: Any) -> None:
        self._deployed_values = deployed_values['resources']
        self._deployed_values_by_name = {resource['name']: resource for resource in deployed_values['resources']}

    @classmethod
    def empty(cls: Union[typing.Callable[typing.Any, T], bool, typing.Type]) -> str:
        return cls({'resources': [], 'schema_version': '2.0'})

    def resource_values(self, name: str):
        if 'api_mapping' in name:
            name = name.split('.')[0]
        try:
            return self._deployed_values_by_name[name]
        except KeyError:
            raise ValueError('Resource does not exist: %s' % name)

    def resource_names(self) -> list:
        return [r['name'] for r in self._deployed_values]