"""Module containing models for base configs"""
import abc
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self
from prefect.blocks.core import Block
from prefect_dbt.utilities import load_profiles_yml

class DbtConfigs(Block, abc.ABC):
    """
    Abstract class for other dbt Configs.

    Attributes:
        extras: Extra target configs' keywords, not yet exposed
            in prefect-dbt, but available in dbt; if there are
            duplicate keys between extras and TargetConfigs,
            an error will be raised.
    """
    extras: Optional[Dict[str, Any]] = Field(default=None, description="Extra target configs' keywords, not yet exposed in prefect-dbt, but available in dbt.")
    allow_field_overrides: bool = Field(default=False, description='If enabled, fields from dbt target configs will override fields provided in extras and credentials.')
    _documentation_url: str = 'https://docs.prefect.io/integrations/prefect-dbt'

    def _populate_configs_json(self, configs_json: Dict[str, Any], fields: Dict[str, Any], model: Optional[BaseModel] = None) -> Dict[str, Any]:
        """
        Recursively populate configs_json.
        """
        override_configs_json: Dict[str, Any] = {}
        for field_name, field in fields.items():
            if model is not None:
                field_value = getattr(model, field_name, None)
                if field.alias:
                    field_name = field.alias
            else:
                field_value = field
            if field_value is None or field_name == 'allow_field_overrides':
                continue
            if isinstance(field_value, BaseModel):
                configs_json = self._populate_configs_json(configs_json, field_value.model_fields, model=field_value)
            elif field_name == 'extras':
                configs_json = self._populate_configs_json(configs_json, field_value)
                override_configs_json.update(configs_json)
            else:
                if field_name in configs_json.keys() and (not self.allow_field_overrides):
                    raise ValueError(f'The keyword, {field_name}, has already been provided in TargetConfigs; remove duplicated keywords to continue')
                if hasattr(field_value, 'get_secret_value'):
                    field_value = field_value.get_secret_value()
                elif isinstance(field_value, Path):
                    field_value = str(field_value)
                configs_json[field_name] = field_value
                if self.allow_field_overrides and model is self or model is None:
                    override_configs_json[field_name] = field_value
        configs_json.update(override_configs_json)
        return configs_json

    def get_configs(self) -> Dict[str, Any]:
        """
        Returns the dbt configs, likely used eventually for writing to profiles.yml.

        Returns:
            A configs JSON.
        """
        return self._populate_configs_json({}, self.model_fields, model=self)

class BaseTargetConfigs(DbtConfigs, abc.ABC):
    type: str = Field(default=..., description='The name of the database warehouse.')
    schema_: str = Field(alias='schema', description='The schema that dbt will build objects into; in BigQuery, a schema is actually a dataset.')
    threads: int = Field(default=4, description='The number of threads representing the max number of paths through the graph dbt may work on at once.')

    @model_validator(mode='before')
    @classmethod
    def handle_target_configs(cls: Type[Self], v: Union[Dict[str, Any], Any]) -> Union[Dict[str, Any], Any]:
        """Handle target configs field aliasing during validation"""
        if isinstance(v, dict):
            if 'schema_' in v:
                v['schema'] = v.pop('schema_')
            for value in v.values():
                if isinstance(value, dict) and 'schema_' in value:
                    value['schema'] = value.pop('schema_')
        return v

class TargetConfigs(BaseTargetConfigs):
    """
    Target configs contain credentials and
    settings, specific to the warehouse you're connecting to.
    To find valid keys, head to the [Available adapters](
    https://docs.getdbt.com/docs/available-adapters) page and
    click the desired adapter's "Profile Setup" hyperlink.

    Attributes:
        type: The name of the database warehouse.
        schema: The schema that dbt will build objects into;
            in BigQuery, a schema is actually a dataset.
        threads: The number of threads representing the max number
            of paths through the graph dbt may work on at once.

    Examples:
        Load stored TargetConfigs:
        