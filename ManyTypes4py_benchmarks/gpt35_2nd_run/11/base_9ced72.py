from pydantic import BaseModel, Field, model_validator
from typing import Any, Dict, Optional, Type
from typing_extensions import Self

class DbtConfigs(Block, abc.ABC):
    extras: Optional[Dict[str, Any]] = Field(default=None, description="Extra target configs' keywords, not yet exposed in prefect-dbt, but available in dbt.")
    allow_field_overrides: bool = Field(default=False, description='If enabled, fields from dbt target configs will override fields provided in extras and credentials.')
    _documentation_url: str = 'https://docs.prefect.io/integrations/prefect-dbt'

    def _populate_configs_json(self, configs_json: Dict[str, Any], fields: Dict[str, Any], model: Optional[BaseModel] = None) -> Dict[str, Any]:
        ...

    def get_configs(self) -> Dict[str, Any]:
        ...

class BaseTargetConfigs(DbtConfigs, abc.ABC):
    type: Any = Field(default=..., description='The name of the database warehouse.')
    schema_: str = Field(alias='schema', description='The schema that dbt will build objects into; in BigQuery, a schema is actually a dataset.')
    threads: int = Field(default=4, description='The number of threads representing the max number of paths through the graph dbt may work on at once.')

    @model_validator(mode='before')
    @classmethod
    def handle_target_configs(cls, v: Any) -> Any:
        ...

class TargetConfigs(BaseTargetConfigs):
    type: Any
    schema: str
    threads: int

    @classmethod
    def from_profiles_yml(cls, profile_name: Optional[str] = None, target_name: Optional[str] = None, profiles_dir: Optional[str] = None, allow_field_overrides: bool = False) -> 'TargetConfigs':
        ...

class GlobalConfigs(DbtConfigs):
    send_anonymous_usage_stats: Optional[bool] = Field(default=None, description='Whether usage stats are sent to dbt.')
    use_colors: Optional[bool] = Field(default=None, description='Colorize the output it prints in your terminal.')
    partial_parse: Optional[bool] = Field(default=None, description='When partial parsing is enabled, dbt will use an stored internal manifest to determine which files have been changed (if any) since it last parsed the project.')
    printer_width: Optional[int] = Field(default=None, description='Length of characters before starting a new line.')
    write_json: Optional[bool] = Field(default=None, description='Determines whether dbt writes JSON artifacts to the target/ directory.')
    warn_error: Optional[bool] = Field(default=None, description='Whether to convert dbt warnings into errors.')
    log_format: Optional[str] = Field(default=None, description="The LOG_FORMAT config specifies how dbt's logs should be formatted. If the value of this config is json, dbt will output fully structured logs in JSON format.")
    debug: Optional[bool] = Field(default=None, description="Whether to redirect dbt's debug logs to standard out.")
    version_check: Optional[bool] = Field(default=None, description="Whether to raise an error if a project's version is used with an incompatible dbt version.")
    fail_fast: Optional[bool] = Field(default=None, description='Make dbt exit immediately if a single resource fails to build.')
    use_experimental_parser: Optional[bool] = Field(default=None, description='Opt into the latest experimental version of the static parser.')
    static_parser: Optional[bool] = Field(default=None, description='Whether to use the [static parser](https://docs.getdbt.com/reference/parsing#static-parser).')

class MissingExtrasRequireError(ImportError):
    def __init__(self, service: str, *args, **kwargs):
        ...
