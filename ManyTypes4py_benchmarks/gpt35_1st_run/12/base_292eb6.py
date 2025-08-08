def _to_environment_variable_value(value: Any) -> str:
    if isinstance(value, (list, set, tuple)):
        return ','.join((str(v) for v in value))
    return str(value)

def _add_environment_variables(schema: Dict[str, Any], model: Type[PrefectBaseSettings]) -> None:
    for property in schema['properties']:
        env_vars: List[str] = []
        schema['properties'][property]['supported_environment_variables'] = env_vars
        field = model.model_fields[property]
        if inspect.isclass(field.annotation) and issubclass(field.annotation, PrefectBaseSettings):
            continue
        elif field.validation_alias:
            if isinstance(field.validation_alias, AliasChoices):
                for alias in field.validation_alias.choices:
                    if isinstance(alias, str):
                        env_vars.append(alias.upper())
        else:
            env_vars.append(f'{model.model_config.get('env_prefix')}{property.upper()}')

def build_settings_config(path: Tuple[str] = tuple(), frozen: bool = False) -> PrefectSettingsConfigDict:
    env_prefix = f'PREFECT_{'_'.join(path).upper()}_' if path else 'PREFECT_'
    return PrefectSettingsConfigDict(env_prefix=env_prefix, env_file='.env', extra='ignore', toml_file='prefect.toml', prefect_toml_table_header=path, pyproject_toml_table_header=('tool', 'prefect', *path), json_schema_extra=_add_environment_variables, frozen=frozen)
