def get_matrix_servers(url: str, server_list_type: ServerListType = ServerListType.ACTIVE_SERVERS) -> List[str]:
    ...

ADDRESS_TYPE: click.ParamType = AddressType()
LOG_LEVEL_CONFIG_TYPE: click.ParamType = LogLevelConfigType()
