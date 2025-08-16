def remove_root(file_path: str) -> str:
    """Remove the first directory of a path"""
    full_path = PurePosixPath(file_path)
    relative_path = PurePosixPath(*full_path.parts[1:])
    return str(relative_path)

def load_yaml(file_name: str, content: str) -> Any:
    """Try to load a YAML file"""
    try:
        return yaml.safe_load(content)
    except yaml.parser.ParserError as ex:
        logger.exception('Invalid YAML in %s', file_name)
        raise ValidationError({file_name: 'Not a valid YAML file'}) from ex

def load_metadata(contents: dict) -> dict:
    """Apply validation and load a metadata file"""
    if METADATA_FILE_NAME not in contents:
        raise IncorrectVersionError(f'Missing {METADATA_FILE_NAME}')
    metadata = load_yaml(METADATA_FILE_NAME, contents[METADATA_FILE_NAME])
    try:
        MetadataSchema().load(metadata)
    except ValidationError as ex:
        if 'version' in ex.messages:
            raise IncorrectVersionError(ex.messages['version'][0]) from ex
        ex.messages = {METADATA_FILE_NAME: ex.messages}
        raise
    return metadata

def validate_metadata_type(metadata: dict, type_: str, exceptions: list):
    """Validate that the type declared in METADATA_FILE_NAME is correct"""
    if metadata and 'type' in metadata:
        type_validator = validate.Equal(type_)
        try:
            type_validator(metadata['type'])
        except ValidationError as exc:
            exc.messages = {METADATA_FILE_NAME: {'type': exc.messages}}
            exceptions.append(exc)

def load_configs(contents: dict, schemas: dict, passwords: dict, exceptions: list, ssh_tunnel_passwords: dict, ssh_tunnel_private_keys: dict, ssh_tunnel_priv_key_passwords: dict) -> dict:
    configs = {}
    db_passwords = {str(uuid): password for uuid, password in db.session.query(Database.uuid, Database.password).all()}
    db_ssh_tunnel_passwords = {str(uuid): password for uuid, password in db.session.query(SSHTunnel.uuid, SSHTunnel.password).all()}
    db_ssh_tunnel_private_keys = {str(uuid): private_key for uuid, private_key in db.session.query(SSHTunnel.uuid, SSHTunnel.private_key).all()}
    db_ssh_tunnel_priv_key_passws = {str(uuid): private_key_password for uuid, private_key_password in db.session.query(SSHTunnel.uuid, SSHTunnel.private_key_password).all()}
    for file_name, content in contents.items():
        if not content:
            continue
        prefix = file_name.split('/')[0]
        schema = schemas.get(f'{prefix}/')
        if schema:
            try:
                config = load_yaml(file_name, content)
                if file_name in passwords:
                    config['password'] = passwords[file_name]
                elif prefix == 'databases' and config['uuid'] in db_passwords:
                    config['password'] = db_passwords[config['uuid']]
                if file_name in ssh_tunnel_passwords:
                    config['ssh_tunnel']['password'] = ssh_tunnel_passwords[file_name]
                elif prefix == 'databases' and config['uuid'] in db_ssh_tunnel_passwords:
                    config['ssh_tunnel']['password'] = db_ssh_tunnel_passwords[config['uuid']]
                if file_name in ssh_tunnel_private_keys:
                    config['ssh_tunnel']['private_key'] = ssh_tunnel_private_keys[file_name]
                elif prefix == 'databases' and config['uuid'] in db_ssh_tunnel_private_keys:
                    config['ssh_tunnel']['private_key'] = db_ssh_tunnel_private_keys[config['uuid']]
                if file_name in ssh_tunnel_priv_key_passwords:
                    config['ssh_tunnel']['private_key_password'] = ssh_tunnel_priv_key_passwords[file_name]
                elif prefix == 'databases' and config['uuid'] in db_ssh_tunnel_priv_key_passws:
                    config['ssh_tunnel']['private_key_password'] = db_ssh_tunnel_priv_key_passws[config['uuid']]
                schema.load(config)
                configs[file_name] = config
            except ValidationError as exc:
                exc.messages = {file_name: exc.messages}
                exceptions.append(exc)
    return configs

def is_valid_config(file_name: str) -> bool:
    path = Path(file_name)
    if path.name.startswith('.') or path.name.startswith('_'):
        return False
    if path.suffix.lower() not in {'.yaml', '.yml'}:
        return False
    return True

def get_contents_from_bundle(bundle: ZipFile) -> dict:
    check_is_safe_zip(bundle)
    return {remove_root(file_name): bundle.read(file_name).decode() for file_name in bundle.namelist() if is_valid_config(file_name)}
