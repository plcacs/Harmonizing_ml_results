def get_config() -> configparser.ConfigParser:
def assert_github_user_exists(github_username: str) -> bool:
def get_ssh_public_keys_from_github(github_username: str) -> List[Dict[str, Any]]:
def assert_user_forked_zulip_server_repo(username: str) -> bool:
def assert_droplet_does_not_exist(my_token: str, droplet_name: str, recreate: bool) -> None:
def get_ssh_keys_string_from_github_ssh_key_dicts(userkey_dicts: List[Dict[str, Any]]) -> str:
def generate_dev_droplet_user_data(username: str, subdomain: str, userkey_dicts: List[Dict[str, Any]]) -> str:
def generate_prod_droplet_user_data(username: str, userkey_dicts: List[Dict[str, Any]]) -> str:
def create_droplet(my_token: str, template_id: str, name: str, tags: List[str], user_data: str, region: str = 'nyc3') -> Tuple[str, str]:
def delete_existing_records(records: List[Any], record_name: str) -> None:
def create_dns_record(my_token: str, record_name: str, ipv4: str, ipv6: str) -> None:
def print_dev_droplet_instructions(username: str, droplet_domain_name: str) -> None:
def print_production_droplet_instructions(droplet_domain_name: str) -> None:
def get_zulip_oneclick_app_slug(api_token: str) -> str:
