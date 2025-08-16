def address_from_userid(user_id: str) -> Optional[Address]:
def is_valid_userid_for_address(user_id: str, address: Address) -> bool:
def get_user_id_from_metadata(address: Address, address_metadata: Optional[AddressMetadata]) -> Optional[UserID]:
def is_valid_username(username: str, server_name: str, user_id: str) -> bool:
def login_with_token(client: GMatrixClient, user_id: str, access_token: str) -> Optional[User]:
def login(client: GMatrixClient, signer: Signer, device_id: DeviceIDs, prev_auth_data: Optional[str] = None, capabilities: Optional[Dict[str, Any]] = None) -> User:
def validate_userid_signature(user: User) -> Optional[Address]:
def validate_user_id_signature(user_id: str, displayname: str) -> Optional[Address]:
def sort_servers_closest(servers: Sequence[str], max_timeout: float = 3.0, samples_per_server: int = 3, sample_delay: float = 0.125) -> Dict[str, float]:
def make_client(handle_messages_callback: Callable, servers: List[str], *args, **kwargs) -> GMatrixClient:
def validate_and_parse_message(data: str, peer_address: Address) -> List[SignedMessage]:
def my_place_or_yours(our_address: Address, partner_address: Address) -> Address:
def make_message_batches(message_texts: Iterable[str], _max_batch_size: int = MATRIX_MAX_BATCH_SIZE) -> Generator[str, None, None]:
def make_user_id(address: Address, home_server: str) -> UserID:
