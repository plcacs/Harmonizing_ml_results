    def __init__(self, address: Address):
        self.balances_mapping: Dict[Address, TokenAmount] = {}
        self.chain_id: ChainID = ChainID(UNIT_CHAIN_ID)
        self.address: Address = address

    @staticmethod
    def can_query_state_for_block(block_identifier: BlockIdentifier) -> bool:

    def gas_price(self) -> TokenAmount:

    def balance(self, address: Address) -> TokenAmount:

    def __init__(self, client: MockJSONRPCClient):

    @staticmethod
    def detail_participants(participant1: Address, participant2: Address, block_identifier: BlockIdentifier, channel_identifier: ChannelID) -> None:

    def __init__(self, token_network: MockTokenNetwork, channel_id: ChannelID):

    def __init__(self, node_address: Address, mocked_addresses: Optional[Dict[str, Address]] = None):

    def payment_channel(self, channel_state: MockChannelState, block_identifier: BlockIdentifier) -> MockPaymentChannel:

    def token_network_registry(self, address: TokenNetworkRegistryAddress, block_identifier: BlockIdentifier) -> Mock:

    def secret_registry(self, address: Address, block_identifier: BlockIdentifier) -> Mock:

    def user_deposit(self, address: Address, block_identifier: BlockIdentifier) -> Mock:

    def service_registry(self, address: Address, block_identifier: BlockIdentifier) -> Mock:

    def one_to_n(self, address: Address, block_identifier: BlockIdentifier) -> Mock:

    def monitoring_service(self, address: Address, block_identifier: BlockIdentifier) -> Mock:

    def __init__(self):

    def __init__(self):

    def __init__(self):

    def __init__(self, message_handler: Optional[Any] = None, state_transition: Optional[Any] = None, private_key: Optional[bytes] = None):

    def add_notification(self, notification: Notification, click_opts: Optional[Dict[str, Any]] = None) -> None:

    def on_messages(self, messages: List[Any]) -> None:

    def handle_and_track_state_changes(self, state_changes: List[Any]) -> None:

    def handle_state_changes(self, state_changes: List[Any]) -> None:

    def sign(self, message: Any) -> None:

    def stop(self) -> None:

    def __del__(self) -> None:

def make_raiden_service_mock(token_network_registry_address: TokenNetworkRegistryAddress, token_network_address: TokenNetworkAddress, channel_identifier: ChannelID, partner: Address) -> MockRaidenService:

def mocked_failed_response(error: Exception, status_code: int = 200) -> Mock:

def mocked_json_response(response_data: Optional[Dict[str, Any]] = None, status_code: int = 200) -> Mock:

    def __init__(self, chain_id: ChainID):

    def get_block(self, block_identifier: BlockIdentifier) -> Dict[str, Any]:

    @property
    def chainId(self) -> ChainID:

def PFSMock(pfs_info: PFSInfo):

    def get_pfs_info(self, url: str) -> PFSInfo:

    def on_new_block(self, block: Block) -> None:

    def update_info(self, confirmed_block_number: Optional[BlockNumber] = None, price: Optional[TokenAmount] = None, matrix_server: Optional[str] = None) -> None:

    def query_address_metadata(self, pfs_config: PFSConfig, user_address: Address) -> AddressMetadata:

    @staticmethod
    def _get_app_address_metadata(app: RaidenService) -> Tuple[Address, AddressMetadata]:

    def add_apps(self, apps: List[RaidenService], add_pfs_config: bool = True) -> None:

    def set_route(self, token_address: TokenAddress, route: List[RaidenService]) -> None:

    def reset_routes(self, token_address: Optional[TokenAddress] = None) -> None:

    def get_best_routes_pfs(self, chain_state: ChainState, token_network_address: TokenNetworkAddress, one_to_n_address: Address, from_address: Address, to_address: Address, amount: TokenAmount, previous_address: Address, pfs_config: PFSConfig, privkey: bytes, pfs_wait_for_block: BlockNumber) -> Tuple[Optional[str], List[RouteState], Optional[TokenAmount]]:

def make_pfs_config() -> PFSConfig:
