    def get_token_network(self, token_address: T_TargetAddress, block_identifier: BlockIdentifier) -> Optional[TokenNetworkAddress]:
    
    def add_token(self, token_address: T_TargetAddress, channel_participant_deposit_limit: TokenAmount, token_network_deposit_limit: TokenAmount, given_block_identifier: BlockIdentifier) -> Tuple[TransactionHash, TokenNetworkAddress]:
    
    def _add_token(self, token_address: T_TargetAddress, channel_participant_deposit_limit: TokenAmount, token_network_deposit_limit: TokenAmount, log_details: Dict[str, Any]) -> Tuple[TransactionHash, TokenNetworkAddress]:
    
    def get_secret_registry_address(self, block_identifier: BlockIdentifier) -> SecretRegistryAddress:
    
    def get_controller(self, block_identifier: BlockIdentifier) -> Address:
    
    def settlement_timeout_min(self, block_identifier: BlockIdentifier) -> int:
    
    def settlement_timeout_max(self, block_identifier: BlockIdentifier) -> int:
    
    def get_token_network_created(self, block_identifier: BlockIdentifier) -> int:
    
    def get_max_token_networks(self, block_identifier: BlockIdentifier) -> int:
