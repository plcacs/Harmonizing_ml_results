    def __init__(self, jsonrpc_client: JSONRPCClient, secret_registry_address: str, contract_manager: ContractManager, block_identifier: BlockIdentifier) -> None:
    def register_secret(self, secret: Secret) -> None:
    def register_secret_batch(self, secrets: List[Secret]) -> List[TransactionHash]:
    def _register_secret_batch(self, secrets_to_register: List[Secret], transaction_result: AsyncResult, log_details: Dict[str, Any]) -> None:
    def get_secret_registration_block_by_secrethash(self, secrethash: SecretHash, block_identifier: BlockIdentifier) -> Optional[BlockNumber]:
    def is_secret_registered(self, secrethash: SecretHash, block_identifier: BlockIdentifier) -> bool:
