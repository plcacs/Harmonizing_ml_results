    def __init__(self, jsonrpc_client: JSONRPCClient, token_address: TokenAddress, contract_manager: ContractManager, block_identifier: BlockIdentifier) -> None:
    def allowance(self, owner: Address, spender: Address, block_identifier: BlockIdentifier) -> TokenAmount:
    def approve(self, allowed_address: Address, allowance: TokenAmount) -> TransactionHash:
    def balance_of(self, address: Address, block_identifier: BlockIdentifier = BLOCK_ID_LATEST) -> TokenAmount:
    def total_supply(self, block_identifier: BlockIdentifier = BLOCK_ID_LATEST) -> Optional[TokenAmount]:
    def transfer(self, to_address: Address, amount: TokenAmount) -> TransactionHash:
