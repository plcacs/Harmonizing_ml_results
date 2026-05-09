from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from uuid import uuid4

class ClientErrorInspectResult(Enum):
    """Represents the action to follow after inspecting a client exception"""
    PROPAGATE_ERROR = 1
    INSUFFICIENT_FUNDS = 2
    TRANSACTION_UNDERPRICED = 3
    TRANSACTION_PENDING = 4
    ALWAYS_FAIL = 5
    TRANSACTION_ALREADY_IMPORTED = 7
    TRANSACTION_PENDING_OR_ALREADY_IMPORTED = 8

class CallType(Enum):
    ESTIMATE_GAS = 1
    CALL = 2

@dataclass
class EthTransfer:
    to_address: str
    gas_price: int
    value: int

    def __post_init__(self):
        typecheck(self.to_address, str)
        typecheck(self.gas_price, int)
        typecheck(self.value, int)

    def to_log_details(self) -> Dict[str, Any]:
        return {'to_address': to_checksum_address(self.to_address), 'value': self.value, 'gas_price': self.gas_price}

@dataclass
class SmartContractCall:
    contract: Any
    function: str
    args: List[Any]
    kwargs: Dict[str, Any]
    value: int

    def __post_init__(self):
        typecheck(self.contract, Any)
        typecheck(self.function, str)
        typecheck(self.args, List[Any])
        typecheck(self.kwargs, Dict[str, Any])
        typecheck(self.value, int)

    def to_log_details(self) -> Dict[str, Any]:
        to_address = to_checksum_address(self.contract.address)
        return {'function_name': self.function, 'to_address': to_address, 'args': self.args, 'kwargs': self.kwargs, 'value': self.value}

@dataclass
class ByteCode:
    contract_name: str
    bytecode: str

    def to_log_details(self) -> Dict[str, Any]:
        return {'contract_name': self.contract_name}

@dataclass
class TransactionPending:
    from_address: str
    data: SmartContractCall
    eth_node: Any
    extra_log_details: Dict[str, Any]

    def __post_init__(self):
        typecheck(self.from_address, str)
        typecheck(self.data, SmartContractCall)
        typecheck(self.eth_node, Any)
        typecheck(self.extra_log_details, Dict[str, Any])
        self.extra_log_details.setdefault('token', str(uuid4()))
        log.debug('Transaction created', **self.to_log_details())

    def to_log_details(self) -> Dict[str, Any]:
        log_details = self.data.to_log_details()
        log_details.update(self.extra_log_details)
        log_details.update({'from_address': to_checksum_address(self.from_address), 'eth_node': self.eth_node})
        return log_details

    def estimate_gas(self, block_identifier: Any) -> Optional[Any]:
        """Estimate the gas and price necessary to run the transaction.

        Returns `None` transaction would fail because it hit an assert/require,
        or if the amount of gas required is larger than the block gas limit.
        """
        fn = getattr(self.data.contract.functions, self.data.function)
        from_address = to_checksum_address(self.from_address)
        if self.eth_node is EthClient.GETH:
            block_identifier = None
        try:
            estimated_gas = fn(*self.data.args, **self.data.kwargs).estimateGas(transaction={'from': from_address}, block_identifier=block_identifier)
        except ValueError as err:
            estimated_gas = None
            inspected_error = inspect_client_error(err, self.eth_node)
            expected_error = inspected_error in (ClientErrorInspectResult.INSUFFICIENT_FUNDS, ClientErrorInspectResult.ALWAYS_FAIL)
            if not expected_error:
                raise err
        block = self.data.contract.web3.eth.get_block(BLOCK_ID_LATEST)
        if estimated_gas is not None:
            gas_price = gas_price_for_fast_transaction(self.data.contract.web3)
            transaction_estimated = TransactionEstimated(from_address=self.from_address, eth_node=self.eth_node, data=self.data, extra_log_details=self.extra_log_details, estimated_gas=safe_gas_limit(estimated_gas), gas_price=gas_price, approximate_block=(BlockHash(block['hash']), BlockNumber(block['number'])))
            log.debug('Transaction gas estimated', **transaction_estimated.to_log_details(), node_gas_price=self.data.contract.web3.eth.gas_price)
            return transaction_estimated
        else:
            log.debug('Transaction gas estimation failed', approximate_block_hash=to_hex(block['hash']), approximate_block_number=block['number'], **self.to_log_details())
            return None

@dataclass
class TransactionEstimated:
    from_address: str
    eth_node: Any
    data: Union[SmartContractCall, ByteCode]
    extra_log_details: Dict[str, Any]
    estimated_gas: int
    gas_price: int
    approximate_block: Tuple[Any, Any]

    def __post_init__(self):
        typecheck(self.from_address, str)
        typecheck(self.eth_node, Any)
        typecheck(self.data, Union[SmartContractCall, ByteCode])
        typecheck(self.extra_log_details, Dict[str, Any])
        typecheck(self.estimated_gas, int)
        typecheck(self.gas_price, int)
        typecheck(self.approximate_block, Tuple[Any, Any])
        self.extra_log_details.setdefault('token', str(uuid4()))
        typecheck(self.from_address, str)
        typecheck(self.data, Union[SmartContractCall, ByteCode])
        typecheck(self.estimated_gas, int)
        typecheck(self.gas_price, int)

    def to_log_details(self) -> Dict[str, Any]:
        log_details = self.data.to_log_details()
        log_details.update(self.extra_log_details)
        log_details.update({'from_address': to_checksum_address(self.from_address), 'eth_node': self.eth_node, 'estimated_gas': self.estimated_gas, 'gas_price': self.gas_price, 'approximate_block_hash': to_hex(self.approximate_block[0]), 'approximate_block_number': self.approximate_block[1]})
        return log_details

class TransactionSent(ABC):
    pass

@dataclass
class TransactionMined:
    from_address: str
    data: Union[SmartContractCall, ByteCode, EthTransfer]
    eth_node: Any
    extra_log_details: Dict[str, Any]
    startgas: int
    gas_price: int
    nonce: Any
    transaction_hash: Any
    receipt: Any
    chain_id: Any

    def __post_init__(self):
        typecheck(self.from_address, str)
        typecheck(self.data, Union[SmartContractCall, ByteCode, EthTransfer])
        typecheck(self.eth_node, Any)
        typecheck(self.extra_log_details, Dict[str, Any])
        typecheck(self.startgas, int)
        typecheck(self.gas_price, int)
        typecheck(self.nonce, Any)
        typecheck(self.transaction_hash, Any)
        typecheck(self.receipt, Any)
        typecheck(self.chain_id, Any)

class JSONRPCClient:
    """Ethereum JSON RPC client."""

    def __init__(self, web3: Any, privkey: bytes, gas_price_strategy: Callable = rpc_gas_price_strategy, block_num_confirmations: int = 0):
        if len(privkey) != 32:
            raise ValueError('Invalid private key')
        if block_num_confirmations < 0:
            raise ValueError('Number of confirmations has to be positive')
        monkey_patch_web3(web3, gas_price_strategy)
        version = web3.clientVersion
        supported, eth_node, _ = is_supported_client(version)
        if eth_node is None or supported is VersionSupport.UNSUPPORTED:
            raise EthNodeInterfaceError(f'Unsupported Ethereum client "{version}"')
        if supported is VersionSupport.WARN:
            log.warning(f'Unsupported Ethereum client version "{version}"')
        address = privatekey_to_address(privkey)
        available_nonce = discover_next_available_nonce(web3, eth_node, address)
        self.eth_node = eth_node
        self.privkey = privkey
        self.address = address
        self.web3 = web3
        self.default_block_num_confirmations = block_num_confirmations
        self.chain_id = ChainID(self.web3.eth.chain_id)
        self._available_nonce = available_nonce
        self._nonce_lock = Semaphore()
        log.debug('JSONRPCClient created', node=to_checksum_address(self.address), available_nonce=available_nonce, client=version)

    def __repr__(self) -> str:
        return f'<JSONRPCClient node:{to_checksum_address(self.address)} nonce:{self._available_nonce}>'

    def block_number(self) -> int:
        """Return the most recent block."""
        return self.web3.eth.block_number

    def get_block(self, block_identifier: Any) -> Any:
        """Given a block number, query the chain to get its corresponding block hash"""
        return self.web3.eth.get_block(block_identifier)

    def _sync_nonce(self) -> None:
        self._available_nonce = discover_next_available_nonce(self.web3, self.eth_node, self.address)

    def get_confirmed_blockhash(self) -> Any:
        """Gets the block CONFIRMATION_BLOCKS in the past and returns its block hash"""
        confirmed_block_number = BlockNumber(self.web3.eth.block_number - self.default_block_num_confirmations)
        if confirmed_block_number < 0:
            confirmed_block_number = BlockNumber(0)
        return self.blockhash_from_blocknumber(confirmed_block_number)

    def blockhash_from_blocknumber(self, block_number: Any) -> Any:
        """Given a block number, query the chain to get its corresponding block hash"""
        block = self.get_block(block_number)
        return BlockHash(bytes(block['hash']))

    def can_query_state_for_block(self, block_identifier: Any) -> bool:
        """
        Returns if the provided block identifier is safe enough to query chain
        state for. If it's close to the state pruning blocks then state should
        not be queried.
        More info: https://github.com/raiden-network/raiden/issues/3566.
        """
        latest_block_number = self.block_number()
        preconditions_block = self.web3.eth.get_block(block_identifier)
        preconditions_block_number = int(preconditions_block['number'])
        difference = latest_block_number - preconditions_block_number
        return difference < NO_STATE_QUERY_AFTER_BLOCKS

    def balance(self, account: str) -> int:
        """Return the balance of the account of the given address."""
        return TokenAmount(self.web3.eth.get_balance(account, BLOCK_ID_PENDING))

    def parity_get_pending_transaction_hash_by_nonce(self, address: str, nonce: Any) -> Optional[Any]:
        """Queries the local parity transaction pool and searches for a transaction.

        Checks the local tx pool for a transaction from a particular address and for
        a given nonce. If it exists it returns the transaction hash.
        """
        msg = f'`parity` specific function must only be called when the client is parity. Client was {self.eth_node}.'
        assert self.eth_node is EthClient.PARITY, msg
        transactions = self.web3.manager.request_blocking(RPCEndpoint('parity_allTransactions'), [])
        log.debug('RETURNED TRANSACTIONS', transactions=transactions)
        for tx in transactions:
            address_match = tx['from'] == address
            if address_match and int(tx['nonce'], 16) == nonce:
                return tx['hash']
        return None

    def estimate_gas(self, contract: Any, function: str, extra_log_details: Dict[str, Any], *args: Any, **kwargs: Any) -> Optional[Any]:
        pending = TransactionPending(from_address=self.address, data=SmartContractCall(contract, function, args, kwargs, value=0), eth_node=self.eth_node, extra_log_details=extra_log_details)
        return pending.estimate_gas(BLOCK_ID_PENDING)

    def transact(self, transaction: Any) -> Any:
        """Allocates an unique `nonce` and send the transaction to the blockchain.

        This can fail for a few reasons:

        - The account doesn't have sufficient Eth to pay for the gas.
        - The gas price was too low.
        - Another transaction with the same `nonce` was sent before. This may
          happen because:
          - Another application is using the same private key.
          - The node restarted and the `nonce` recovered by
            `discover_next_available_nonce` was too low, which can happend
            because:
            - The transactions currenlty in the pool are not taken into
              account.
            - There was a gap in the `nonce`s of the transaction in the pool,
              once the gap is filled a `nonce` is reused. This is most likely a
              bug.
        """
        client = self

        @dataclass
        class TransactionSlot:
            from_address: str
            eth_node: Any
            data: Union[SmartContractCall, ByteCode, EthTransfer]
            extra_log_details: Dict[str, Any]
            startgas: int
            gas_price: int
            nonce: Any

            def __post_init__(self):
                self.extra_log_details.setdefault('token', str(uuid4()))
                typecheck(self.from_address, str)
                typecheck(self.data, Union[SmartContractCall, ByteCode, EthTransfer])
                typecheck(self.startgas, int)
                typecheck(self.gas_price, int)
                typecheck(self.nonce, Any)

            def to_log_details(self) -> Dict[str, Any]:
                log_details = self.data.to_log_details()
                log_details.update(self.extra_log_details)
                log_details.update({'node': to_checksum_address(client.address), 'from_address': to_checksum_address(self.from_address), 'eth_node': self.eth_node, 'startgas': self.startgas, 'gas_price': self.gas_price, 'nonce': self.nonce})
                return log_details

        @dataclass
        class TransactionSentImplementation(TransactionSent):
            from_address: str
            eth_node: Any
            data: Union[SmartContractCall, ByteCode, EthTransfer]
            extra_log_details: Dict[str, Any]
            startgas: int
            gas_price: int
            nonce: Any
            transaction_hash: Any
            chain_id: Any

            def __post_init__(self):
                self.extra_log_details.setdefault('token', str(uuid4()))
                typecheck(self.from_address, str)
                typecheck(self.data, Union[SmartContractCall, ByteCode, EthTransfer])
                typecheck(self.startgas, int)
                typecheck(self.gas_price, int)
                typecheck(self.nonce, Any)
                typecheck(self.transaction_hash, Any)
                typecheck(self.chain_id, Any)

            def to_log_details(self) -> Dict[str, Any]:
                log_details = self.data.to_log_details()
                log_details.update(self.extra_log_details)
                log_details.update({'node': to_checksum_address(client.address), 'from_address': to_checksum_address(self.from_address), 'eth_node': self.eth_node, 'startgas': self.startgas, 'gas_price': self.gas_price, 'nonce': self.nonce, 'transaction_hash': encode_hex(self.transaction_hash), 'chain_id': self.chain_id})
                return log_details
        try:
            with self._nonce_lock:
                available_nonce = self._available_nonce
                if isinstance(transaction, EthTransfer):
                    slot = TransactionSlot(from_address=self.address, eth_node=self.eth_node, data=transaction, extra_log_details={}, startgas=TRANSACTION_INTRINSIC_GAS, gas_price=transaction.gas_price, nonce=available_nonce)
                else:
                    slot = TransactionSlot(from_address=transaction.from_address, eth_node=transaction.eth_node, data=transaction.data, extra_log_details=transaction.extra_log_details, startgas=transaction.estimated_gas, gas_price=transaction.gas_price, nonce=available_nonce)
                log_details = slot.to_log_details()
                if isinstance(slot.data, SmartContractCall):
                    function_call = slot.data
                    data = get_transaction_data(web3=function_call.contract.web3, abi=function_call.contract.abi, function_name=function_call.function, args=function_call.args, kwargs=function_call.kwargs)
                    transaction_data = {'data': decode_hex(data), 'gas': slot.startgas, 'nonce': slot.nonce, 'value': slot.data.value, 'to': function_call.contract.address, 'gasPrice': slot.gas_price, 'chainId': self.chain_id}
                    log.debug('Transaction to call smart contract function will be sent', **log_details)
                elif isinstance(slot.data, EthTransfer):
                    transaction_data = {'to': to_checksum_address(slot.data.to_address), 'gas': slot.startgas, 'nonce': slot.nonce, 'value': slot.data.value, 'gasPrice': slot.gas_price, 'chainId': self.chain_id}
                    log.debug('Transaction to transfer ether will be sent', **log_details)
                else:
                    transaction_data = {'data': slot.data.bytecode, 'gas': slot.startgas, 'nonce': slot.nonce, 'value': 0, 'gasPrice': slot.gas_price, 'chainId': self.chain_id}
                    log.debug('Transaction to deploy smart contract will be sent', **log_details)
                signed_txn = client.web3.eth.account.sign_transaction(transaction_data, client.privkey)
                tx_hash = client.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
                self._available_nonce = Nonce(self._available_nonce + 1)
        except ValueError as e:
            if isinstance(slot.data, SmartContractCall):
                error_msg = 'Transaction to call smart contract function failed'
            elif isinstance(slot.data, EthTransfer):
                error_msg = 'Transaction to transfer ether failed'
            else:
                error_msg = 'Transaction to deploy smart contract failed'
            action = inspect_client_error(e, self.eth_node)
            if action == ClientErrorInspectResult.INSUFFICIENT_FUNDS:
                reason = 'Transaction failed due to insufficient ETH balance. Please top up your ETH account.'
                log.critical(error_msg, **log_details, reason=reason)
                raise InsufficientEth(reason)
            if action == ClientErrorInspectResult.TRANSACTION_UNDERPRICED:
                reason = "Transaction was rejected. This is potentially caused by the reuse of the previous transaction nonce as well as paying an amount of gas less than or equal to the previous transaction's gas amount"
                log.critical(error_msg, **log_details, reason=reason)
                raise ReplacementTransactionUnderpriced(reason)
            if action in THE_NONCE_WAS_REUSED:
                reason = 'Transaction rejected because the nonce has been already mined.'
                log.critical(error_msg, **log_details, reason=reason)
                raise EthereumNonceTooLow(reason)
            reason = f'Unexpected error in underlying Ethereum node: {str(e)}'
            log.critical(error_msg, **log_details, reason=reason)
            raise RaidenUnrecoverableError(reason)
        transaction_sent = TransactionSentImplementation(from_address=slot.from_address, eth_node=slot.eth_node, data=slot.data, extra_log_details=slot.extra_log_details, startgas=slot.startgas, gas_price=slot.gas_price, nonce=slot.nonce, transaction_hash=TransactionHash(tx_hash), chain_id=self.chain_id)
        log.debug('Transaction sent', **transaction_sent.to_log_details())
        return transaction_sent

    def new_contract_proxy(self, abi: List[Any], contract_address: str) -> Any:
        return self.web3.eth.contract(abi=abi, address=contract_address)

    def deploy_single_contract(self, contract_name: str, contract: Dict[str, Any], constructor_parameters: Tuple[Any] = None) -> Tuple[Any, Any]:
        """
        Deploy a single solidity contract without dependencies.

        Args:
            contract_name: The name of the contract to compile.
            contract: The dictionary containing the contract information (like ABI and BIN)
            constructor_parameters: A tuple of arguments to pass to the constructor.
        """
        ctor_parameters = constructor_parameters or ()
        contract_object = self.web3.eth.contract(abi=contract['abi'], bytecode=contract['bin'])
        contract_transaction = contract_object.constructor(*ctor_parameters).buildTransaction()
        constructor_call = ByteCode(contract_name, contract_transaction['data'])
        block = self.get_block(BLOCK_ID_LATEST)
        gas_with_margin = int(contract_transaction['gas'] * 1.5)
        gas_price = gas_price_for_fast_transaction(self.web3)
        transaction = TransactionEstimated(from_address=self.address, data=constructor_call, eth_node=self.eth_node, extra_log_details={}, estimated_gas=gas_with_margin, gas_price=gas_price, approximate_block=(BlockHash(block['hash']), block['number']))
        transaction_sent = self.transact(transaction)
        transaction_mined = self.poll_transaction(transaction_sent)
        maybe_contract_address = transaction_mined.receipt['contractAddress']
        assert maybe_contract_address is not None, "'contractAddress' not set in receipt"
        contract_address = to_canonical_address(maybe_contract_address)
        if not was_transaction_successfully_mined(transaction_mined):
            check_transaction_failure(transaction_mined, self)
            raise RuntimeError(f'Deployment of {contract_name} failed! Most likely a require from the constructor was not satisfied, or there is a compiler bug.')
        deployed_code = self.web3.eth.get_code(contract_address)
        if not deployed_code:
            raise RaidenUnrecoverableError(f'Contract deployment of {contract_name} was successfull but address has no code! This is likely a bug in the ethereum client.')
        return (self.new_contract_proxy(abi=contract['abi'], contract_address=contract_address), transaction_mined.receipt)

    def poll_transaction(self, transaction_sent: Any) -> Any:
        """Wait until the `transaction_hash` is mined, confirmed, handling
        reorgs.

        Consider the following reorg, where a transaction is mined at block B,
        but it is not mined in the canonical chain A-C-D:

             A -> B   D
             *--> C --^

        When the Ethereum node looks at block B, from its perspective the
        transaction is mined and it has a receipt. After the reorg it does not
        have a receipt. This can happen on PoW and PoA based chains.

        Args:
            transaction_hash: Transaction hash that we are waiting for.
        """
        transaction_hash_hex = encode_hex(transaction_sent.transaction_hash)
        while True:
            tx_receipt = None
            try:
                tx_receipt = self.web3.eth.get_transaction_receipt(transaction_hash_hex)
            except TransactionNotFound:
                pass
            is_transaction_mined = tx_receipt and tx_receipt.get('blockNumber') is not None
            if is_transaction_mined:
                assert tx_receipt is not None, MYPY_ANNOTATION
                confirmation_block = tx_receipt['blockNumber'] + self.default_block_num_confirmations
                block_number = self.block_number()
                is_transaction_confirmed = block_number >= confirmation_block
                if is_transaction_confirmed:
                    transaction_mined = TransactionMined(from_address=transaction_sent.from_address, data=transaction_sent.data, eth_node=transaction_sent.eth_node, extra_log_details=transaction_sent.extra_log_details, startgas=transaction_sent.startgas, gas_price=transaction_sent.gas_price, nonce=transaction_sent.nonce, transaction_hash=transaction_sent.transaction_hash, receipt=tx_receipt, chain_id=transaction_sent.chain_id)
                    return transaction_mined
            gevent.sleep(1.0)

    def get_filter_events(self, contract_address: str, topics: List[Any] = None, from_block: Any = GENESIS_BLOCK_NUMBER, to_block: Any = BLOCK_ID_LATEST) -> List[Any]:
        """Get events for the given query."""
        logs_blocks_sanity_check(from_block, to_block)
        return self.web3.eth.get_logs(FilterParams({'fromBlock': from_block, 'toBlock': to_block, 'address': contract_address, 'topics': topics}))

    def check_for_insufficient_eth(self, transaction_name: str, transaction_executed: bool, required_gas: int, block_identifier: Any) -> None:
        """After estimate gas failure checks if our address has enough balance.

        If the account did not have enough ETH balance to execute the
        transaction, it raises an `InsufficientEth` error.

        Note:
            This check contains a race condition, it could be the case that a
            new block is mined changing the account's balance.
            https://github.com/raiden-network/raiden/issues/3890#issuecomment-485857726
        """
        if transaction_executed:
            return
        our_address = to_checksum_address(self.address)
        balance = self.web3.eth.get_balance(our_address, block_identifier)
        required_balance = required_gas * gas_price_for_fast_transaction(self.web3)
        if balance < required_balance:
            msg = f'Failed to execute {transaction_name} due to insufficient ETH'
            log.critical(msg, required_wei=required_balance, actual_wei=balance)
            raise InsufficientEth(msg)

    def wait_until_block(self, target_block_number: int, retry_timeout: float = 0.5) -> int:
        current_block = self.block_number()
        while current_block < target_block_number:
            current_block = self.block_number()
            gevent.sleep(retry_timeout)
        return current_block

    def transaction_failed_with_a_require(self, transaction_hash: Any) -> Optional[bool]:
        """Tries to determine if the transaction with `transaction_hash`
        failed because of a `require` expression.
        """
        if self.eth_node == EthClient.GETH:
            try:
                trace = self.web3.manager.request_blocking(RPCEndpoint('debug_traceTransaction'), [to_hex(transaction_hash), {}])
            except ValueError:
                return None
            return trace.structLogs[-1].op == GETH_REQUIRE_OPCODE
        if self.eth_node == EthClient.PARITY:
            try:
                response = self.web3.manager.request_blocking(RPCEndpoint('trace_replayTransaction'), [to_hex(transaction_hash), ['trace']])
            except ValueError:
                return None
            first_trace = response.trace[0]
            return first_trace['error'] == PARITY_REQUIRE_ERROR
        return None
