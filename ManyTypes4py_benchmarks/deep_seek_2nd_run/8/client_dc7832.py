import json
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union, TypeVar, Generic, cast
from uuid import uuid4
import gevent
import structlog
from eth_typing import ChecksumAddress
from eth_utils import decode_hex, encode_hex, is_bytes, is_checksum_address, to_canonical_address, to_hex
from eth_utils.toolz import assoc
from gevent.lock import Semaphore
from hexbytes import HexBytes
from requests.exceptions import ReadTimeout
from web3 import HTTPProvider, Web3
from web3._utils.contracts import encode_transaction_data, find_matching_fn_abi, prepare_transaction
from web3._utils.empty import empty
from web3.contract import Contract, ContractFunction
from web3.eth import Eth
from web3.exceptions import BlockNotFound, TransactionNotFound
from web3.gas_strategies.rpc import rpc_gas_price_strategy
from web3.middleware import simple_cache_middleware
from web3.types import ABIFunction, BlockData, CallOverrideParams, FilterParams, LogReceipt, RPCEndpoint, RPCResponse, TxParams, TxReceipt, Wei
from raiden.constants import BLOCK_ID_LATEST, BLOCK_ID_PENDING, GENESIS_BLOCK_NUMBER, NO_STATE_QUERY_AFTER_BLOCKS, NULL_ADDRESS_CHECKSUM, RECEIPT_FAILURE_CODE, TRANSACTION_INTRINSIC_GAS, WEB3_BLOCK_NOT_FOUND_RETRY_COUNT, EthClient
from raiden.exceptions import AddressWithoutCode, EthereumNonceTooLow, EthNodeInterfaceError, InsufficientEth, RaidenError, RaidenUnrecoverableError, ReplacementTransactionUnderpriced
from raiden.network.rpc.middleware import block_hash_cache_middleware
from raiden.utils.ethereum_clients import VersionSupport, is_supported_client
from raiden.utils.formatting import to_checksum_address
from raiden.utils.keys import privatekey_to_address
from raiden.utils.smart_contracts import safe_gas_limit
from raiden.utils.typing import ABI, MYPY_ANNOTATION, Address, AddressHex, BlockHash, BlockIdentifier, BlockNumber, CompiledContract, Nonce, PrivateKey, T_Address, T_Nonce, T_TransactionHash, TokenAmount, TransactionHash, typecheck
from raiden_contracts.utils.type_aliases import ChainID, T_ChainID

log = structlog.get_logger(__name__)
GETH_REQUIRE_OPCODE = 'Missing opcode 0xfe'
PARITY_REQUIRE_ERROR = 'Bad instruction'
EXTRA_DATA_LENGTH = 66

def logs_blocks_sanity_check(from_block: Union[int, str], to_block: Union[int, str]) -> None:
    """Checks that the from/to blocks passed onto log calls contain only appropriate types"""
    is_valid_from = isinstance(from_block, int) or isinstance(from_block, str)
    assert is_valid_from, 'event log from block can be integer or latest,pending, earliest'
    is_valid_to = isinstance(to_block, int) or isinstance(to_block, str)
    assert is_valid_to, 'event log to block can be integer or latest,pending, earliest'

def check_transaction_failure(transaction: Any, client: Any) -> None:
    """Raise an exception if the transaction consumed all the gas."""
    if was_transaction_successfully_mined(transaction):
        return
    receipt = transaction.receipt
    gas_used = receipt['gasUsed']
    if gas_used >= transaction.startgas:
        failed_with_require = client.transaction_failed_with_a_require(transaction.transaction_hash)
        if failed_with_require is True:
            if isinstance(transaction.data, SmartContractCall):
                smart_contract_function = transaction.data.function
                msg = f'`{smart_contract_function}` failed because of a require. This looks like a bug in the smart contract.'
            elif isinstance(transaction.data, ByteCode):
                contract_name = transaction.data.contract_name
                msg = f'Deploying {contract_name} failed with a require, this looks like a error detection or compiler bug!'
            else:
                typecheck(transaction.data, EthTransfer)
                msg = 'EthTransfer failed with a require. This looks like a bug in the detection code or in the client reporting!'
        elif failed_with_require is False:
            if isinstance(transaction.data, SmartContractCall):
                smart_contract_function = transaction.data.function
                msg = f'`{smart_contract_function}` failed and all gas was used ({gas_used}), but the last opcode was *not* a failed `require`. This can happen for a few reasons: 1. The smart contract code may have an assert inside an if statement, at the time of gas estimation the condition was false, but another transaction changed the state of the smart contrat making the condition true. 2. The call to `{smart_contract_function}` executes an opcode with variable gas, at the time of gas estimation the cost was low, but another transaction changed the environment so that the new cost is high.  This is particularly problematic storage is set to `0`, since the cost of a `SSTORE` increases 4 times. 3. The cost of the function varies with external state, if the cost increases because of another transaction the transaction can fail.'
            elif isinstance(transaction.data, ByteCode):
                contract_name = transaction.data.contract_name
                msg = f'Deploying {contract_name} failed because all gas was used, this looks like a gas estimation bug!'
            else:
                typecheck(transaction.data, EthTransfer)
                msg = 'EthTransfer failed!'
        elif isinstance(transaction.data, SmartContractCall):
            smart_contract_function = transaction.data.function
            msg = f'`{smart_contract_function}` failed and all gas was used ({gas_used}). This can happen for a few reasons: 1. The smart contract code may have an assert inside an if statement, at the time of gas estimation the condition was false, but another transaction changed the state of the smart contrat making the condition true. 2. The call to `{smart_contract_function}` executes an opcode with variable gas, at the time of gas estimation the cost was low, but another transaction changed the environment so that the new cost is high. This is particularly problematic storage is set to `0`, since the cost of a `SSTORE` increases 4 times. 3. The cost of the function varies with external state, if the cost increases because of another transaction the transaction can fail. 4. There is a bug in thesmart contract and a `require` condition failed.'
        elif isinstance(transaction.data, ByteCode):
            contract_name = transaction.data.contract_name
            msg = f'Deploying {contract_name} failed because all the gas was used!'
        else:
            typecheck(transaction.data, EthTransfer)
            msg = 'EthTransfer failed!'
        if gas_used > transaction.startgas:
            msg = 'The receipt `gasUsed` reported in the receipt is higher than the transaction startgas!.' + msg
        raise RaidenError(msg)

def was_transaction_successfully_mined(transaction: Any) -> bool:
    """`True` if the transaction was successfully mined, `False` otherwise."""
    if 'status' not in transaction.receipt:
        raise AssertionError('Transaction receipt does not contain a status field. Upgrade your client')
    return transaction.receipt['status'] != RECEIPT_FAILURE_CODE

def geth_assert_rpc_interfaces(web3: Web3) -> None:
    try:
        web3.clientVersion
    except ValueError:
        raise EthNodeInterfaceError("The underlying geth node does not have the web3 rpc interface enabled. Please run it with '--http.api eth,net,web3'")
    try:
        web3.eth.block_number
    except ValueError:
        raise EthNodeInterfaceError("The underlying geth node does not have the eth rpc interface enabled. Please run it with '--http.api eth,net,web3'")
    try:
        web3.net.version
    except ValueError:
        raise EthNodeInterfaceError("The underlying geth node does not have the net rpc interface enabled. Please run it with '--http.api eth,net,web3'")

def parity_assert_rpc_interfaces(web3: Web3) -> None:
    try:
        web3.clientVersion
    except ValueError:
        raise EthNodeInterfaceError('The underlying parity node does not have the web3 rpc interface enabled. Please run it with --jsonrpc-apis=eth,net,web3,parity')
    try:
        web3.eth.block_number
    except ValueError:
        raise EthNodeInterfaceError('The underlying parity node does not have the eth rpc interface enabled. Please run it with --jsonrpc-apis=eth,net,web3,parity')
    try:
        web3.net.version
    except ValueError:
        raise EthNodeInterfaceError('The underlying parity node does not have the net rpc interface enabled. Please run it with --jsonrpc-apis=eth,net,web3,parity')
    try:
        web3.manager.request_blocking(RPCEndpoint('parity_nextNonce'), [NULL_ADDRESS_CHECKSUM])
    except ValueError:
        raise EthNodeInterfaceError('The underlying parity node does not have the parity rpc interface enabled. Please run it with --jsonrpc-apis=eth,net,web3,parity')

def parity_discover_next_available_nonce(web3: Web3, address: Address) -> Nonce:
    """Returns the next available nonce for `address`."""
    next_nonce_encoded = web3.manager.request_blocking(RPCEndpoint('parity_nextNonce'), [to_checksum_address(address)])
    return Nonce(int(next_nonce_encoded, 16))

def geth_discover_next_available_nonce(web3: Web3, address: Address) -> Nonce:
    """Returns the next available nonce for `address`."""
    return web3.eth.get_transaction_count(address, BLOCK_ID_PENDING)

def discover_next_available_nonce(web3: Web3, eth_node: EthClient, address: Address) -> Nonce:
    """Returns the next available nonce for `address`."""
    if eth_node is EthClient.PARITY:
        parity_assert_rpc_interfaces(web3)
        available_nonce = parity_discover_next_available_nonce(web3, address)
    elif eth_node is EthClient.GETH:
        geth_assert_rpc_interfaces(web3)
        available_nonce = geth_discover_next_available_nonce(web3, address)
    return available_nonce

def check_address_has_code(client: Any, address: Address, contract_name: str, given_block_identifier: BlockIdentifier) -> None:
    """Checks that the given address contains code."""
    if is_bytes(given_block_identifier):
        assert isinstance(given_block_identifier, bytes), MYPY_ANNOTATION
        block_hash = encode_hex(given_block_identifier)
        given_block_identifier = client.web3.eth.get_block(block_hash)['number']
    result = client.web3.eth.get_code(address, given_block_identifier)
    if not result:
        raise AddressWithoutCode('[{}]Address {} does not contain code'.format(contract_name, to_checksum_address(address)))

def check_address_has_code_handle_pruned_block(client: Any, address: Address, contract_name: str, given_block_identifier: BlockIdentifier) -> None:
    """Checks that the given address contains code."""
    try:
        check_address_has_code(client, address, contract_name, given_block_identifier)
    except ValueError:
        check_address_has_code(client, address, contract_name, 'latest')

def get_transaction_data(web3: Web3, abi: ABI, function_name: str, args: Optional[List[Any]] = None, kwargs: Optional[Dict[str, Any]] = None) -> bytes:
    """Get encoded transaction data"""
    args = args or []
    fn_abi = find_matching_fn_abi(abi=abi, abi_codec=web3.codec, fn_identifier=function_name, args=args, kwargs=kwargs)
    return encode_transaction_data(web3=web3, fn_identifier=function_name, contract_abi=abi, fn_abi=fn_abi, args=args, kwargs=kwargs)

def gas_price_for_fast_transaction(web3: Web3) -> int:
    try:
        maybe_price = web3.eth.generate_gas_price()
        if maybe_price is not None:
            price = int(maybe_price)
        else:
            price = int(web3.eth.gas_price)
    except AttributeError:
        price = int(web3.eth.gas_price)
    except IndexError:
        price = int(web3.eth.gas_price)
    return price

class ClientErrorInspectResult(Enum):
    """Represents the action to follow after inspecting a client exception"""
    PROPAGATE_ERROR = 1
    INSUFFICIENT_FUNDS = 2
    TRANSACTION_UNDERPRICED = 3
    TRANSACTION_PENDING = 4
    ALWAYS_FAIL = 5
    TRANSACTION_ALREADY_IMPORTED = 7
    TRANSACTION_PENDING_OR_ALREADY_IMPORTED = 8

THE_NONCE_WAS_REUSED = (ClientErrorInspectResult.TRANSACTION_PENDING, ClientErrorInspectResult.TRANSACTION_ALREADY_IMPORTED, ClientErrorInspectResult.TRANSACTION_PENDING_OR_ALREADY_IMPORTED)

def inspect_client_error(val_err: ValueError, eth_node: EthClient) -> ClientErrorInspectResult:
    json_response = str(val_err).replace("'", '"').replace('("', '(').replace('")', ')')
    try:
        error = json.loads(json_response)
    except json.JSONDecodeError:
        return ClientErrorInspectResult.PROPAGATE_ERROR
    if eth_node is EthClient.GETH:
        if error['code'] == -32000:
            if 'insufficient funds' in error['message']:
                return ClientErrorInspectResult.INSUFFICIENT_FUNDS
            if 'always failing transaction' in error['message'] or 'execution reverted' in error['message'] or 'invalid opcode: opcode 0xfe not defined' in error['message']:
                return ClientErrorInspectResult.ALWAYS_FAIL
            if 'replacement transaction underpriced' in error['message']:
                return ClientErrorInspectResult.TRANSACTION_UNDERPRICED
            if 'known transaction:' in error['message']:
                return ClientErrorInspectResult.TRANSACTION_PENDING
            if 'already know' in error['message']:
                return ClientErrorInspectResult.TRANSACTION_PENDING
            if 'nonce too low' in error['message']:
                return ClientErrorInspectResult.TRANSACTION_ALREADY_IMPORTED
    elif eth_node is EthClient.PARITY:
        if error['code'] == -32010:
            if 'Insufficient funds' in error['message']:
                return ClientErrorInspectResult.INSUFFICIENT_FUNDS
            if 'another transaction with same nonce in the queue' in error['message']:
                return ClientErrorInspectResult.TRANSACTION_UNDERPRICED
            if 'Transaction nonce is too low. Try incrementing the nonce.' in error['message']:
                return ClientErrorInspectResult.TRANSACTION_PENDING_OR_ALREADY_IMPORTED
            if 'Transaction with the same hash was already imported' in error['message']:
                return ClientErrorInspectResult.TRANSACTION_PENDING_OR_ALREADY_IMPORTED
        elif error['code'] == -32015 and 'Transaction execution error' in error['message']:
            return ClientErrorInspectResult.ALWAYS_FAIL
    return ClientErrorInspectResult.PROPAGATE_ERROR

class CallType(Enum):
    ESTIMATE_GAS = 1
    CALL = 2

def check_value_error(value_error: ValueError, call_type: CallType) -> bool:
    """
    For parity and geth >= v1.9.15, failing calls and functions do not return
    None if the transaction will fail but instead throw a ValueError exception.
    """
    try:
        error_data = json.loads(str(value_error).replace("'", '"'))
    except json.JSONDecodeError:
        return False
    expected_errors = {CallType.ESTIMATE_GAS: [(3, 'execution reverted:'), (-32016, 'The execution failed due to an exception')], CallType.CALL: [(-32000, 'invalid opcode: opcode 0xfe not defined'), (-32000, 'execution reverted'), (-32015, 'VM execution error')]}
    if call_type not in expected_errors:
        raise ValueError('Called check_value_error() with illegal call type')
    for expected_code, expected_msg in expected_errors[call_type]:
        if error_data['code'] == expected_code and expected_msg in error_data['message']:
            return True
    return False

def is_infura(web3: Web3) -> bool:
    return isinstance(web3.provider, HTTPProvider) and web3.provider.endpoint_uri is not None and ('infura.io' in web3.provider.endpoint_uri)

def patched_web3_eth_estimate_gas(self: Eth, transaction: TxParams, block_identifier: Optional[BlockIdentifier] = None) -> Optional[Wei]:
    if 'from' not in transaction and is_checksum_address(self.default_account):
        transaction = assoc(transaction, 'from', self.default_account)
    if block_identifier is None:
        params = [transaction]
    else:
        params = [transaction, block_identifier]
    try:
        result = self.web3.manager.request_blocking(RPCEndpoint('eth_estimateGas'), params)
    except ValueError as e:
        if check_value_error(e, CallType.ESTIMATE_GAS):
            result = None
        else:
            raise e
    except ReadTimeout:
        result = None
    return result

def patched_web3_eth_call