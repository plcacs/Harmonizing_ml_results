import json
from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from uuid import uuid4
import gevent
import structlog
from eth_typing import ChecksumAddress
from eth_utils import (
    decode_hex,
    encode_hex,
    is_bytes,
    is_checksum_address,
    to_canonical_address,
    to_hex,
)
from eth_utils.toolz import assoc
from gevent.lock import Semaphore
from hexbytes import HexBytes
from requests.exceptions import ReadTimeout
from web3 import HTTPProvider, Web3
from web3._utils.contracts import (
    encode_transaction_data,
    find_matching_fn_abi,
    prepare_transaction,
)
from web3._utils.empty import empty
from web3.contract import Contract, ContractFunction
from web3.eth import Eth
from web3.exceptions import BlockNotFound, TransactionNotFound
from web3.gas_strategies.rpc import rpc_gas_price_strategy
from web3.middleware import simple_cache_middleware
from web3.types import (
    ABIFunction,
    BlockData,
    CallOverrideParams,
    FilterParams,
    LogReceipt,
    RPCEndpoint,
    RPCResponse,
    TxParams,
    TxReceipt,
    Wei,
)
from raiden.constants import (
    BLOCK_ID_LATEST,
    BLOCK_ID_PENDING,
    GENESIS_BLOCK_NUMBER,
    NO_STATE_QUERY_AFTER_BLOCKS,
    NULL_ADDRESS_CHECKSUM,
    RECEIPT_FAILURE_CODE,
    TRANSACTION_INTRINSIC_GAS,
    WEB3_BLOCK_NOT_FOUND_RETRY_COUNT,
    EthClient,
)
from raiden.exceptions import (
    AddressWithoutCode,
    EthereumNonceTooLow,
    EthNodeInterfaceError,
    InsufficientEth,
    RaidenError,
    RaidenUnrecoverableError,
    ReplacementTransactionUnderpriced,
)
from raiden.network.rpc.middleware import block_hash_cache_middleware
from raiden.utils.ethereum_clients import VersionSupport, is_supported_client
from raiden.utils.formatting import to_checksum_address
from raiden.utils.keys import privatekey_to_address
from raiden.utils.smart_contracts import safe_gas_limit
from raiden.utils.typing import (
    ABI,
    MYPY_ANNOTATION,
    Address,
    AddressHex,
    BlockHash,
    BlockIdentifier,
    BlockNumber,
    CompiledContract,
    Nonce,
    PrivateKey,
    T_Address,
    T_Nonce,
    T_TransactionHash,
    TokenAmount,
    TransactionHash,
    typecheck,
)
from raiden_contracts.utils.type_aliases import ChainID, T_ChainID

log = structlog.get_logger(__name__)
GETH_REQUIRE_OPCODE = "Missing opcode 0xfe"
PARITY_REQUIRE_ERROR = "Bad instruction"
EXTRA_DATA_LENGTH = 66


def logs_blocks_sanity_check(
    from_block: Union[int, str], to_block: Union[int, str]
) -> None:
    """Checks that the from/to blocks passed onto log calls contain only appropriate types"""
    is_valid_from = isinstance(from_block, int) or isinstance(from_block, str)
    assert is_valid_from, "event log from block can be integer or latest,pending, earliest"
    is_valid_to = isinstance(to_block, int) or isinstance(to_block, str)
    assert is_valid_to, "event log to block can be integer or latest,pending, earliest"


def check_transaction_failure(transaction: "TransactionMined", client: "JSONRPCClient") -> None:
    """Raise an exception if the transaction consumed all the gas."""
    if was_transaction_successfully_mined(transaction):
        return
    receipt: TxReceipt = transaction.receipt
    gas_used: int = receipt["gasUsed"]
    if gas_used >= transaction.startgas:
        failed_with_require: Optional[bool] = client.transaction_failed_with_a_require(
            transaction.transaction_hash
        )
        if failed_with_require is True:
            if isinstance(transaction.data, SmartContractCall):
                smart_contract_function: str = transaction.data.function
                msg = (
                    f"`{smart_contract_function}` failed because of a require. This looks like a bug in the smart contract."
                )
            elif isinstance(transaction.data, ByteCode):
                contract_name: str = transaction.data.contract_name
                msg = (
                    f"Deploying {contract_name} failed with a require, this looks like a error detection or compiler bug!"
                )
            else:
                typecheck(transaction.data, EthTransfer)
                msg = (
                    "EthTransfer failed with a require. This looks like a bug in the detection code or in the client reporting!"
                )
        elif failed_with_require is False:
            if isinstance(transaction.data, SmartContractCall):
                smart_contract_function: str = transaction.data.function
                msg = (
                    f"`{smart_contract_function}` failed and all gas was used ({gas_used}), but the last opcode was *not* a failed `require`. This can happen for a few reasons: 1. The smart contract code may have an assert inside an if statement, at the time of gas estimation the condition was false, but another transaction changed the state of the smart contrat making the condition true. 2. The call to `{smart_contract_function}` executes an opcode with variable gas, at the time of gas estimation the cost was low, but another transaction changed the environment so that the new cost is high.  This is particularly problematic storage is set to `0`, since the cost of a `SSTORE` increases 4 times. 3. The cost of the function varies with external state, if the cost increases because of another transaction the transaction can fail."
                )
            elif isinstance(transaction.data, ByteCode):
                contract_name: str = transaction.data.contract_name
                msg = (
                    f"Deploying {contract_name} failed because all gas was used, this looks like a gas estimation bug!"
                )
            else:
                typecheck(transaction.data, EthTransfer)
                msg = "EthTransfer failed!"
        elif isinstance(transaction.data, SmartContractCall):
            smart_contract_function: str = transaction.data.function
            msg = (
                f"`{smart_contract_function}` failed and all gas was used ({gas_used}). This can happen for a few reasons: 1. The smart contract code may have an assert inside an if statement, at the time of gas estimation the condition was false, but another transaction changed the state of the smart contrat making the condition true. 2. The call to `{smart_contract_function}` executes an opcode with variable gas, at the time of gas estimation the cost was low, but another transaction changed the environment so that the new cost is high. This is particularly problematic storage is set to `0`, since the cost of a `SSTORE` increases 4 times. 3. The cost of the function varies with external state, if the cost increases because of another transaction the transaction can fail. 4. There is a bug in thesmart contract and a `require` condition failed."
            )
        elif isinstance(transaction.data, ByteCode):
            contract_name: str = transaction.data.contract_name
            msg = f"Deploying {contract_name} failed because all the gas was used!"
        else:
            typecheck(transaction.data, EthTransfer)
            msg = "EthTransfer failed!"
        if gas_used > transaction.startgas:
            msg = (
                "The receipt `gasUsed` reported in the receipt is higher than the transaction startgas!."
                + msg
            )
        raise RaidenError(msg)


def was_transaction_successfully_mined(transaction: "TransactionMined") -> bool:
    """`True` if the transaction was successfully mined, `False` otherwise."""
    if "status" not in transaction.receipt:
        raise AssertionError("Transaction receipt does not contain a status field. Upgrade your client")
    return transaction.receipt["status"] != RECEIPT_FAILURE_CODE


def geth_assert_rpc_interfaces(web3: Web3) -> None:
    try:
        web3.clientVersion  # type: ignore
    except ValueError:
        raise EthNodeInterfaceError(
            "The underlying geth node does not have the web3 rpc interface enabled. Please run it with '--http.api eth,net,web3'"
        )
    try:
        web3.eth.block_number  # type: ignore
    except ValueError:
        raise EthNodeInterfaceError(
            "The underlying geth node does not have the eth rpc interface enabled. Please run it with '--http.api eth,net,web3'"
        )
    try:
        web3.net.version  # type: ignore
    except ValueError:
        raise EthNodeInterfaceError(
            "The underlying geth node does not have the net rpc interface enabled. Please run it with '--http.api eth,net,web3'"
        )


def parity_assert_rpc_interfaces(web3: Web3) -> None:
    try:
        web3.clientVersion  # type: ignore
    except ValueError:
        raise EthNodeInterfaceError(
            "The underlying parity node does not have the web3 rpc interface enabled. Please run it with --jsonrpc-apis=eth,net,web3,parity"
        )
    try:
        web3.eth.block_number  # type: ignore
    except ValueError:
        raise EthNodeInterfaceError(
            "The underlying parity node does not have the eth rpc interface enabled. Please run it with --jsonrpc-apis=eth,net,web3,parity"
        )
    try:
        web3.net.version  # type: ignore
    except ValueError:
        raise EthNodeInterfaceError(
            "The underlying parity node does not have the net rpc interface enabled. Please run it with --jsonrpc-apis=eth,net,web3,parity"
        )
    try:
        web3.manager.request_blocking(RPCEndpoint("parity_nextNonce"), [NULL_ADDRESS_CHECKSUM])  # type: ignore
    except ValueError:
        raise EthNodeInterfaceError(
            "The underlying parity node does not have the parity rpc interface enabled. Please run it with --jsonrpc-apis=eth,net,web3,parity"
        )


def parity_discover_next_available_nonce(web3: Web3, address: Address) -> Nonce:
    """Returns the next available nonce for `address`."""
    next_nonce_encoded: str = web3.manager.request_blocking(RPCEndpoint("parity_nextNonce"), [to_checksum_address(address)])  # type: ignore
    return Nonce(int(next_nonce_encoded, 16))


def geth_discover_next_available_nonce(web3: Web3, address: Address) -> Nonce:
    """Returns the next available nonce for `address`."""
    return Nonce(web3.eth.get_transaction_count(address, BLOCK_ID_PENDING))  # type: ignore


def discover_next_available_nonce(web3: Web3, eth_node: EthClient, address: Address) -> Nonce:
    """Returns the next available nonce for `address`.

    Nonce discovery had a few iterations:

    - Initially the client fetched the next avaialble nonce before sending
      every transaction.
    - Commit e4edcde0ce tried to handle race conditions were two concurrent
      *local* transactions would use the same nonce, since the nonce was
      fecthed without synchronization. The strategy was to query it often and
      update the local state, with the addition of a lock to synchronize its
      usage.
    - Commit f750fbd7f7 removed the regular updates, it didn't fix race
      conditions for *multiple* clients or restarts, and made a requirement
      that only one app can use the private key at the time. This removed
      remote race conditions.
    - Commit d3d204022e started using the txpool interface to account for all
      sent transactions.
    - Commit 83f3d8f3bd simplified the nonce update since Geth's API was fixed.
      And added the special case for Parity.
    """
    if eth_node is EthClient.PARITY:
        parity_assert_rpc_interfaces(web3)
        available_nonce = parity_discover_next_available_nonce(web3, address)
    elif eth_node is EthClient.GETH:
        geth_assert_rpc_interfaces(web3)
        available_nonce = geth_discover_next_available_nonce(web3, address)
    else:
        raise EthNodeInterfaceError(f"Unsupported Ethereum client: {eth_node}")
    return available_nonce


def check_address_has_code(
    client: "JSONRPCClient", address: Address, contract_name: str, given_block_identifier: BlockIdentifier
) -> None:
    """Checks that the given address contains code.

    Use this function to detect errors prior to sending transactions, which is
    faster to interact and easier to debug. These are the problem that can be
    prevented:

    - Ensure the target address indeed has code. Otherwise transaction would
      fail to execute.
    - Ensure the smart contract exists for the expected time frame. This can be
      important to verify the metadata is correct, which is necessary to reduce
      the range which events are queried, and also to detect bugs were a piece
      of code tries to interact with a smart contract that has not finished
      deploying.
    - If possible, check the bytecode of the target smart contract. This is
      important because it will validate the smart contract ABI (assuming the
      metadata is correct). Invalid ABI lead to subtle hard to debug bugs.
    """
    if is_bytes(given_block_identifier):
        assert isinstance(given_block_identifier, bytes), MYPY_ANNOTATION
        block_hash: str = encode_hex(given_block_identifier)
        given_block_identifier = client.web3.eth.get_block(block_hash)["number"]  # type: ignore
    result: HexBytes = client.web3.eth.get_code(address, given_block_identifier)  # type: ignore
    if not result:
        raise AddressWithoutCode(
            f"[{contract_name}]Address {to_checksum_address(address)} does not contain code"
        )


def check_address_has_code_handle_pruned_block(
    client: "JSONRPCClient", address: Address, contract_name: str, given_block_identifier: BlockIdentifier
) -> None:
    """Checks that the given address contains code.

    If `given_block_identifier` points to a pruned block, fallbacks to use
    `latest` instead.
    """
    try:
        check_address_has_code(client, address, contract_name, given_block_identifier)
    except ValueError:
        check_address_has_code(client, address, contract_name, "latest")


def get_transaction_data(
    web3: Web3,
    abi: ABI,
    function_name: str,
    args: Optional[Sequence[Any]] = None,
    kwargs: Optional[Dict[str, Any]] = None,
) -> bytes:
    """Get encoded transaction data"""
    args = args or []
    fn_abi: ABIFunction = find_matching_fn_abi(
        abi=abi,
        abi_codec=web3.codec,  # type: ignore
        fn_identifier=function_name,
        args=args,
        kwargs=kwargs,
    )
    return encode_transaction_data(
        web3=web3,  # type: ignore
        fn_identifier=function_name,
        contract_abi=abi,
        fn_abi=fn_abi,
        args=args,
        kwargs=kwargs,
    )


def gas_price_for_fast_transaction(web3: Web3) -> int:
    try:
        maybe_price = web3.eth.generate_gas_price()  # type: ignore
        if maybe_price is not None:
            price = int(maybe_price)
        else:
            price = int(web3.eth.gas_price)  # type: ignore
    except AttributeError:
        price = int(web3.eth.gas_price)  # type: ignore
    except IndexError:
        price = int(web3.eth.gas_price)  # type: ignore
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


THE_NONCE_WAS_REUSED = (
    ClientErrorInspectResult.TRANSACTION_PENDING,
    ClientErrorInspectResult.TRANSACTION_ALREADY_IMPORTED,
    ClientErrorInspectResult.TRANSACTION_PENDING_OR_ALREADY_IMPORTED,
)


def inspect_client_error(
    val_err: ValueError, eth_node: EthClient
) -> ClientErrorInspectResult:
    json_response: str = str(val_err).replace("'", '"').replace('("', "(").replace('")', ")")
    try:
        error: Dict[str, Any] = json.loads(json_response)
    except json.JSONDecodeError:
        return ClientErrorInspectResult.PROPAGATE_ERROR
    if eth_node is EthClient.GETH:
        if error.get("code") == -32000:
            message: str = error.get("message", "")
            if "insufficient funds" in message:
                return ClientErrorInspectResult.INSUFFICIENT_FUNDS
            if (
                "always failing transaction" in message
                or "execution reverted" in message
                or "invalid opcode: opcode 0xfe not defined" in message
            ):
                return ClientErrorInspectResult.ALWAYS_FAIL
            if "replacement transaction underpriced" in message:
                return ClientErrorInspectResult.TRANSACTION_UNDERPRICED
            if "known transaction:" in message:
                return ClientErrorInspectResult.TRANSACTION_PENDING
            if "already know" in message:
                return ClientErrorInspectResult.TRANSACTION_PENDING
            if "nonce too low" in message:
                return ClientErrorInspectResult.TRANSACTION_ALREADY_IMPORTED
    elif eth_node is EthClient.PARITY:
        if error.get("code") == -32010:
            message: str = error.get("message", "")
            if "Insufficient funds" in message:
                return ClientErrorInspectResult.INSUFFICIENT_FUNDS
            if "another transaction with same nonce in the queue" in message:
                return ClientErrorInspectResult.TRANSACTION_UNDERPRICED
            if "Transaction nonce is too low. Try incrementing the nonce." in message:
                return ClientErrorInspectResult.TRANSACTION_PENDING_OR_ALREADY_IMPORTED
            if "Transaction with the same hash was already imported" in message:
                return ClientErrorInspectResult.TRANSACTION_PENDING_OR_ALREADY_IMPORTED
        elif error.get("code") == -32015 and "Transaction execution error" in error.get("message", ""):
            return ClientErrorInspectResult.ALWAYS_FAIL
    return ClientErrorInspectResult.PROPAGATE_ERROR


class CallType(Enum):
    ESTIMATE_GAS = 1
    CALL = 2


def check_value_error(value_error: ValueError, call_type: CallType) -> bool:
    """
    For parity and geth >= v1.9.15, failing calls and functions do not return
    None if the transaction will fail but instead throw a ValueError exception.

    This function checks the thrown exception to see if it's the correct one and
    if yes returns True, if not returns False
    """
    try:
        error_data: Dict[str, Any] = json.loads(str(value_error).replace("'", '"'))
    except json.JSONDecodeError:
        return False
    expected_errors: Dict[CallType, List[Tuple[int, str]]] = {
        CallType.ESTIMATE_GAS: [
            (3, "execution reverted:"),
            (-32016, "The execution failed due to an exception"),
        ],
        CallType.CALL: [
            (-32000, "invalid opcode: opcode 0xfe not defined"),
            (-32000, "execution reverted"),
            (-32015, "VM execution error"),
        ],
    }
    if call_type not in expected_errors:
        raise ValueError("Called check_value_error() with illegal call type")
    for expected_code, expected_msg in expected_errors[call_type]:
        if error_data.get("code") == expected_code and expected_msg in error_data.get("message", ""):
            return True
    return False


def is_infura(web3: Web3) -> bool:
    return (
        isinstance(web3.provider, HTTPProvider)
        and web3.provider.endpoint_uri is not None
        and "infura.io" in web3.provider.endpoint_uri
    )


def patched_web3_eth_estimate_gas(
    self: Eth, transaction: Dict[str, Any], block_identifier: Optional[BlockIdentifier] = None
) -> Optional[int]:
    if "from" not in transaction and is_checksum_address(self.default_account):  # type: ignore
        transaction = assoc(transaction, "from", self.default_account)  # type: ignore
    if block_identifier is None:
        params: List[Dict[str, Any]] = [transaction]
    else:
        params = [transaction, block_identifier]
    try:
        result: int = self.web3.manager.request_blocking(RPCEndpoint("eth_estimateGas"), params)  # type: ignore
    except ValueError as e:
        if check_value_error(e, CallType.ESTIMATE_GAS):
            result = None
        else:
            raise e
    except ReadTimeout:
        result = None
    return result


def patched_web3_eth_call(
    self: Eth,
    transaction: Dict[str, Any],
    block_identifier: Optional[BlockIdentifier] = None,
    state_override: Optional[CallOverrideParams] = None,
) -> HexBytes:
    if "from" not in transaction and is_checksum_address(self.default_account):  # type: ignore
        transaction = assoc(transaction, "from", self.default_account)  # type: ignore
    if block_identifier is None:
        block_identifier = self.defaultBlock  # type: ignore
    args: List[Any] = [transaction, block_identifier]
    if state_override is not None:
        args.append(state_override)
    try:
        result: str = self.web3.manager.request_blocking(RPCEndpoint("eth_call"), args)  # type: ignore
    except ValueError as e:
        if check_value_error(e, CallType.CALL):
            result = ""
        else:
            raise e
    return HexBytes(result)


def estimate_gas_for_function(
    address: Address,
    web3: Web3,
    fn_identifier: str,
    transaction: Optional[Dict[str, Any]] = None,
    contract_abi: Optional[ABI] = None,
    fn_abi: Optional[ABIFunction] = None,
    block_identifier: Optional[BlockIdentifier] = None,
    *args: Any,
    **kwargs: Any,
) -> Wei:
    """Temporary workaround until next web3.py release (5.X.X)"""
    estimate_transaction = prepare_transaction(
        address=to_checksum_address(address),
        web3=web3,
        fn_identifier=fn_identifier,
        contract_abi=contract_abi,
        fn_abi=fn_abi,
        transaction=transaction,
        fn_args=args,
        fn_kwargs=kwargs,
    )
    try:
        gas_estimate: int = web3.eth.estimateGas(estimate_transaction, block_identifier)  # type: ignore
    except ValueError as e:
        if check_value_error(e, CallType.ESTIMATE_GAS):
            gas_estimate = 0
        else:
            raise e
    return Wei(gas_estimate)


def patched_contractfunction_estimateGas(
    self: ContractFunction,
    transaction: Optional[Dict[str, Any]] = None,
    block_identifier: Optional[BlockIdentifier] = None,
) -> Wei:
    if transaction is None:
        estimate_gas_transaction: Dict[str, Any] = {}
    else:
        estimate_gas_transaction = transaction
    if "data" in estimate_gas_transaction:
        raise ValueError("Cannot set data in estimateGas transaction")
    if "to" in estimate_gas_transaction:
        raise ValueError("Cannot set to in estimateGas transaction")
    if self.address:
        estimate_gas_transaction.setdefault("to", self.address)  # type: ignore
    if self.web3.eth.default_account is not empty:  # type: ignore
        estimate_gas_transaction.setdefault("from", self.web3.eth.default_account)  # type: ignore
    if "to" not in estimate_gas_transaction:
        if isinstance(self, type):
            raise ValueError(
                "When using `Contract.estimateGas` from a contract factory you must provide a `to` address with the transaction"
            )
        else:
            raise ValueError("Please ensure that this contract instance has an address.")
    return estimate_gas_for_function(
        self.address,  # type: ignore
        self.web3,  # type: ignore
        self.function_identifier,  # type: ignore
        estimate_gas_transaction,
        self.contract_abi,  # type: ignore
        self.abi,  # type: ignore
        block_identifier,
        *self.args,  # type: ignore
        **self.kwargs,  # type: ignore
    )


def make_sane_poa_middleware(
    make_request: Callable[[str, Any], RPCResponse], web3: Web3
) -> Callable[[str, Any], RPCResponse]:
    """Simpler geth_poa_middleware that doesn't break with ``null`` responses."""

    def middleware(method: str, params: Any) -> RPCResponse:
        response: RPCResponse = make_request(method, params)
        result = response.get("result")
        is_get_block_poa: bool = (
            method.startswith("eth_getBlockBy")
            and result is not None
            and (len(result["extraData"]) != EXTRA_DATA_LENGTH)
        )
        if is_get_block_poa:
            extra_data = result.pop("extraData")
            response["result"] = {**result, "proofOfAuthorityData": HexBytes(extra_data)}
        return response

    return middleware


def make_patched_web3_get_block(original_func: Callable[..., BlockData]) -> Callable[..., BlockData]:
    """Patch Eth.get_block() to retry in case of ``BlockNotFound``

    Infura sometimes erroneously returns a `null` response for
    ``eth_getBlockByNumber`` and ``eth_getBlockByHash`` for existing blocks.

    This generates a wrapper method that tries to perform the request up to
    ``WEB3_BLOCK_NOT_FOUND_RETRY_COUNT`` times.

    If no result is returned after the final retry the last ``BlockNotFound`` exception is
    re-raised.

    See:
      - https://github.com/raiden-network/raiden/issues/3201
      - https://github.com/INFURA/infura/issues/43
    """

    def patched_web3_get_block(block_identifier: BlockIdentifier, full_transactions: bool = False) -> BlockData:
        last_ex: Optional[BlockNotFound] = None
        for remaining_retries in range(WEB3_BLOCK_NOT_FOUND_RETRY_COUNT, 0, -1):
            try:
                return original_func(block_identifier, full_transactions)
            except BlockNotFound as ex:
                log.warning(
                    "Block not found, retrying",
                    remaining_retries=remaining_retries - 1,
                    block_identifier=block_identifier,
                )
                last_ex = ex
                gevent.sleep(0.1)
        assert last_ex, "Retries can only happen due to exceptions"
        raise last_ex

    return patched_web3_get_block


def monkey_patch_web3(web3: Web3, gas_price_strategy: Callable[..., Any]) -> None:
    try:
        web3.middleware_onion.add(block_hash_cache_middleware)  # type: ignore
        web3.middleware_onion.add(simple_cache_middleware)  # type: ignore
        web3.eth.set_gas_price_strategy(gas_price_strategy)  # type: ignore
        web3.middleware_onion.inject(make_sane_poa_middleware, layer=0)  # type: ignore
    except ValueError:
        pass
    ContractFunction.estimateGas = patched_contractfunction_estimateGas
    Eth.estimateGas = patched_web3_eth_estimate_gas  # type: ignore
    Eth.call = patched_web3_eth_call  # type: ignore
    if is_infura(web3):
        web3.eth.get_block = make_patched_web3_get_block(web3.eth.get_block)  # type: ignore


@dataclass
class EthTransfer:
    to_address: Address
    gas_price: int
    value: int

    def __post_init__(self) -> None:
        typecheck(self.to_address, T_Address)
        typecheck(self.gas_price, int)
        typecheck(self.value, int)

    def to_log_details(self) -> Dict[str, Any]:
        return {
            "to_address": to_checksum_address(self.to_address),
            "value": self.value,
            "gas_price": self.gas_price,
        }


@dataclass
class SmartContractCall:
    contract: Contract
    function: str
    args: Sequence[Any]
    kwargs: Dict[str, Any]
    value: int

    def __post_init__(self) -> None:
        typecheck(self.contract, Contract)
        typecheck(self.function, str)
        typecheck(self.value, int)

    def to_log_details(self) -> Dict[str, Any]:
        to_address: str = to_checksum_address(self.contract.address)  # type: ignore
        return {
            "function_name": self.function,
            "to_address": to_address,
            "args": self.args,
            "kwargs": self.kwargs,
            "value": self.value,
        }


@dataclass
class ByteCode:
    contract_name: str
    bytecode: str

    def to_log_details(self) -> Dict[str, Any]:
        return {"contract_name": self.contract_name}


@dataclass
class TransactionPending:
    from_address: Address
    data: Union[SmartContractCall, ByteCode, EthTransfer]
    eth_node: EthClient
    extra_log_details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        typecheck(self.from_address, T_Address)
        typecheck(self.data, (SmartContractCall, ByteCode, EthTransfer))
        self.extra_log_details.setdefault("token", str(uuid4()))
        log.debug("Transaction created", **self.to_log_details())

    def to_log_details(self) -> Dict[str, Any]:
        log_details: Dict[str, Any] = self.data.to_log_details()
        log_details.update(self.extra_log_details)
        log_details.update(
            {
                "from_address": to_checksum_address(self.from_address),
                "eth_node": self.eth_node,
            }
        )
        return log_details

    def estimate_gas(self, block_identifier: BlockIdentifier) -> Optional["TransactionEstimated"]:
        """Estimate the gas and price necessary to run the transaction.

        Returns `None` transaction would fail because it hit an assert/require,
        or if the amount of gas required is larger than the block gas limit.
        """
        fn = getattr(self.data.contract.functions, self.data.function)  # type: ignore
        from_address: str = to_checksum_address(self.from_address)
        if self.eth_node is EthClient.GETH:
            block_identifier = None
        try:
            estimated_gas: Optional[int] = fn(*self.data.args, **self.data.kwargs).estimateGas(  # type: ignore
                transaction={"from": from_address}, block_identifier=block_identifier
            )
        except ValueError as err:
            estimated_gas = None
            inspected_error: ClientErrorInspectResult = inspect_client_error(err, self.eth_node)
            expected_error: bool = inspected_error in (
                ClientErrorInspectResult.INSUFFICIENT_FUNDS,
                ClientErrorInspectResult.ALWAYS_FAIL,
            )
            if not expected_error:
                raise err
        block: BlockData = self.data.contract.web3.eth.get_block(BLOCK_ID_LATEST)  # type: ignore
        if estimated_gas is not None:
            gas_price: int = gas_price_for_fast_transaction(self.data.contract.web3)  # type: ignore
            transaction_estimated: "TransactionEstimated" = TransactionEstimated(
                from_address=self.from_address,
                eth_node=self.eth_node,
                data=self.data,
                extra_log_details=self.extra_log_details,
                estimated_gas=safe_gas_limit(estimated_gas),
                gas_price=gas_price,
                approximate_block=(BlockHash(block["hash"]), BlockNumber(block["number"])),
            )
            log.debug(
                "Transaction gas estimated",
                **transaction_estimated.to_log_details(),
                node_gas_price=self.data.contract.web3.eth.gas_price,  # type: ignore
            )
            return transaction_estimated
        else:
            log.debug(
                "Transaction gas estimation failed",
                approximate_block_hash=to_hex(block["hash"]),
                approximate_block_number=block["number"],
                **self.to_log_details(),
            )
            return None


@dataclass
class TransactionEstimated:
    from_address: Address
    data: Union[SmartContractCall, ByteCode]
    eth_node: EthClient
    extra_log_details: Dict[str, Any] = field(default_factory=dict)
    estimated_gas: int = 0
    gas_price: int = 0
    approximate_block: Tuple[BlockHash, BlockNumber] = (b"", 0)

    def __post_init__(self) -> None:
        self.extra_log_details.setdefault("token", str(uuid4()))
        typecheck(self.from_address, T_Address)
        typecheck(self.data, (SmartContractCall, ByteCode))
        typecheck(self.estimated_gas, int)
        typecheck(self.gas_price, int)

    def to_log_details(self) -> Dict[str, Any]:
        log_details: Dict[str, Any] = self.data.to_log_details()
        log_details.update(self.extra_log_details)
        log_details.update(
            {
                "from_address": to_checksum_address(self.from_address),
                "eth_node": self.eth_node,
                "estimated_gas": self.estimated_gas,
                "gas_price": self.gas_price,
                "approximate_block_hash": to_hex(self.approximate_block[0]),
                "approximate_block_number": self.approximate_block[1],
            }
        )
        return log_details


class TransactionSent(ABC):
    pass


@dataclass
class TransactionMined:
    from_address: Address
    data: Union[SmartContractCall, ByteCode, EthTransfer]
    eth_node: EthClient
    extra_log_details: Dict[str, Any]
    startgas: int
    gas_price: int
    nonce: Nonce
    transaction_hash: TransactionHash
    receipt: TxReceipt
    chain_id: ChainID

    # Assuming these fields are populated when TransactionMined is created
    # If not, they should be added appropriately


@dataclass
class TransactionSlot:
    from_address: Address
    eth_node: EthClient
    data: Union[SmartContractCall, ByteCode, EthTransfer]
    extra_log_details: Dict[str, Any]
    startgas: int
    gas_price: int
    nonce: Nonce

    def __post_init__(self) -> None:
        self.extra_log_details.setdefault("token", str(uuid4()))
        typecheck(self.from_address, T_Address)
        typecheck(self.data, (SmartContractCall, ByteCode, EthTransfer))
        typecheck(self.startgas, int)
        typecheck(self.gas_price, int)
        typecheck(self.nonce, T_Nonce)

    def to_log_details(self) -> Dict[str, Any]:
        log_details: Dict[str, Any] = self.data.to_log_details()
        log_details.update(self.extra_log_details)
        log_details.update(
            {
                "node": to_checksum_address(self.from_address),
                "from_address": to_checksum_address(self.from_address),
                "eth_node": self.eth_node,
                "startgas": self.startgas,
                "gas_price": self.gas_price,
                "nonce": self.nonce,
            }
        )
        return log_details


@dataclass
class TransactionSentImplementation(TransactionSent):
    from_address: Address
    data: Union[SmartContractCall, ByteCode, EthTransfer]
    eth_node: EthClient
    extra_log_details: Dict[str, Any]
    startgas: int
    gas_price: int
    nonce: Nonce
    transaction_hash: TransactionHash
    chain_id: ChainID

    def __post_init__(self) -> None:
        self.extra_log_details.setdefault("token", str(uuid4()))
        typecheck(self.from_address, T_Address)
        typecheck(self.data, (SmartContractCall, ByteCode, EthTransfer))
        typecheck(self.startgas, int)
        typecheck(self.gas_price, int)
        typecheck(self.nonce, T_Nonce)
        typecheck(self.transaction_hash, T_TransactionHash)
        typecheck(self.chain_id, T_ChainID)

    def to_log_details(self) -> Dict[str, Any]:
        log_details: Dict[str, Any] = self.data.to_log_details()
        log_details.update(self.extra_log_details)
        log_details.update(
            {
                "node": to_checksum_address(self.from_address),
                "from_address": to_checksum_address(self.from_address),
                "eth_node": self.eth_node,
                "startgas": self.startgas,
                "gas_price": self.gas_price,
                "nonce": self.nonce,
                "transaction_hash": encode_hex(self.transaction_hash),
                "chain_id": self.chain_id,
            }
        )
        return log_details


class JSONRPCClient:
    """Ethereum JSON RPC client."""

    def __init__(
        self,
        web3: Web3,
        privkey: PrivateKey,
        gas_price_strategy: Optional[Callable[..., Any]] = rpc_gas_price_strategy,
        block_num_confirmations: int = 0,
    ) -> None:
        if len(privkey) != 32:
            raise ValueError("Invalid private key")
        if block_num_confirmations < 0:
            raise ValueError("Number of confirmations has to be positive")
        if gas_price_strategy is not None:
            monkey_patch_web3(web3, gas_price_strategy)
        else:
            monkey_patch_web3(web3, rpc_gas_price_strategy)
        version: str = web3.clientVersion  # type: ignore
        supported, eth_node, _ = is_supported_client(version)
        if eth_node is None or supported is VersionSupport.UNSUPPORTED:
            raise EthNodeInterfaceError(f'Unsupported Ethereum client "{version}"')
        if supported is VersionSupport.WARN:
            log.warning(f'Unsupported Ethereum client version "{version}"')
        address: Address = privatekey_to_address(privkey)
        available_nonce: Nonce = discover_next_available_nonce(web3, eth_node, address)
        self.eth_node: EthClient = eth_node
        self.privkey: PrivateKey = privkey
        self.address: Address = address
        self.web3: Web3 = web3
        self.default_block_num_confirmations: int = block_num_confirmations
        self.chain_id: ChainID = ChainID(self.web3.eth.chain_id)  # type: ignore
        self._available_nonce: Nonce = available_nonce
        self._nonce_lock: Semaphore = Semaphore()
        log.debug(
            "JSONRPCClient created",
            node=to_checksum_address(self.address),
            available_nonce=available_nonce,
            client=version,
        )

    def __repr__(self) -> str:
        return f"<JSONRPCClient node:{to_checksum_address(self.address)} nonce:{self._available_nonce}>"

    def block_number(self) -> int:
        """Return the most recent block."""
        return self.web3.eth.block_number  # type: ignore

    def get_block(self, block_identifier: BlockIdentifier) -> BlockData:
        """Given a block number, query the chain to get its corresponding block hash"""
        return self.web3.eth.get_block(block_identifier)  # type: ignore

    def _sync_nonce(self) -> None:
        self._available_nonce = discover_next_available_nonce(
            self.web3, self.eth_node, self.address
        )

    def get_confirmed_blockhash(self) -> BlockHash:
        """Gets the block CONFIRMATION_BLOCKS in the past and returns its block hash"""
        confirmed_block_number: BlockNumber = BlockNumber(
            self.web3.eth.block_number - self.default_block_num_confirmations  # type: ignore
        )
        if confirmed_block_number < 0:
            confirmed_block_number = BlockNumber(0)
        return self.blockhash_from_blocknumber(confirmed_block_number)

    def blockhash_from_blocknumber(self, block_number: BlockNumber) -> BlockHash:
        """Given a block number, query the chain to get its corresponding block hash"""
        block: BlockData = self.get_block(block_number)
        return BlockHash(bytes(block["hash"]))

    def can_query_state_for_block(self, block_identifier: BlockIdentifier) -> bool:
        """
        Returns if the provided block identifier is safe enough to query chain
        state for. If it's close to the state pruning blocks then state should
        not be queried.
        More info: https://github.com/raiden-network/raiden/issues/3566.
        """
        latest_block_number: int = self.block_number()
        preconditions_block: BlockData = self.web3.eth.get_block(block_identifier)  # type: ignore
        preconditions_block_number: int = int(preconditions_block["number"])
        difference: int = latest_block_number - preconditions_block_number
        return difference < NO_STATE_QUERY_AFTER_BLOCKS

    def balance(self, account: Address) -> TokenAmount:
        """Return the balance of the account of the given address."""
        return TokenAmount(self.web3.eth.get_balance(account, BLOCK_ID_PENDING))  # type: ignore

    def parity_get_pending_transaction_hash_by_nonce(
        self, address: Address, nonce: Nonce
    ) -> Optional[TransactionHash]:
        """Queries the local parity transaction pool and searches for a transaction.

        Checks the local tx pool for a transaction from a particular address and for
        a given nonce. If it exists it returns the transaction hash.
        """
        msg: str = (
            "`parity` specific function must only be called when the client is parity. "
            f"Client was {self.eth_node}."
        )
        assert self.eth_node is EthClient.PARITY, msg
        transactions: List[Dict[str, Any]] = self.web3.manager.request_blocking(  # type: ignore
            RPCEndpoint("parity_allTransactions"), []
        )
        log.debug("RETURNED TRANSACTIONS", transactions=transactions)
        for tx in transactions:
            address_match: bool = tx["from"] == address
            if address_match and int(tx["nonce"], 16) == nonce:
                return TransactionHash(tx["hash"])
        return None

    def estimate_gas(
        self,
        contract: Contract,
        function: str,
        extra_log_details: Dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> Optional["TransactionEstimated"]:
        pending: TransactionPending = TransactionPending(
            from_address=self.address,
            data=SmartContractCall(contract, function, args, kwargs, value=0),
            eth_node=self.eth_node,
            extra_log_details=extra_log_details,
        )
        return pending.estimate_gas(BLOCK_ID_PENDING)

    def transact(
        self, transaction: Union[EthTransfer, "TransactionEstimated", ByteCode]
    ) -> TransactionSent:
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
        client: JSONRPCClient = self

        try:
            with self._nonce_lock:
                available_nonce: Nonce = self._available_nonce
                if isinstance(transaction, EthTransfer):
                    slot: TransactionSlot = TransactionSlot(
                        from_address=self.address,
                        eth_node=self.eth_node,
                        data=transaction,
                        extra_log_details={},
                        startgas=TRANSACTION_INTRINSIC_GAS,
                        gas_price=transaction.gas_price,
                        nonce=available_nonce,
                    )
                else:
                    slot = TransactionSlot(
                        from_address=transaction.from_address,  # type: ignore
                        eth_node=transaction.eth_node,  # type: ignore
                        data=transaction.data,  # type: ignore
                        extra_log_details=transaction.extra_log_details,  # type: ignore
                        startgas=transaction.estimated_gas,  # type: ignore
                        gas_price=transaction.gas_price,  # type: ignore
                        nonce=available_nonce,
                    )
                log_details: Dict[str, Any] = slot.to_log_details()
                if isinstance(slot.data, SmartContractCall):
                    function_call: SmartContractCall = slot.data
                    data: bytes = get_transaction_data(
                        web3=function_call.contract.web3,  # type: ignore
                        abi=function_call.contract.abi,  # type: ignore
                        function_name=function_call.function,
                        args=function_call.args,
                        kwargs=function_call.kwargs,
                    )
                    transaction_data: Dict[str, Any] = {
                        "data": decode_hex(data),
                        "gas": slot.startgas,
                        "nonce": slot.nonce,
                        "value": slot.data.value,
                        "to": function_call.contract.address,  # type: ignore
                        "gasPrice": slot.gas_price,
                        "chainId": self.chain_id,
                    }
                    log.debug("Transaction to call smart contract function will be sent", **log_details)
                elif isinstance(slot.data, EthTransfer):
                    transaction_data = {
                        "to": to_checksum_address(slot.data.to_address),
                        "gas": slot.startgas,
                        "nonce": slot.nonce,
                        "value": slot.data.value,
                        "gasPrice": slot.gas_price,
                        "chainId": self.chain_id,
                    }
                    log.debug("Transaction to transfer ether will be sent", **log_details)
                else:
                    transaction_data = {
                        "data": slot.data.bytecode,
                        "gas": slot.startgas,
                        "nonce": slot.nonce,
                        "value": 0,
                        "gasPrice": slot.gas_price,
                        "chainId": self.chain_id,
                    }
                    log.debug("Transaction to deploy smart contract will be sent", **log_details)
                signed_txn: Any = client.web3.eth.account.sign_transaction(  # type: ignore
                    transaction_data, client.privkey
                )
                tx_hash: bytes = client.web3.eth.send_raw_transaction(signed_txn.rawTransaction)  # type: ignore
                self._available_nonce = Nonce(self._available_nonce + 1)
        except ValueError as e:
            if isinstance(transaction, SmartContractCall):
                error_msg = "Transaction to call smart contract function failed"
            elif isinstance(transaction, EthTransfer):
                error_msg = "Transaction to transfer ether failed"
            else:
                error_msg = "Transaction to deploy smart contract failed"
            action: ClientErrorInspectResult = inspect_client_error(e, self.eth_node)
            if action == ClientErrorInspectResult.INSUFFICIENT_FUNDS:
                reason: str = "Transaction failed due to insufficient ETH balance. Please top up your ETH account."
                log.critical(error_msg, **log_details, reason=reason)
                raise InsufficientEth(reason)
            if action == ClientErrorInspectResult.TRANSACTION_UNDERPRICED:
                reason = "Transaction was rejected. This is potentially caused by the reuse of the previous transaction nonce as well as paying an amount of gas less than or equal to the previous transaction's gas amount"
                log.critical(error_msg, **log_details, reason=reason)
                raise ReplacementTransactionUnderpriced(reason)
            if action in THE_NONCE_WAS_REUSED:
                reason = "Transaction rejected because the nonce has been already mined."
                log.critical(error_msg, **log_details, reason=reason)
                raise EthereumNonceTooLow(reason)
            reason = f"Unexpected error in underlying Ethereum node: {str(e)}"
            log.critical(error_msg, **log_details, reason=reason)
            raise RaidenUnrecoverableError(reason)
        transaction_sent: TransactionSentImplementation = TransactionSentImplementation(
            from_address=slot.from_address,
            eth_node=slot.eth_node,
            data=slot.data,
            extra_log_details=slot.extra_log_details,
            startgas=slot.startgas,
            gas_price=slot.gas_price,
            nonce=slot.nonce,
            transaction_hash=TransactionHash(tx_hash),
            chain_id=self.chain_id,
        )
        log.debug("Transaction sent", **transaction_sent.to_log_details())
        return transaction_sent

    def new_contract_proxy(self, abi: ABI, contract_address: Address) -> Contract:
        return self.web3.eth.contract(abi=abi, address=contract_address)  # type: ignore

    def deploy_single_contract(
        self,
        contract_name: str,
        contract: Dict[str, Any],
        constructor_parameters: Optional[Sequence[Any]] = None,
    ) -> Tuple[Contract, TxReceipt]:
        """
        Deploy a single solidity contract without dependencies.

        Args:
            contract_name: The name of the contract to compile.
            contract: The dictionary containing the contract information (like ABI and BIN)
            constructor_parameters: A tuple of arguments to pass to the constructor.
        """
        ctor_parameters: Sequence[Any] = constructor_parameters or ()
        contract_object: Contract = self.web3.eth.contract(abi=contract["abi"], bytecode=contract["bin"])  # type: ignore
        contract_transaction: Dict[str, Any] = contract_object.constructor(*ctor_parameters).buildTransaction()  # type: ignore
        constructor_call: ByteCode = ByteCode(contract_name, contract_transaction["data"])
        block: BlockData = self.get_block(BLOCK_ID_LATEST)
        gas_with_margin: int = int(contract_transaction["gas"] * 1.5)
        gas_price: int = gas_price_for_fast_transaction(self.web3)
        transaction: TransactionEstimated = TransactionEstimated(
            from_address=self.address,
            data=constructor_call,
            eth_node=self.eth_node,
            extra_log_details={},
            estimated_gas=gas_with_margin,
            gas_price=gas_price,
            approximate_block=(BlockHash(block["hash"]), BlockNumber(block["number"])),
        )
        transaction_sent: TransactionSent = self.transact(transaction)
        transaction_mined: TransactionMined = self.poll_transaction(transaction_sent)
        maybe_contract_address: Optional[str] = transaction_mined.receipt.get("contractAddress")
        assert maybe_contract_address is not None, "'contractAddress' not set in receipt"
        contract_address: Address = to_canonical_address(maybe_contract_address)
        if not was_transaction_successfully_mined(transaction_mined):
            check_transaction_failure(transaction_mined, self)
            raise RuntimeError(
                f"Deployment of {contract_name} failed! Most likely a require from the constructor was not satisfied, or there is a compiler bug."
            )
        deployed_code: HexBytes = self.web3.eth.get_code(contract_address)  # type: ignore
        if not deployed_code:
            raise RaidenUnrecoverableError(
                f"Contract deployment of {contract_name} was successfull but address has no code! This is likely a bug in the ethereum client."
            )
        return (self.new_contract_proxy(abi=contract["abi"], contract_address=contract_address), transaction_mined.receipt)

    def poll_transaction(self, transaction_sent: TransactionSent) -> "TransactionMined":
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
        transaction_hash_hex: str = encode_hex(transaction_sent.transaction_hash)  # type: ignore
        while True:
            tx_receipt: Optional[TxReceipt] = None
            try:
                tx_receipt = self.web3.eth.get_transaction_receipt(transaction_hash_hex)  # type: ignore
            except TransactionNotFound:
                pass
            is_transaction_mined: bool = tx_receipt is not None and tx_receipt.get("blockNumber") is not None
            if is_transaction_mined:
                assert tx_receipt is not None, MYPY_ANNOTATION
                confirmation_block: int = tx_receipt["blockNumber"] + self.default_block_num_confirmations  # type: ignore
                block_number: int = self.block_number()
                is_transaction_confirmed: bool = block_number >= confirmation_block
                if is_transaction_confirmed:
                    transaction_mined: "TransactionMined" = TransactionMined(
                        from_address=transaction_sent.from_address,
                        data=transaction_sent.data,
                        eth_node=transaction_sent.eth_node,
                        extra_log_details=transaction_sent.extra_log_details,
                        startgas=transaction_sent.startgas,
                        gas_price=transaction_sent.gas_price,
                        nonce=transaction_sent.nonce,
                        transaction_hash=transaction_sent.transaction_hash,
                        receipt=tx_receipt,
                        chain_id=transaction_sent.chain_id,
                    )
                    return transaction_mined
            gevent.sleep(1.0)

    def get_filter_events(
        self,
        contract_address: Address,
        topics: Optional[List[Any]] = None,
        from_block: Union[int, str] = GENESIS_BLOCK_NUMBER,
        to_block: Union[int, str] = BLOCK_ID_LATEST,
    ) -> List[LogReceipt]:
        """Get events for the given query."""
        logs_blocks_sanity_check(from_block, to_block)
        return self.web3.eth.get_logs(  # type: ignore
            FilterParams(
                {
                    "fromBlock": from_block,
                    "toBlock": to_block,
                    "address": contract_address,
                    "topics": topics,
                }
            )
        )

    def check_for_insufficient_eth(
        self,
        transaction_name: str,
        transaction_executed: bool,
        required_gas: int,
        block_identifier: BlockIdentifier,
    ) -> None:
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
        our_address: str = to_checksum_address(self.address)
        balance: int = self.web3.eth.get_balance(our_address, block_identifier)  # type: ignore
        required_balance: int = required_gas * gas_price_for_fast_transaction(self.web3)
        if balance < required_balance:
            msg: str = f"Failed to execute {transaction_name} due to insufficient ETH"
            log.critical(msg, required_wei=required_balance, actual_wei=balance)
            raise InsufficientEth(msg)

    def wait_until_block(
        self, target_block_number: int, retry_timeout: float = 0.5
    ) -> int:
        current_block: int = self.block_number()
        while current_block < target_block_number:
            current_block = self.block_number()
            gevent.sleep(retry_timeout)
        return current_block

    def transaction_failed_with_a_require(self, transaction_hash: TransactionHash) -> Optional[bool]:
        """Tries to determine if the transaction with `transaction_hash`
        failed because of a `require` expression.
        """
        if self.eth_node == EthClient.GETH:
            try:
                trace: Dict[str, Any] = self.web3.manager.request_blocking(  # type: ignore
                    RPCEndpoint("debug_traceTransaction"), [to_hex(transaction_hash), {}]
                )
            except ValueError:
                return None
            return trace["structLogs"][-1]["op"] == GETH_REQUIRE_OPCODE
        if self.eth_node == EthClient.PARITY:
            try:
                response = self.web3.manager.request_blocking(  # type: ignore
                    RPCEndpoint("trace_replayTransaction"), [to_hex(transaction_hash), ["trace"]]
                )
            except ValueError:
                return None
            first_trace: Dict[str, Any] = response["trace"][0]
            return first_trace.get("error") == PARITY_REQUIRE_ERROR
        return None
