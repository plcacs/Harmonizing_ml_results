#!/usr/bin/env python3
import errno
import logging
import socket
from hashlib import sha256
from http import HTTPStatus
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gevent
import gevent.pool
import opentracing
import structlog
from eth_utils import encode_hex
from flask import Flask, Response, request, send_from_directory, url_for, jsonify
from flask_cors import CORS
from flask_opentracing import FlaskTracing
from flask_restful import Api
from gevent.event import Event
from gevent.pywsgi import WSGIServer
from hexbytes import HexBytes
from raiden_webui import RAIDEN_WEBUI_PATH
from werkzeug.exceptions import NotFound
from werkzeug.routing import BaseConverter
from raiden.api.exceptions import ChannelNotFound, NonexistingChannel
from raiden.api.objects import AddressList, PartnersPerTokenList
from raiden.api.python import RaidenAPI
from raiden.api.rest_utils import api_error, api_response
from raiden.api.v1.encoding import (
    AddressListSchema,
    ChannelStateSchema,
    EventPaymentReceivedSuccessSchema,
    EventPaymentSentFailedSchema,
    EventPaymentSentSuccessSchema,
    HexAddressConverter,
    InvalidEndpoint,
    NotificationSchema,
    PartnersPerTokenListSchema,
    PaymentSchema,
)
from raiden.api.v1.resources import (
    AddressResource,
    ChannelsResource,
    ChannelsResourceByTokenAddress,
    ChannelsResourceByTokenAndPartnerAddress,
    ConnectionsInfoResource,
    ConnectionsResource,
    ContractsResource,
    MintTokenResource,
    NodeSettingsResource,
    NotificationsResource,
    PartnersResourceByTokenAddress,
    PaymentEventsResource,
    PaymentResource,
    PendingTransfersResource,
    PendingTransfersResourceByTokenAddress,
    PendingTransfersResourceByTokenAndPartnerAddress,
    RaidenInternalEventsResource,
    RegisterTokenResource,
    ShutdownResource,
    StatusResource,
    TokensResource,
    UserDepositResource,
    VersionResource,
    create_blueprint,
)
from raiden.constants import UINT256_MAX, Environment
from raiden.exceptions import (
    AddressWithoutCode,
    AlreadyRegisteredTokenAddress,
    APIServerPortInUseError,
    BrokenPreconditionError,
    DepositMismatch,
    DepositOverLimit,
    DuplicatedChannelError,
    InsufficientEth,
    InsufficientFunds,
    InsufficientGasReserve,
    InvalidAmount,
    InvalidBinaryAddress,
    InvalidNumberInput,
    InvalidPaymentIdentifier,
    InvalidRevealTimeout,
    InvalidSecret,
    InvalidSecretHash,
    InvalidSettleTimeout,
    InvalidToken,
    InvalidTokenAddress,
    MaxTokenNetworkNumberReached,
    MintFailed,
    PaymentConflict,
    RaidenRecoverableError,
    SamePeerAddress,
    ServiceRequestFailed,
    TokenNetworkDeprecated,
    TokenNotRegistered,
    UnexpectedChannelState,
    UnknownTokenAddress,
    UserDepositNotConfigured,
    WithdrawMismatch,
)
from raiden.network.rpc.client import JSONRPCClient
from raiden.settings import RestApiConfig
from raiden.transfer import channel, views
from raiden.transfer.events import (
    EventPaymentReceivedSuccess,
    EventPaymentSentFailed,
    EventPaymentSentSuccess,
)
from raiden.transfer.state import ChannelState, NettingChannelState, RouteState
from raiden.ui.sync import blocks_to_sync
from raiden.utils.formatting import optional_address_to_string, to_checksum_address
from raiden.utils.gevent import spawn_named
from raiden.utils.http import split_endpoint
from raiden.utils.runnable import Runnable
from raiden.utils.system import get_system_spec
from raiden.utils.transfers import create_default_identifier
from raiden.utils.typing import (
    Address,
    Any as RaidenAny,
    BlockTimeout,
    Dict as RaidenDict,
    Endpoint,
    List as RaidenList,
    Optional as RaidenOptional,
    PaymentAmount,
    PaymentID,
    Secret,
    SecretHash,
    TargetAddress,
    TokenAddress,
    TokenAmount,
    TokenNetworkRegistryAddress,
    WithdrawAmount,
    cast,
)

log = structlog.get_logger(__name__)
CHANNEL_NETWORK_STATE = 'network_state'
URLS_V1: List[
    Union[
        Tuple[str, Type],
        Tuple[str, Type, str],
    ]
] = [
    ('/address', AddressResource),
    ('/version', VersionResource),
    ('/settings', NodeSettingsResource),
    ('/contracts', ContractsResource),
    ('/channels', ChannelsResource),
    ('/channels/<hexaddress:token_address>', ChannelsResourceByTokenAddress),
    ('/channels/<hexaddress:token_address>/<hexaddress:partner_address>', ChannelsResourceByTokenAndPartnerAddress),
    ('/connections/<hexaddress:token_address>', ConnectionsResource),
    ('/connections', ConnectionsInfoResource),
    ('/notifications', NotificationsResource),
    ('/payments', PaymentEventsResource, 'paymentresource'),
    ('/payments/<hexaddress:token_address>', PaymentEventsResource, 'token_paymentresource'),
    ('/payments/<hexaddress:token_address>/<hexaddress:target_address>', PaymentResource, 'token_target_paymentresource'),
    ('/tokens', TokensResource),
    ('/tokens/<hexaddress:token_address>/partners', PartnersResourceByTokenAddress),
    ('/tokens/<hexaddress:token_address>', RegisterTokenResource),
    ('/pending_transfers', PendingTransfersResource, 'pending_transfers_resource'),
    ('/pending_transfers/<hexaddress:token_address>', PendingTransfersResourceByTokenAddress, 'pending_transfers_resource_by_token'),
    ('/pending_transfers/<hexaddress:token_address>/<hexaddress:partner_address>', PendingTransfersResourceByTokenAndPartnerAddress, 'pending_transfers_resource_by_token_and_partner'),
    ('/user_deposit', UserDepositResource),
    ('/status', StatusResource),
    ('/shutdown', ShutdownResource),
    ('/_debug/raiden_events', RaidenInternalEventsResource),
    ('/_testing/tokens/<hexaddress:token_address>/mint', MintTokenResource, 'tokensmintresource'),
]


def endpoint_not_found(e: Exception) -> Response:
    errors: List[str] = ['invalid endpoint']
    if isinstance(e, InvalidEndpoint):
        errors.append(e.description)
    return api_error(errors, HTTPStatus.NOT_FOUND)


def hexbytes_to_str(map_: Dict[str, Any]) -> None:
    """Converts values that are of type `HexBytes` to strings."""
    for k, v in map_.items():
        if isinstance(v, HexBytes):
            map_[k] = encode_hex(v)


def encode_byte_values(map_: Dict[str, Any]) -> None:
    """Converts values that are of type `bytes` to strings."""
    for k, v in map_.items():
        if isinstance(v, bytes):
            map_[k] = encode_hex(v)


def encode_object_to_str(map_: Dict[str, Any]) -> None:
    for k, v in map_.items():
        if isinstance(v, int) or k == 'args':
            continue
        if not isinstance(v, str):
            map_[k] = repr(v)


def restapi_setup_urls(flask_api_context: Api, rest_api: Any, urls: List[Union[Tuple[str, Type], Tuple[str, Type, str]]]) -> None:
    for url_tuple in urls:
        if len(url_tuple) == 2:
            route, resource_cls = url_tuple
            endpoint = resource_cls.__name__.lower()
        elif len(url_tuple) == 3:
            route, resource_cls, endpoint = url_tuple
        else:
            raise ValueError(f'Invalid URL format: {url_tuple!r}')
        flask_api_context.add_resource(resource_cls, route, resource_class_kwargs={'rest_api_object': rest_api}, endpoint=endpoint)


def restapi_setup_type_converters(flask_app: Flask, names_to_converters: Dict[str, Type[BaseConverter]]) -> None:
    flask_app.url_map.converters.update(names_to_converters)


class APIServer(Runnable):
    """
    Runs the API-server that routes the endpoint to the resources.
    """
    _api_prefix: str = '/api/1'

    def __init__(self, rest_api: "RestAPI", config: RestApiConfig, eth_rpc_endpoint: Optional[str] = None) -> None:
        super().__init__()
        if rest_api.version != 1:
            raise ValueError(f'Invalid api version: {rest_api.version}')
        self._api_prefix = f'/api/v{rest_api.version}'
        flask_app: Flask = Flask(__name__)
        if config.cors_domain_list:
            CORS(flask_app, origins=config.cors_domain_list)
        if config.enable_tracing:
            FlaskTracing(opentracing.tracer, trace_all_requests=True, app=flask_app)
        if eth_rpc_endpoint:
            if not eth_rpc_endpoint.startswith('http'):
                eth_rpc_endpoint = f'http://{eth_rpc_endpoint}'
            flask_app.config['WEB3_ENDPOINT'] = eth_rpc_endpoint
        blueprint = create_blueprint()
        flask_api_context: Api = Api(blueprint, prefix=self._api_prefix)
        restapi_setup_type_converters(flask_app, {'hexaddress': cast(Type[BaseConverter], HexAddressConverter)})
        restapi_setup_urls(flask_api_context, rest_api, URLS_V1)
        self.stop_event: Event = Event()
        self.config: RestApiConfig = config
        self.rest_api: "RestAPI" = rest_api
        self.flask_app: Flask = flask_app
        self.blueprint = blueprint
        self.flask_api_context: Api = flask_api_context
        self.wsgiserver: Optional[WSGIServer] = None
        self.flask_app.register_blueprint(self.blueprint)
        self.flask_app.config['WEBUI_PATH'] = RAIDEN_WEBUI_PATH
        self.flask_app.register_error_handler(HTTPStatus.NOT_FOUND, endpoint_not_found)
        self.flask_app.register_error_handler(Exception, self.unhandled_exception)
        self.flask_app.before_request(self._check_shutdown_before_handle_request)
        self.flask_app.config['PROPAGATE_EXCEPTIONS'] = True
        if config.web_ui_enabled:
            for route in ('/ui/<path:file_name>', '/ui', '/ui/', '/index.html', '/'):
                self.flask_app.add_url_rule(route, route, view_func=self._serve_webui, methods=('GET',))

    def _check_shutdown_before_handle_request(self) -> Optional[Response]:
        """
        We don't want to handle requests when shutting down.
        """
        if self.stop_event.is_set():
            return api_error('Raiden API is shutting down', HTTPStatus.SERVICE_UNAVAILABLE)
        return None

    def _serve_webui(self, file_name: str = 'index.html') -> Response:
        try:
            if not file_name:
                raise NotFound
            web3: Optional[str] = self.flask_app.config.get('WEB3_ENDPOINT')
            if 'config.' in file_name and file_name.endswith('.json'):
                environment_type: str = self.rest_api.raiden_api.raiden.config.environment_type.name.lower()
                config_dict: Dict[str, Any] = {
                    'raiden': self._api_prefix,
                    'web3': web3,
                    'settle_timeout': self.rest_api.raiden_api.raiden.config.settle_timeout,
                    'reveal_timeout': self.rest_api.raiden_api.raiden.config.reveal_timeout,
                    'environment_type': environment_type,
                }
                host_header: Optional[str] = request.headers.get('Host')
                if web3 and host_header:
                    web3_host, web3_port = split_endpoint(web3)
                    if web3_host in ('localhost', '127.0.0.1'):
                        host, _ = split_endpoint(request.host)
                        web3_port_str: str = ''
                        if web3_port:
                            web3_port_str = f':{web3_port}'
                        web3 = f'http://{host}{web3_port_str}'
                        config_dict['web3'] = web3
                response = jsonify(config_dict)
            else:
                response = send_from_directory(self.flask_app.config['WEBUI_PATH'], file_name)
        except (NotFound, AssertionError):
            if file_name.endswith('.json'):
                response = api_error('Service unavailable, try again later', HTTPStatus.SERVICE_UNAVAILABLE)
            else:
                response = send_from_directory(self.flask_app.config['WEBUI_PATH'], 'index.html')
        return response

    def _run(self) -> None:
        try:
            if self.wsgiserver is not None:
                self.wsgiserver.serve_forever()
        except gevent.GreenletExit:
            raise
        except Exception:
            self.stop()
            raise

    def start(self) -> None:
        self.stop_event.clear()
        log.debug('REST API starting', host=self.config.host, port=self.config.port)
        wsgi_log = logging.getLogger(__name__ + '.pywsgi')
        pool = gevent.pool.Pool()
        wsgiserver = WSGIServer((self.config.host, self.config.port), self.flask_app, log=wsgi_log, error_log=wsgi_log, spawn=pool)
        try:
            wsgiserver.init_socket()
        except socket.error as e:
            if e.errno == errno.EADDRINUSE:
                raise APIServerPortInUseError(f'{self.config.host}:{self.config.port}')
            raise
        self.wsgiserver = wsgiserver
        log.debug('REST API started', host=self.config.host, port=self.config.port)
        super().start()

    def stop(self) -> None:
        self.stop_event.set()
        log.debug('REST API stopping', host=self.config.host, port=self.config.port, node=self.rest_api.checksum_address)
        if self.wsgiserver is not None:
            self.wsgiserver.stop(timeout=5)
            self.wsgiserver = None
        log.debug('REST API stopped', host=self.config.host, port=self.config.port, node=self.rest_api.checksum_address)

    def unhandled_exception(self, exception: Exception) -> Response:
        """Flask.errorhandler when an exception wasn't correctly handled"""
        log.critical('Unhandled exception when processing endpoint request', exc_info=True, node=self.rest_api.checksum_address)
        self.greenlet.kill(exception)
        return api_error([str(exception)], HTTPStatus.INTERNAL_SERVER_ERROR)


class RestAPI:
    """
    This wraps around the actual RaidenAPI in api/python.
    It will provide the additional, necessary RESTful logic and the proper JSON-encoding of the Objects provided by the RaidenAPI.
    """
    version: int = 1

    def __init__(self, raiden_api: Optional[RaidenAPI] = None, rpc_client: Optional[JSONRPCClient] = None) -> None:
        self._rpc_client: Optional[JSONRPCClient] = rpc_client
        self._raiden_api: Optional[RaidenAPI] = raiden_api
        self.channel_schema: ChannelStateSchema = ChannelStateSchema()
        self.address_list_schema: AddressListSchema = AddressListSchema()
        self.partner_per_token_list_schema: PartnersPerTokenListSchema = PartnersPerTokenListSchema()
        self.payment_schema: PaymentSchema = PaymentSchema()
        self.sent_success_payment_schema: EventPaymentSentSuccessSchema = EventPaymentSentSuccessSchema()
        self.received_success_payment_schema: EventPaymentReceivedSuccessSchema = EventPaymentReceivedSuccessSchema()
        self.failed_payment_schema: EventPaymentSentFailedSchema = EventPaymentSentFailedSchema()
        self.notification_schema: NotificationSchema = NotificationSchema()

    @property
    def rpc_client(self) -> JSONRPCClient:
        assert self._rpc_client is not None, 'rpc_client accessed but not initialized.'
        return self._rpc_client

    @property
    def checksum_address(self) -> Optional[str]:
        return to_checksum_address(self.raiden_api.address) if self.available else None

    @property
    def raiden_api(self) -> RaidenAPI:
        assert self._raiden_api is not None, 'raiden_api accessed but not initialized'
        return self._raiden_api

    @raiden_api.setter
    def raiden_api(self, raiden_api: RaidenAPI) -> None:
        self._raiden_api = raiden_api

    @property
    def available(self) -> bool:
        return self._raiden_api is not None

    def get_our_address(self) -> Response:
        return api_response(result=dict(our_address=self.checksum_address))

    @classmethod
    def get_raiden_version(cls) -> Response:
        return api_response(result=dict(version=get_system_spec()['raiden']))

    def get_node_settings(self) -> Response:
        pfs_config = self.raiden_api.raiden.config.pfs_config
        settings: Dict[str, Any] = dict(pathfinding_service_address=pfs_config and pfs_config.info.url)
        return api_response(result=settings)

    def get_contract_versions(self) -> Response:
        raiden = self.raiden_api.raiden
        service_registry_address: Optional[str] = (
            to_checksum_address(raiden.default_service_registry.address)
            if raiden.default_service_registry
            else None
        )
        user_deposit_address: Optional[str] = (
            to_checksum_address(raiden.default_user_deposit.address)
            if raiden.default_user_deposit
            else None
        )
        monitoring_service_address: Optional[str] = (
            to_checksum_address(raiden.default_msc_address)
            if raiden.default_msc_address
            else None
        )
        one_to_n_address: Optional[str] = (
            to_checksum_address(raiden.default_one_to_n_address)
            if raiden.default_one_to_n_address
            else None
        )
        contracts: Dict[str, Any] = dict(
            contracts_version=raiden.proxy_manager.contract_manager.contracts_version,
            token_network_registry_address=to_checksum_address(raiden.default_registry.address),
            secret_registry_address=to_checksum_address(raiden.default_secret_registry.address),
            service_registry_address=service_registry_address,
            user_deposit_address=user_deposit_address,
            monitoring_service_address=monitoring_service_address,
            one_to_n_address=one_to_n_address,
        )
        return api_response(result=contracts)

    def register_token(self, registry_address: Address, token_address: Address) -> Response:
        if self.raiden_api.raiden.config.environment_type == Environment.PRODUCTION:
            return api_error(errors='Registering a new token is currently disabled in production mode', status_code=HTTPStatus.NOT_IMPLEMENTED)
        conflict_exceptions: Tuple[Any, ...] = (
            AddressWithoutCode,
            AlreadyRegisteredTokenAddress,
            BrokenPreconditionError,
            InvalidBinaryAddress,
            InvalidToken,
            InvalidTokenAddress,
            RaidenRecoverableError,
        )
        log.debug('Registering token', node=self.checksum_address, registry_address=to_checksum_address(registry_address), token_address=to_checksum_address(token_address))
        try:
            token_network_address = self.raiden_api.token_network_register(
                registry_address=registry_address,
                token_address=token_address,
                channel_participant_deposit_limit=TokenAmount(UINT256_MAX),
                token_network_deposit_limit=TokenAmount(UINT256_MAX),
            )
        except conflict_exceptions as e:
            return api_error(errors=str(e), status_code=HTTPStatus.CONFLICT)
        except InsufficientEth as e:
            return api_error(errors=str(e), status_code=HTTPStatus.PAYMENT_REQUIRED)
        except MaxTokenNetworkNumberReached as e:
            return api_error(errors=str(e), status_code=HTTPStatus.FORBIDDEN)
        return api_response(result=dict(token_network_address=to_checksum_address(token_network_address)), status_code=HTTPStatus.CREATED)

    def mint_token_for(self, token_address: Address, to: Address, value: int) -> Response:
        if self.raiden_api.raiden.config.environment_type == Environment.PRODUCTION:
            return api_error(errors='Minting a token is currently disabled in production mode', status_code=HTTPStatus.NOT_IMPLEMENTED)
        log.debug('Minting token', node=self.checksum_address, token_address=to_checksum_address(token_address), to=to_checksum_address(to), value=value)
        try:
            transaction_hash = self.raiden_api.mint_token_for(token_address=token_address, to=to, value=value)
        except MintFailed as e:
            return api_error(f'Minting failed: {str(e)}', status_code=HTTPStatus.BAD_REQUEST)
        except InsufficientEth as e:
            return api_error(errors=str(e), status_code=HTTPStatus.PAYMENT_REQUIRED)
        return api_response(status_code=HTTPStatus.OK, result=dict(transaction_hash=encode_hex(transaction_hash)))

    def open(
        self,
        registry_address: Address,
        partner_address: Address,
        token_address: Address,
        settle_timeout: Optional[int] = None,
        reveal_timeout: Optional[int] = None,
        total_deposit: Optional[int] = None,
    ) -> Response:
        log.debug('Opening channel', node=self.checksum_address, registry_address=to_checksum_address(registry_address), partner_address=to_checksum_address(partner_address), token_address=to_checksum_address(token_address), settle_timeout=settle_timeout, reveal_timeout=reveal_timeout)
        confirmed_block_identifier = views.get_confirmed_blockhash(self.raiden_api.raiden)
        try:
            token = self.raiden_api.raiden.proxy_manager.token(token_address, block_identifier=confirmed_block_identifier)
        except AddressWithoutCode as e:
            return api_error(errors=str(e), status_code=HTTPStatus.CONFLICT)
        balance: int = token.balance_of(self.raiden_api.raiden.address)
        if total_deposit is not None and total_deposit > balance:
            error_msg = 'Not enough balance to deposit. {} Available={} Needed={}'.format(to_checksum_address(token_address), balance, total_deposit)
            return api_error(errors=error_msg, status_code=HTTPStatus.PAYMENT_REQUIRED)
        try:
            self.raiden_api.channel_open(
                registry_address=registry_address,
                token_address=token_address,
                partner_address=partner_address,
                settle_timeout=settle_timeout,
                reveal_timeout=reveal_timeout,
            )
            status_code = HTTPStatus.CREATED
        except DuplicatedChannelError:
            channel_status = channel.get_status(self.raiden_api.get_channel(registry_address, token_address, partner_address))
            if channel_status != ChannelState.STATE_OPENED:
                return api_error(errors='Channel is not in an open state.', status_code=HTTPStatus.CONFLICT)
            status_code = HTTPStatus.OK
        except (
            InvalidRevealTimeout,
            InvalidSettleTimeout,
            TokenNetworkDeprecated,
            InvalidBinaryAddress,
            SamePeerAddress,
            AddressWithoutCode,
            DuplicatedChannelError,
            TokenNotRegistered,
        ) as e:
            return api_error(errors=str(e), status_code=HTTPStatus.CONFLICT)
        except (InsufficientEth, InsufficientFunds, InsufficientGasReserve) as e:
            return api_error(errors=str(e), status_code=HTTPStatus.PAYMENT_REQUIRED)
        if total_deposit:
            log.debug('Depositing to new channel', node=self.checksum_address, registry_address=to_checksum_address(registry_address), token_address=to_checksum_address(token_address), partner_address=to_checksum_address(partner_address), total_deposit=total_deposit)
            try:
                self.raiden_api.set_total_channel_deposit(registry_address=registry_address, token_address=token_address, partner_address=partner_address, total_deposit=total_deposit)
            except (InsufficientEth, InsufficientFunds) as e:
                return api_error(errors=str(e), status_code=HTTPStatus.PAYMENT_REQUIRED)
            except (NonexistingChannel, UnknownTokenAddress) as e:
                return api_error(errors=str(e), status_code=HTTPStatus.BAD_REQUEST)
            except (DepositOverLimit, DepositMismatch, UnexpectedChannelState) as e:
                return api_error(errors=str(e), status_code=HTTPStatus.CONFLICT)
        result = self._updated_channel_state_from_addresses(registry_address, partner_address, token_address)
        return api_response(result=result, status_code=status_code)

    def leave(self, registry_address: Address, token_address: Address) -> Response:
        log.debug('Leaving token network', node=self.checksum_address, registry_address=to_checksum_address(registry_address), token_address=to_checksum_address(token_address))
        closed_channels = self.raiden_api.token_network_leave(registry_address, token_address)
        closed_channels_result: List[Any] = []
        for channel_state in closed_channels:
            result = self._updated_channel_state(registry_address, channel_state)
            closed_channels_result.append(result)
        return api_response(result=closed_channels_result)

    def get_connection_managers_info(self, registry_address: Address) -> Response:
        log.debug('Getting connection managers info', node=self.checksum_address, registry_address=to_checksum_address(registry_address))
        connection_managers: Dict[str, Any] = {}
        for token in self.raiden_api.get_tokens_list(registry_address):
            open_channels = views.get_channelstate_open(chain_state=views.state_from_raiden(self.raiden_api.raiden), token_network_registry_address=registry_address, token_address=token)
            if open_channels:
                connection_managers[to_checksum_address(token)] = {'sum_deposits': str(views.get_our_deposits_for_token_network(views.state_from_raiden(self.raiden_api.raiden), registry_address, token)), 'channels': str(len(open_channels))}
        return api_response(result=connection_managers)

    def get_channel_list(self, registry_address: Address, token_address: Optional[Address] = None, partner_address: Optional[Address] = None) -> Response:
        log.debug('Getting channel list', node=self.checksum_address, registry_address=to_checksum_address(registry_address), token_address=optional_address_to_string(token_address), partner_address=optional_address_to_string(partner_address))
        raiden_service_result = self.raiden_api.get_channel_list(registry_address, token_address, partner_address)
        channels_result: List[Any] = []
        for channel_state in raiden_service_result:
            result = self._updated_channel_state(registry_address, channel_state)
            channels_result.append(result)
        return api_response(result=channels_result)

    def get_tokens_list(self, registry_address: Address) -> Response:
        log.debug('Getting token list', node=self.checksum_address, registry_address=to_checksum_address(registry_address))
        raiden_service_result = self.raiden_api.get_tokens_list(registry_address)
        tokens_list = AddressList(raiden_service_result)
        result = self.address_list_schema.dump(tokens_list)
        return api_response(result=result)

    def get_token_network_for_token(self, registry_address: Address, token_address: Address) -> Response:
        log.debug('Getting token network for token', node=self.checksum_address, token_address=to_checksum_address(token_address))
        token_network_address = self.raiden_api.get_token_network_address_for_token_address(registry_address=registry_address, token_address=token_address)
        if token_network_address is not None:
            return api_response(result=to_checksum_address(token_network_address))
        else:
            pretty_address = to_checksum_address(token_address)
            message = f'No token network registered for token "{pretty_address}"'
            return api_error(message, status_code=HTTPStatus.NOT_FOUND)

    def get_raiden_events_payment_history_with_timestamps(
        self,
        registry_address: Address,
        token_address: Optional[Address] = None,
        target_address: Optional[Address] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> Response:
        log.debug('Getting payment history', node=self.checksum_address, token_address=optional_address_to_string(token_address), target_address=optional_address_to_string(target_address), limit=limit, offset=offset)
        try:
            service_result = self.raiden_api.get_raiden_events_payment_history_with_timestamps(
                registry_address=registry_address, token_address=token_address, target_address=target_address, limit=limit, offset=offset
            )
        except (InvalidNumberInput, InvalidBinaryAddress, InvalidTokenAddress) as e:
            return api_error(str(e), status_code=HTTPStatus.BAD_REQUEST)
        result: List[Any] = []
        chain_state = views.state_from_raiden(self.raiden_api.raiden)
        for event in service_result:
            if isinstance(event.event, EventPaymentSentSuccess):
                serialized_event = self.sent_success_payment_schema.serialize(chain_state=chain_state, event=event)
            elif isinstance(event.event, EventPaymentSentFailed):
                serialized_event = self.failed_payment_schema.serialize(chain_state=chain_state, event=event)
            elif isinstance(event.event, EventPaymentReceivedSuccess):
                serialized_event = self.received_success_payment_schema.serialize(chain_state=chain_state, event=event)
            else:
                log.warning('Unexpected event', node=self.checksum_address, unexpected_event=event.event)
                continue
            result.append(serialized_event)
        return api_response(result=result)

    def get_raiden_internal_events_with_timestamps(self, limit: int, offset: int) -> Response:
        assert self.raiden_api.raiden.wal, 'Raiden Service has to be initialized'
        events = [str(e) for e in self.raiden_api.raiden.wal.storage.get_events_with_timestamps(limit=limit, offset=offset)]
        return api_response(result=events)

    def get_channel(self, registry_address: Address, token_address: Address, partner_address: Address) -> Response:
        log.debug('Getting channel', node=self.checksum_address, registry_address=to_checksum_address(registry_address), token_address=to_checksum_address(token_address), partner_address=to_checksum_address(partner_address))
        try:
            result = self._updated_channel_state_from_addresses(registry_address, partner_address, token_address)
            if result is None:
                msg = f"Channel with partner '{to_checksum_address(partner_address)}' for token '{to_checksum_address(token_address)}' could not be found."
                return api_error(errors=msg, status_code=HTTPStatus.NOT_FOUND)
            return api_response(result=result)
        except ChannelNotFound as e:
            return api_error(errors=str(e), status_code=HTTPStatus.NOT_FOUND)

    def get_partners_by_token(self, registry_address: Address, token_address: Address) -> Response:
        log.debug('Getting partners by token', node=self.checksum_address, registry_address=to_checksum_address(registry_address), token_address=to_checksum_address(token_address))
        return_list: List[Dict[str, Any]] = []
        try:
            raiden_service_result = self.raiden_api.get_channel_list(registry_address, token_address)
        except InvalidBinaryAddress as e:
            return api_error(errors=str(e), status_code=HTTPStatus.CONFLICT)
        for result in raiden_service_result:
            return_list.append({
                'partner_address': result.partner_state.address,
                'channel': url_for('v1_resources.channelsresourcebytokenandpartneraddress', token_address=token_address, partner_address=result.partner_state.address)
            })
        schema_list = PartnersPerTokenList(return_list)
        result = self.partner_per_token_list_schema.dump(schema_list)
        return api_response(result=result)

    def initiate_payment(
        self,
        registry_address: Address,
        token_address: Address,
        target_address: Address,
        amount: int,
        identifier: Optional[int],
        secret: Optional[bytes],
        secret_hash: Optional[bytes],
        lock_timeout: int,
        route_states: Optional[Any] = None,
    ) -> Response:
        log.debug('Initiating payment', node=self.checksum_address, registry_address=to_checksum_address(registry_address), token_address=to_checksum_address(token_address), target_address=to_checksum_address(target_address), amount=amount, payment_identifier=identifier, secret=secret, secret_hash=secret_hash, lock_timeout=lock_timeout)
        if identifier is None:
            identifier = create_default_identifier()
        try:
            payment_status = self.raiden_api.transfer_and_wait(
                registry_address=registry_address,
                token_address=token_address,
                target=target_address,
                amount=amount,
                identifier=identifier,
                secret=secret,
                secrethash=secret_hash,
                lock_timeout=lock_timeout,
                route_states=route_states,
            )
        except (InvalidAmount, InvalidBinaryAddress, InvalidSecret, InvalidSecretHash, InvalidPaymentIdentifier, PaymentConflict, SamePeerAddress, UnknownTokenAddress) as e:
            return api_error(errors=str(e), status_code=HTTPStatus.CONFLICT)
        except InsufficientFunds as e:
            return api_error(errors=str(e), status_code=HTTPStatus.PAYMENT_REQUIRED)
        result = payment_status.payment_done.get()
        if isinstance(result, EventPaymentSentFailed):
            return api_error(errors=f"Payment couldn't be completed because: {result.reason}", status_code=HTTPStatus.CONFLICT)
        assert isinstance(result, EventPaymentSentSuccess)
        payment: Dict[str, Any] = {
            'initiator_address': self.checksum_address,
            'registry_address': registry_address,
            'token_address': token_address,
            'target_address': target_address,
            'amount': amount,
            'identifier': identifier,
            'secret': result.secret,
            'secret_hash': sha256(result.secret).digest(),
        }
        result = self.payment_schema.dump(payment)
        return api_response(result=result)

    def get_new_notifications(self) -> Response:
        log.debug('Requesting new notifications')
        notifications = self.raiden_api.get_new_notifications()
        result = self.notification_schema.dump(notifications, many=True)
        return api_response(result=result)

    def _updated_channel_state_from_addresses(self, registry_address: Address, partner_address: Address, token_address: Address) -> Optional[Dict[str, Any]]:
        chain_state = views.state_from_raiden(self.raiden_api.raiden)
        updated_channel_state = views.get_channelstate_for(chain_state=chain_state, token_network_registry_address=registry_address, token_address=token_address, partner_address=partner_address)
        if updated_channel_state:
            result = self.channel_schema.dump(updated_channel_state)
            result[CHANNEL_NETWORK_STATE] = views.get_node_network_status(chain_state, updated_channel_state.partner_state.address).value
        else:
            result = None
        return result

    def _updated_channel_state(self, registry_address: Address, channel_state: ChannelState) -> Optional[Dict[str, Any]]:
        chain_state = views.state_from_raiden(self.raiden_api.raiden)
        updated_channel_state = views.get_channelstate_for(chain_state=chain_state, token_network_registry_address=registry_address, token_address=channel_state.token_address, partner_address=channel_state.partner_state.address)
        if updated_channel_state:
            result = self.channel_schema.dump(updated_channel_state)
            result[CHANNEL_NETWORK_STATE] = views.get_node_network_status(chain_state, updated_channel_state.partner_state.address).value
        else:
            result = None
        return result

    def _deposit(self, registry_address: Address, channel_state: ChannelState, total_deposit: int) -> Response:
        log.debug('Depositing to channel', node=self.checksum_address, registry_address=to_checksum_address(registry_address), channel_identifier=channel_state.identifier, total_deposit=total_deposit)
        if channel.get_status(channel_state) != ChannelState.STATE_OPENED:
            return api_error(errors="Can't set total deposit on a closed channel", status_code=HTTPStatus.CONFLICT)
        try:
            self.raiden_api.set_total_channel_deposit(registry_address, channel_state.token_address, channel_state.partner_state.address, total_deposit)
        except (InsufficientEth, InsufficientFunds) as e:
            return api_error(errors=str(e), status_code=HTTPStatus.PAYMENT_REQUIRED)
        except DepositOverLimit as e:
            return api_error(errors=str(e), status_code=HTTPStatus.CONFLICT)
        except DepositMismatch as e:
            return api_error(errors=str(e), status_code=HTTPStatus.CONFLICT)
        except TokenNetworkDeprecated as e:
            return api_error(errors=str(e), status_code=HTTPStatus.CONFLICT)
        result = self._updated_channel_state(registry_address, channel_state)
        return api_response(result=result)

    def _withdraw(self, registry_address: Address, channel_state: ChannelState, total_withdraw: int) -> Response:
        log.debug('Withdrawing from channel', node=self.checksum_address, registry_address=to_checksum_address(registry_address), channel_identifier=channel_state.identifier, total_withdraw=total_withdraw)
        if channel.get_status(channel_state) != ChannelState.STATE_OPENED:
            return api_error(errors="Can't withdraw from a closed channel", status_code=HTTPStatus.CONFLICT)
        try:
            self.raiden_api.set_total_channel_withdraw(registry_address, channel_state.token_address, channel_state.partner_state.address, total_withdraw)
        except (NonexistingChannel, UnknownTokenAddress) as e:
            return api_error(errors=str(e), status_code=HTTPStatus.BAD_REQUEST)
        except (InsufficientFunds, WithdrawMismatch, ServiceRequestFailed) as e:
            return api_error(errors=str(e), status_code=HTTPStatus.CONFLICT)
        result = self._updated_channel_state(registry_address, channel_state)
        return api_response(result=result)

    def _set_channel_reveal_timeout(self, registry_address: Address, channel_state: ChannelState, reveal_timeout: int) -> Response:
        log.debug('Set channel reveal timeout', node=self.checksum_address, channel_identifier=channel_state.identifier, reveal_timeout=reveal_timeout)
        if channel.get_status(channel_state) != ChannelState.STATE_OPENED:
            return api_error(errors="Can't update the reveal timeout of a closed channel", status_code=HTTPStatus.CONFLICT)
        try:
            self.raiden_api.set_reveal_timeout(registry_address=registry_address, token_address=channel_state.token_address, partner_address=channel_state.partner_state.address, reveal_timeout=reveal_timeout)
        except (NonexistingChannel, UnknownTokenAddress, InvalidBinaryAddress) as e:
            return api_error(errors=str(e), status_code=HTTPStatus.BAD_REQUEST)
        except InvalidRevealTimeout as e:
            return api_error(errors=str(e), status_code=HTTPStatus.CONFLICT)
        result = self._updated_channel_state(registry_address, channel_state)
        return api_response(result=result)

    def _close(self, registry_address: Address, channel_state: ChannelState) -> Response:
        log.debug('Closing channel', node=self.checksum_address, registry_address=to_checksum_address(registry_address), channel_identifier=channel_state.identifier)
        if channel.get_status(channel_state) != ChannelState.STATE_OPENED:
            return api_error(errors='Attempted to close an already closed channel', status_code=HTTPStatus.CONFLICT)
        try:
            self.raiden_api.channel_close(registry_address, channel_state.token_address, channel_state.partner_state.address)
        except InsufficientEth as e:
            return api_error(errors=str(e), status_code=HTTPStatus.PAYMENT_REQUIRED)
        result = self._updated_channel_state(registry_address, channel_state)
        return api_response(result=result)

    def patch_channel(
        self,
        registry_address: Address,
        token_address: Address,
        partner_address: Address,
        total_deposit: Optional[int] = None,
        total_withdraw: Optional[int] = None,
        reveal_timeout: Optional[int] = None,
        state: Optional[str] = None,
    ) -> Response:
        log.debug('Patching channel', node=self.checksum_address, registry_address=to_checksum_address(registry_address), token_address=to_checksum_address(token_address), partner_address=to_checksum_address(partner_address), total_deposit=total_deposit, reveal_timeout=reveal_timeout, state=state)
        if reveal_timeout is not None and state is not None:
            return api_error(errors="Can not update a channel's reveal timeout and state at the same time", status_code=HTTPStatus.CONFLICT)
        if total_deposit is not None and state is not None:
            return api_error(errors="Can not update a channel's total deposit and state at the same time", status_code=HTTPStatus.CONFLICT)
        if total_withdraw is not None and state is not None:
            return api_error(errors="Can not update a channel's total withdraw and state at the same time", status_code=HTTPStatus.CONFLICT)
        if total_withdraw is not None and total_deposit is not None:
            return api_error(errors="Can not update a channel's total withdraw and total deposit at the same time", status_code=HTTPStatus.CONFLICT)
        if reveal_timeout is not None and total_deposit is not None:
            return api_error(errors="Can not update a channel's reveal timeout and total deposit at the same time", status_code=HTTPStatus.CONFLICT)
        if reveal_timeout is not None and total_withdraw is not None:
            return api_error(errors="Can not update a channel's reveal timeout and total withdraw at the same time", status_code=HTTPStatus.CONFLICT)
        if total_deposit is not None and total_deposit < 0:
            return api_error(errors='Amount to deposit must not be negative.', status_code=HTTPStatus.CONFLICT)
        if total_withdraw is not None and total_withdraw < 0:
            return api_error(errors='Amount to withdraw must not be negative.', status_code=HTTPStatus.CONFLICT)
        empty_request: bool = total_deposit is None and state is None and total_withdraw is None and reveal_timeout is None
        if empty_request:
            return api_error(errors="Nothing to do. Should either provide 'total_deposit', 'total_withdraw', 'reveal_timeout' or 'state' argument", status_code=HTTPStatus.BAD_REQUEST)
        try:
            channel_state = self.raiden_api.get_channel(registry_address=registry_address, token_address=token_address, partner_address=partner_address)
        except ChannelNotFound:
            return api_error(errors='Requested channel for token {} and partner {} not found'.format(to_checksum_address(token_address), to_checksum_address(partner_address)), status_code=HTTPStatus.CONFLICT)
        except InvalidBinaryAddress as e:
            return api_error(errors=str(e), status_code=HTTPStatus.CONFLICT)
        if total_deposit is not None:
            result = self._deposit(registry_address, channel_state, total_deposit)
        elif total_withdraw is not None:
            result = self._withdraw(registry_address, channel_state, total_withdraw)
        elif reveal_timeout is not None:
            result = self._set_channel_reveal_timeout(registry_address=registry_address, channel_state=channel_state, reveal_timeout=reveal_timeout)
        elif state == ChannelState.STATE_CLOSED.value:
            result = self._close(registry_address, channel_state)
        else:
            result = api_error(errors=f'Provided invalid channel state {state}', status_code=HTTPStatus.BAD_REQUEST)
        return result

    def get_pending_transfers(self, token_address: Optional[Address] = None, partner_address: Optional[Address] = None) -> Response:
        try:
            return api_response(self.raiden_api.get_pending_transfers(token_address=token_address, partner_address=partner_address))
        except (ChannelNotFound, UnknownTokenAddress) as e:
            return api_error(errors=str(e), status_code=HTTPStatus.NOT_FOUND)

    def _deposit_to_udc(self, total_deposit: int) -> Response:
        log.debug('Depositing to UDC', node=self.checksum_address, total_deposit=total_deposit)
        try:
            transaction_hash = self.raiden_api.set_total_udc_deposit(total_deposit)
        except (InsufficientEth, InsufficientFunds) as e:
            return api_error(errors=str(e), status_code=HTTPStatus.PAYMENT_REQUIRED)
        except (DepositOverLimit, DepositMismatch) as e:
            return api_error(errors=str(e), status_code=HTTPStatus.CONFLICT)
        except UserDepositNotConfigured as e:
            return api_error(errors=str(e), status_code=HTTPStatus.NOT_FOUND)
        except RaidenRecoverableError as e:
            return api_error(errors=str(e), status_code=HTTPStatus.CONFLICT)
        return api_response(status_code=HTTPStatus.OK, result=dict(transaction_hash=encode_hex(transaction_hash)))

    def _plan_withdraw_from_udc(self, planned_withdraw_amount: int) -> Response:
        log.debug('Planning a withdraw from UDC', node=self.checksum_address, planned_withdraw_amount=planned_withdraw_amount)
        try:
            transaction_hash, planned_withdraw_block_number = self.raiden_api.plan_udc_withdraw(planned_withdraw_amount)
        except InsufficientEth as e:
            return api_error(errors=str(e), status_code=HTTPStatus.PAYMENT_REQUIRED)
        except WithdrawMismatch as e:
            return api_error(errors=str(e), status_code=HTTPStatus.CONFLICT)
        except UserDepositNotConfigured as e:
            return api_error(errors=str(e), status_code=HTTPStatus.NOT_FOUND)
        except RaidenRecoverableError as e:
            return api_error(errors=str(e), status_code=HTTPStatus.CONFLICT)
        result = dict(transaction_hash=encode_hex(transaction_hash), planned_withdraw_block_number=planned_withdraw_block_number)
        return api_response(status_code=HTTPStatus.OK, result=result)

    def _withdraw_from_udc(self, amount: int) -> Response:
        log.debug('Withdraw from UDC', node=self.checksum_address, amount=amount)
        try:
            transaction_hash = self.raiden_api.withdraw_from_udc(amount)
        except InsufficientEth as e:
            return api_error(errors=str(e), status_code=HTTPStatus.PAYMENT_REQUIRED)
        except WithdrawMismatch as e:
            return api_error(errors=str(e), status_code=HTTPStatus.CONFLICT)
        except UserDepositNotConfigured as e:
            return api_error(errors=str(e), status_code=HTTPStatus.NOT_FOUND)
        except RaidenRecoverableError as e:
            return api_error(errors=str(e), status_code=HTTPStatus.CONFLICT)
        return api_response(status_code=HTTPStatus.OK, result=dict(transaction_hash=encode_hex(transaction_hash)))

    def send_udc_transaction(
        self,
        total_deposit: Optional[int] = None,
        planned_withdraw_amount: Optional[int] = None,
        withdraw_amount: Optional[int] = None,
    ) -> Response:
        log.debug('Sending UDC transaction', node=self.checksum_address, total_deposit=total_deposit, planned_withdraw_amount=planned_withdraw_amount, withdraw_amount=withdraw_amount)
        if total_deposit is not None and planned_withdraw_amount is not None:
            return api_error(errors='Cannot deposit to UDC and plan a withdraw at the same time', status_code=HTTPStatus.BAD_REQUEST)
        if total_deposit is not None and withdraw_amount is not None:
            return api_error(errors='Cannot deposit to UDC and withdraw at the same time', status_code=HTTPStatus.BAD_REQUEST)
        if withdraw_amount is not None and planned_withdraw_amount is not None:
            return api_error(errors='Cannot withdraw from UDC and plan a withdraw at the same time', status_code=HTTPStatus.BAD_REQUEST)
        empty_request: bool = total_deposit is None and planned_withdraw_amount is None and withdraw_amount is None
        if empty_request:
            return api_error(errors="Nothing to do. Should either provide 'total_deposit', 'planned_withdraw_amount' or 'withdraw_amount' argument", status_code=HTTPStatus.BAD_REQUEST)
        if total_deposit is not None:
            result = self._deposit_to_udc(total_deposit)
        elif planned_withdraw_amount is not None:
            result = self._plan_withdraw_from_udc(planned_withdraw_amount)
        elif withdraw_amount is not None:
            result = self._withdraw_from_udc(withdraw_amount)
        return result

    def get_status(self) -> Response:
        if self.available:
            return api_response(result=dict(status='ready'))
        else:
            to_sync = blocks_to_sync(self.rpc_client)
            if to_sync > 0:
                return api_response(result=dict(status='syncing', blocks_to_sync=to_sync))
            else:
                return api_response(result=dict(status='unavailable'))

    def shutdown(self) -> Response:
        shutdown_greenlet = spawn_named('trigger shutdown', self.raiden_api.shutdown)
        shutdown_greenlet.link_exception(self.raiden_api.raiden.on_error)
        return api_response(result=dict(status='shutdown'))