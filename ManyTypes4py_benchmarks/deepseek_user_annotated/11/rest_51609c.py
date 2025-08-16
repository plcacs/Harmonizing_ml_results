import errno
import logging
import socket
from hashlib import sha256
from http import HTTPStatus
from typing import Type, Optional, Dict, List, Any, cast, Union, Tuple

import gevent
import gevent.pool
import opentracing
import structlog
from eth_utils import encode_hex
from flask import Flask, Response, request, send_from_directory, url_for
from flask.json import jsonify
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
    MYPY_ANNOTATION,
    Address,
    BlockTimeout,
    Endpoint,
    PaymentAmount,
    PaymentID,
    Secret,
    SecretHash,
    TargetAddress,
    TokenAddress,
    TokenAmount,
    TokenNetworkRegistryAddress,
    WithdrawAmount,
    typecheck,
)

log = structlog.get_logger(__name__)

CHANNEL_NETWORK_STATE = "network_state"
URLS_V1: List[Tuple[str, Type[Any], ...] = [
    ("/address", AddressResource),
    ("/version", VersionResource),
    ("/settings", NodeSettingsResource),
    ("/contracts", ContractsResource),
    ("/channels", ChannelsResource),
    ("/channels/<hexaddress:token_address>", ChannelsResourceByTokenAddress),
    (
        "/channels/<hexaddress:token_address>/<hexaddress:partner_address>",
        ChannelsResourceByTokenAndPartnerAddress,
    ),
    ("/connections/<hexaddress:token_address>", ConnectionsResource),
    ("/connections", ConnectionsInfoResource),
    ("/notifications", NotificationsResource),
    ("/payments", PaymentEventsResource, "paymentresource"),
    ("/payments/<hexaddress:token_address>", PaymentEventsResource, "token_paymentresource"),
    (
        "/payments/<hexaddress:token_address>/<hexaddress:target_address>",
        PaymentResource,
        "token_target_paymentresource",
    ),
    ("/tokens", TokensResource),
    ("/tokens/<hexaddress:token_address>/partners", PartnersResourceByTokenAddress),
    ("/tokens/<hexaddress:token_address>", RegisterTokenResource),
    ("/pending_transfers", PendingTransfersResource, "pending_transfers_resource"),
    (
        "/pending_transfers/<hexaddress:token_address>",
        PendingTransfersResourceByTokenAddress,
        "pending_transfers_resource_by_token",
    ),
    (
        "/pending_transfers/<hexaddress:token_address>/<hexaddress:partner_address>",
        PendingTransfersResourceByTokenAndPartnerAddress,
        "pending_transfers_resource_by_token_and_partner",
    ),
    ("/user_deposit", UserDepositResource),
    ("/status", StatusResource),
    ("/shutdown", ShutdownResource),
    ("/_debug/raiden_events", RaidenInternalEventsResource),
    ("/_testing/tokens/<hexaddress:token_address>/mint", MintTokenResource, "tokensmintresource"),
]


def endpoint_not_found(e: Any) -> Response:
    errors = ["invalid endpoint"]
    if isinstance(e, InvalidEndpoint):
        errors.append(e.description)
    return api_error(errors, HTTPStatus.NOT_FOUND)


def hexbytes_to_str(map_: Dict) -> None:
    """Converts values that are of type `HexBytes` to strings."""
    for k, v in map_.items():
        if isinstance(v, HexBytes):
            map_[k] = encode_hex(v)


def encode_byte_values(map_: Dict) -> None:
    """Converts values that are of type `bytes` to strings."""
    for k, v in map_.items():
        if isinstance(v, bytes):
            map_[k] = encode_hex(v)


def encode_object_to_str(map_: Dict) -> None:
    for k, v in map_.items():
        if isinstance(v, int) or k == "args":
            continue
        if not isinstance(v, str):
            map_[k] = repr(v)


def restapi_setup_urls(flask_api_context: Api, rest_api: "RestAPI", urls: List[Tuple[str, Type[Any], ...]]) -> None:
    for url_tuple in urls:
        if len(url_tuple) == 2:
            route, resource_cls = url_tuple
            endpoint = resource_cls.__name__.lower()
        elif len(url_tuple) == 3:
            route, resource_cls, endpoint = url_tuple
        else:
            raise ValueError(f"Invalid URL format: {url_tuple!r}")
        flask_api_context.add_resource(
            resource_cls,
            route,
            resource_class_kwargs={"rest_api_object": rest_api},
            endpoint=endpoint,
        )


def restapi_setup_type_converters(
    flask_app: Flask, names_to_converters: Dict[str, Type[BaseConverter]]
) -> None:
    flask_app.url_map.converters.update(names_to_converters)


class APIServer(Runnable):  # pragma: no unittest
    """
    Runs the API-server that routes the endpoint to the resources.
    The API is wrapped in multiple layers, and the Server should be invoked this way::

        # instance of the raiden-api
        raiden_api = RaidenAPI(...)

        # wrap the raiden-api with rest-logic and encoding
        rest_api = RestAPI(raiden_api)

        # create the server and link the api-endpoints with flask / flask-restful middleware
        api_server = APIServer(rest_api, {'host: '127.0.0.1', 'port': 5001})

        # run the server greenlet
        api_server.start()
    """

    _api_prefix = "/api/1"

    def __init__(
        self,
        rest_api: "RestAPI",
        config: RestApiConfig,
        eth_rpc_endpoint: Optional[str] = None,
    ) -> None:
        super().__init__()
        if rest_api.version != 1:
            raise ValueError(f"Invalid api version: {rest_api.version}")
        self._api_prefix = f"/api/v{rest_api.version}"

        flask_app = Flask(__name__)
        if config.cors_domain_list:
            CORS(flask_app, origins=config.cors_domain_list)

        if config.enable_tracing:
            FlaskTracing(opentracing.tracer, trace_all_requests=True, app=flask_app)

        if eth_rpc_endpoint:
            if not eth_rpc_endpoint.startswith("http"):
                eth_rpc_endpoint = f"http://{eth_rpc_endpoint}"
            flask_app.config["WEB3_ENDPOINT"] = eth_rpc_endpoint

        blueprint = create_blueprint()
        flask_api_context = Api(blueprint, prefix=self._api_prefix)

        restapi_setup_type_converters(
            flask_app, {"hexaddress": cast(Type[BaseConverter], HexAddressConverter)}
        )

        restapi_setup_urls(flask_api_context, rest_api, URLS_V1)

        self.stop_event = Event()

        self.config = config
        self.rest_api = rest_api
        self.flask_app = flask_app
        self.blueprint = blueprint
        self.flask_api_context = flask_api_context
        self.wsgiserver: Optional[WSGIServer] = None

        self.flask_app.register_blueprint(self.blueprint)
        self.flask_app.config["WEBUI_PATH"] = RAIDEN_WEBUI_PATH

        self.flask_app.register_error_handler(HTTPStatus.NOT_FOUND, endpoint_not_found)
        self.flask_app.register_error_handler(Exception, self.unhandled_exception)

        # FIXME: There's a type annotation bug on flask, fixed on this commit:
        # https://github.com/pallets/flask/commit/a960236117442bec67f89c30dfa014e05483da5a
        # Remove type ignore on the next flask release this commit is included
        self.flask_app.before_request(self._check_shutdown_before_handle_request)

        # needed so flask_restful propagates the exception to our error handler above
        # or else, it'll replace it with a E500 response
        self.flask_app.config["PROPAGATE_EXCEPTIONS"] = True

        if config.web_ui_enabled:
            for route in ("/ui/<path:file_name>", "/ui", "/ui/", "/index.html", "/"):
                self.flask_app.add_url_rule(
                    route, route, view_func=self._serve_webui, methods=("GET",)
                )

    def _check_shutdown_before_handle_request(self) -> Optional[Response]:
        """
        We don't want to handle requests when shutting down
        When the `before_request` hook returns a value, the request will not be processed further
        """
        if self.stop_event.is_set():
            return api_error("Raiden API is shutting down", HTTPStatus.SERVICE_UNAVAILABLE)

        return None

    def _serve_webui(self, file_name: str = "index.html") -> Response:
        try:
            if not file_name:
                raise NotFound

            web3 = self.flask_app.config.get("WEB3_ENDPOINT")
            if "config." in file_name and file_name.endswith(".json"):
                environment_type = (
                    self.rest_api.raiden_api.raiden.config.environment_type.name.lower()
                )
                config = {
                    "raiden": self._api_prefix,
                    "web3": web3,
                    "settle_timeout": self.rest_api.raiden_api.raiden.config.settle_timeout,
                    "reveal_timeout": self.rest_api.raiden_api.raiden.config.reveal_timeout,
                    "environment_type": environment_type,
                }

                # if raiden sees eth rpc endpoint as localhost, replace it by Host header,
                # which is the hostname by which the client/browser sees/access the raiden node
                host_header = request.headers.get("Host")
                if web3 and host_header:
                    web3_host, web3_port = split_endpoint(web3)
                    if web3_host in ("localhost", "127.0.0.1"):
                        host, _ = split_endpoint(Endpoint(host_header))
                        web3_port_str = ""
                        if web3_port:
                            web3_port_str = f":{web3_port}"
                        web3 = f"http://{host}{web3_port_str}"
                        config["web3"] = web3

                response = jsonify(config)
            else:
                response = send_from_directory(self.flask_app.config["WEBUI_PATH"], file_name)
        except (NotFound, AssertionError):
            if file_name.endswith(".json"):
                response = api_error(
                    "Service unavailable, try again later", HTTPStatus.SERVICE_UNAVAILABLE
                )
            else:
                response = send_from_directory(self.flask_app.config["WEBUI_PATH"], "index.html")
        return response

    def _run(self) -> None:  # type: ignore
        try:
            # stop may have been executed before _run was scheduled, in this
            # case wsgiserver will be None
            if self.wsgiserver is not None:
                self.wsgiserver.serve_forever()
        except gevent.GreenletExit:  # pylint: disable=try-except-raise
            raise
        except Exception:
            self.stop()  # ensure cleanup and wait on subtasks
            raise

    def start(self) -> None:
        self.stop_event.clear()

        log.debug("REST API starting", host=self.config.host, port=self.config.port)

        # WSGI expects an stdlib logger. With structlog there's conflict of
        # method names. Rest unhandled exception will be re-raised here:
        wsgi_log = logging.getLogger(__name__ + ".pywsgi")

        # server.stop() clears the handle and the pool, this is okay since a
        # new WSGIServer is created on each start
        pool = gevent.pool.Pool()
        wsgiserver = WSGIServer(
            (self.config.host, self.config.port),
            self.flask_app,
            log=wsgi_log,
            error_log=wsgi_log,
            spawn=pool,
        )

        try:
            wsgiserver.init_socket()
        except socket.error as e:
            if e.errno == errno.EADDRINUSE:
                raise APIServerPortInUseError(f"{self.config.host}:{self.config.port}")
            raise

        self.wsgiserver = wsgiserver

        log.debug("REST API started", host=self.config.host, port=self.config.port)

        super().start()

    def stop(self) -> None:
        # Set event to prevent accepting new requests
        self.stop_event.set()

        log.debug(
            "REST API stopping",
            host=self.config.host,
            port=self.config.port,
            node=self.rest_api.checksum_address,
        )

        if self.wsgiserver is not None:
            # It is very important to have a timeout here, the existing endpoints only return
            # once the operation is completed, which for a deposit it means waiting for the
            # transaction to be mined and confirmed, which can potentially take a few minutes.
            # If a timeout is not provided here, that would lead to a very and unpredictable
            # shutdown timeout, which leads to problems with our scenario player tooling.

            self.wsgiserver.stop(timeout=5)
            self.wsgiserver = None

        log.debug(
            "REST API stopped",
            host=self.config.host,
            port=self.config.port,
            node=self.rest_api.checksum_address,
        )

    def unhandled_exception(self, exception: Exception) -> Response:
        """Flask.errorhandler when an exception wasn't correctly handled"""
        log.critical(
            "Unhandled exception when processing endpoint request",
            exc_info=True,
            node=self.rest_api.checksum_address,
        )
        self.greenlet.kill(exception)
        return api_error([str(exception)], HTTPStatus.INTERNAL_SERVER_ERROR)


class RestAPI:  # pragma: no unittest
    """
    This wraps around the actual RaidenAPI in api/python.
    It will provide the additional, neccessary RESTful logic and
    the proper JSON-encoding of the Objects provided by the RaidenAPI
    """

    version = 1

    def __init__(self, raiden_api: Optional[RaidenAPI] = None, rpc_client: Optional[JSONRPCClient] = None) -> None:
        self._rpc_client = rpc_client
        self._raiden_api = raiden_api

        self.channel_schema = ChannelStateSchema()
        self.address_list_schema = AddressListSchema()
        self.partner_per_token_list_schema = PartnersPerTokenListSchema()
        self.payment_schema = PaymentSchema()
        self.sent_success_payment_schema = EventPaymentSentSuccessSchema()
        self.received_success_payment_schema = EventPaymentReceivedSuccessSchema()
        self.failed_payment_schema = EventPaymentSentFailedSchema()
        self.notification_schema = NotificationSchema()

    @property
    def rpc_client(self) -> JSONRPCClient:
        assert self._rpc_client is not None, "rpc_client accessed but not initialized."
        return self._rpc_client

    @property
    def checksum_address(self) -> Optional[str]:
        return to_checksum_address(self.raiden_api.address