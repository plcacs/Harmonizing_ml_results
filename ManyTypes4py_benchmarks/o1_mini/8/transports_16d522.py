import http
import requests
from typing import Any, Dict, List, Optional, Union
from apistar import exceptions
from apistar.client import decoders, encoders
from requests.auth import AuthBase
from requests.cookies import CookiePolicy
from requests.sessions import Session
from requests.models import Response


class BlockAllCookies(http.cookiejar.CookiePolicy):
    """
    A cookie policy that rejects all cookies.
    Used to override the default `requests` behavior.
    """
    return_ok: bool = False
    set_ok: bool = False
    domain_return_ok: bool = False
    path_return_ok: bool = False
    netscape: bool = True
    rfc2965: bool = False
    hide_cookie2: bool = False

    def return_ok(self, *args, **kwargs) -> bool:
        return False

    def set_ok(self, *args, **kwargs) -> bool:
        return False

    def domain_return_ok(self, *args, **kwargs) -> bool:
        return False

    def path_return_ok(self, *args, **kwargs) -> bool:
        return False


class BaseTransport:
    schemes: Optional[List[str]] = None

    def send(
        self,
        method: str,
        url: str,
        query_params: Optional[Dict[str, Any]] = None,
        content: Optional[Any] = None,
        encoding: Optional[str] = None
    ) -> Any:
        raise NotImplementedError()


class HTTPTransport(BaseTransport):
    schemes: List[str] = ['http', 'https']
    default_decoders: List[decoders.Decoder] = [
        decoders.JSONDecoder(),
        decoders.TextDecoder(),
        decoders.DownloadDecoder()
    ]
    default_encoders: List[encoders.Encoder] = [
        encoders.JSONEncoder(),
        encoders.MultiPartEncoder(),
        encoders.URLEncodedEncoder()
    ]

    def __init__(
        self,
        auth: Optional[AuthBase] = None,
        decoders: Optional[List[decoders.Decoder]] = None,
        encoders: Optional[List[encoders.Encoder]] = None,
        headers: Optional[Dict[str, str]] = None,
        session: Optional[Session] = None,
        allow_cookies: bool = True
    ) -> None:
        from apistar import __version__
        if session is None:
            session = requests.Session()
        if auth is not None:
            session.auth = auth
        if not allow_cookies:
            session.cookies.set_policy(BlockAllCookies())
        self.session: Session = session
        self.decoders: List[decoders.Decoder] = list(decoders) if decoders else list(self.default_decoders)
        self.encoders: List[encoders.Encoder] = list(encoders) if encoders else list(self.default_encoders)
        self.headers: Dict[str, str] = {
            'accept': ', '.join([decoder.media_type for decoder in self.decoders]),
            'user-agent': f'apistar {__version__}'
        }
        if headers:
            self.headers.update({key.lower(): value for key, value in headers.items()})

    def send(
        self,
        method: str,
        url: str,
        query_params: Optional[Dict[str, Any]] = None,
        content: Optional[Any] = None,
        encoding: Optional[str] = None
    ) -> Any:
        options: Dict[str, Any] = self.get_request_options(query_params, content, encoding)
        response: Response = self.session.request(method, url, **options)
        result: Any = self.decode_response_content(response)
        if 400 <= response.status_code <= 599:
            title: str = f'{response.status_code} {response.reason}'
            raise exceptions.ErrorResponse(
                title=title,
                status_code=response.status_code,
                content=result
            )
        return result

    def get_encoder(self, encoding: str) -> encoders.Encoder:
        """
        Given the value of the encoding, return the appropriate encoder for
        handling the request content.
        """
        content_type: str = encoding.split(';')[0].strip().lower()
        main_type: str = content_type.split('/')[0] + '/*'
        wildcard_type: str = '*/*'
        for codec in self.encoders:
            if codec.media_type in (content_type, main_type, wildcard_type):
                return codec
        text: str = f"Unsupported encoding '{encoding}' for request."
        message: exceptions.ErrorMessage = exceptions.ErrorMessage(
            text=text,
            code='cannot-encode-request'
        )
        raise exceptions.ClientError(messages=[message])

    def get_decoder(self, content_type: Optional[str] = None) -> decoders.Decoder:
        """
        Given the value of a 'Content-Type' header, return the appropriate
        decoder for handling the response content.
        """
        if content_type is None:
            return self.decoders[0]
        content_type = content_type.split(';')[0].strip().lower()
        main_type = content_type.split('/')[0] + '/*'
        wildcard_type = '*/*'
        for codec in self.decoders:
            if codec.media_type in (content_type, main_type, wildcard_type):
                return codec
        text: str = f"Unsupported encoding '{content_type}' in response Content-Type header."
        message: exceptions.ErrorMessage = exceptions.ErrorMessage(
            text=text,
            code='cannot-decode-response'
        )
        raise exceptions.ClientError(messages=[message])

    def get_request_options(
        self,
        query_params: Optional[Dict[str, Any]] = None,
        content: Optional[Any] = None,
        encoding: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Return the 'options' for sending the outgoing request.
        """
        options: Dict[str, Any] = {'headers': dict(self.headers), 'params': query_params}
        if content is None:
            return options
        if encoding is None:
            raise ValueError("Encoding must be provided when content is not None.")
        encoder: encoders.Encoder = self.get_encoder(encoding)
        encoder.encode(options, content)
        return options

    def decode_response_content(self, response: Response) -> Any:
        """
        Given an HTTP response, return the decoded data.
        """
        if not response.content:
            return None
        content_type: Optional[str] = response.headers.get('content-type')
        decoder: decoders.Decoder = self.get_decoder(content_type)
        return decoder.decode(response)
