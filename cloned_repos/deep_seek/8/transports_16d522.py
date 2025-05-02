import http
import requests
from typing import Optional, List, Dict, Any, Union, Type
from apistar import exceptions
from apistar.client import decoders, encoders

class BlockAllCookies(http.cookiejar.CookiePolicy):
    """
    A cookie policy that rejects all cookies.
    Used to override the default `requests` behavior.
    """
    return_ok: Any = set_ok = domain_return_ok = path_return_ok = lambda self, *args, **kwargs: False
    netscape: bool = True
    rfc2965: bool = hide_cookie2: bool = False

class BaseTransport:
    schemes: Optional[List[str]] = None

    def send(self, method: str, url: str, query_params: Optional[Dict[str, Any]] = None, content: Optional[Any] = None, encoding: Optional[str] = None) -> Any:
        raise NotImplementedError()

class HTTPTransport(BaseTransport):
    schemes: List[str] = ['http', 'https']
    default_decoders: List[Any] = [decoders.JSONDecoder(), decoders.TextDecoder(), decoders.DownloadDecoder()]
    default_encoders: List[Any] = [encoders.JSONEncoder(), encoders.MultiPartEncoder(), encoders.URLEncodedEncoder()]

    def __init__(
        self,
        auth: Optional[Any] = None,
        decoders: Optional[List[Any]] = None,
        encoders: Optional[List[Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        session: Optional[requests.Session] = None,
        allow_cookies: bool = True
    ) -> None:
        from apistar import __version__
        if session is None:
            session = requests.Session()
        if auth is not None:
            session.auth = auth
        if not allow_cookies:
            session.cookies.set_policy(BlockAllCookies())
        self.session: requests.Session = session
        self.decoders: List[Any] = list(decoders) if decoders else list(self.default_decoders)
        self.encoders: List[Any] = list(encoders) if encoders else list(self.default_encoders)
        self.headers: Dict[str, str] = {
            'accept': ', '.join([decoder.media_type for decoder in self.decoders]),
            'user-agent': 'apistar %s' % __version__
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
        response: requests.Response = self.session.request(method, url, **options)
        result: Any = self.decode_response_content(response)
        if 400 <= response.status_code <= 599:
            title: str = '%d %s' % (response.status_code, response.reason)
            raise exceptions.ErrorResponse(title=title, status_code=response.status_code, content=result)
        return result

    def get_encoder(self, encoding: str) -> Any:
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
        text: str = "Unsupported encoding '%s' for request." % encoding
        message: exceptions.ErrorMessage = exceptions.ErrorMessage(text=text, code='cannot-encode-request')
        raise exceptions.ClientError(messages=[message])

    def get_decoder(self, content_type: Optional[str] = None) -> Any:
        """
        Given the value of a 'Content-Type' header, return the appropriate
        decoder for handling the response content.
        """
        if content_type is None:
            return self.decoders[0]
        content_type = content_type.split(';')[0].strip().lower()
        main_type: str = content_type.split('/')[0] + '/*'
        wildcard_type: str = '*/*'
        for codec in self.decoders:
            if codec.media_type in (content_type, main_type, wildcard_type):
                return codec
        text: str = "Unsupported encoding '%s' in response Content-Type header." % content_type
        message: exceptions.ErrorMessage = exceptions.ErrorMessage(text=text, code='cannot-decode-response')
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
        encoder: Any = self.get_encoder(encoding)
        encoder.encode(options, content)
        return options

    def decode_response_content(self, response: requests.Response) -> Any:
        """
        Given an HTTP response, return the decoded data.
        """
        if not response.content:
            return None
        content_type: Optional[str] = response.headers.get('content-type')
        decoder: Any = self.get_decoder(content_type)
        return decoder.decode(response)
