import http
import requests
from apistar import exceptions
from apistar.client import decoders, encoders

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

class BaseTransport:
    schemes: list[str] = None

    def send(self, method: str, url: str, query_params: dict[str, str] | None = None, content: bytes | str | None = None, encoding: str | None = None) -> None:
        raise NotImplementedError()

class HTTPTransport(BaseTransport):
    schemes: list[str] = ['http', 'https']
    default_decoders: list[decoders.Decoder] = [decoders.JSONDecoder(), decoders.TextDecoder(), decoders.DownloadDecoder()]
    default_encoders: list[encoders.Encoder] = [encoders.JSONEncoder(), encoders.MultiPartEncoder(), encoders.URLEncodedEncoder()]

    def __init__(self, auth: tuple[str, str] | None = None, decoders: list[decoders.Decoder] | None = None, encoders: list[encoders.Encoder] | None = None, headers: dict[str, str] | None = None, session: requests.Session | None = None, allow_cookies: bool = True) -> None:
        from apistar import __version__
        if session is None:
            session = requests.Session()
        if auth is not None:
            session.auth = auth
        if not allow_cookies:
            session.cookies.set_policy(BlockAllCookies())
        self.session: requests.Session = session
        self.decoders: list[decoders.Decoder] = list(decoders) if decoders else list(self.default_decoders)
        self.encoders: list[encoders.Encoder] = list(encoders) if encoders else list(self.default_encoders)
        self.headers: dict[str, str] = {'accept': ', '.join([decoder.media_type for decoder in self.decoders]), 'user-agent': f'apistar {__version__}'}
        if headers:
            self.headers.update({key.lower(): value for key, value in headers.items()})

    def send(self, method: str, url: str, query_params: dict[str, str] | None = None, content: bytes | str | None = None, encoding: str | None = None) -> str:
        options = self.get_request_options(query_params, content, encoding)
        response = self.session.request(method, url, **options)
        result = self.decode_response_content(response)
        if 400 <= response.status_code <= 599:
            title = f'{response.status_code} {response.reason}'
            raise exceptions.ErrorResponse(title=title, status_code=response.status_code, content=result)
        return result

    def get_encoder(self, encoding: str) -> encoders.Encoder:
        """
        Given the value of the encoding, return the appropriate encoder for
        handling the request content.
        """
        content_type = encoding.split(';')[0].strip().lower()
        main_type = content_type.split('/')[0] + '/*'
        wildcard_type = '*/*'
        for codec in self.encoders:
            if codec.media_type in (content_type, main_type, wildcard_type):
                return codec
        text = f"Unsupported encoding '{encoding}' for request."
        message = exceptions.ErrorMessage(text=text, code='cannot-encode-request')
        raise exceptions.ClientError(messages=[message])

    def get_decoder(self, content_type: str | None) -> decoders.Decoder:
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
        text = f"Unsupported encoding '{content_type}' in response Content-Type header."
        message = exceptions.ErrorMessage(text=text, code='cannot-decode-response')
        raise exceptions.ClientError(messages=[message])

    def get_request_options(self, query_params: dict[str, str] | None, content: bytes | str | None, encoding: str | None) -> dict[str, str | dict[str, str]]:
        """
        Return the 'options' for sending the outgoing request.
        """
        options: dict[str, str | dict[str, str]] = {'headers': dict(self.headers), 'params': query_params}
        if content is None:
            return options
        encoder = self.get_encoder(encoding)
        encoder.encode(options, content)
        return options

    def decode_response_content(self, response: requests.Response) -> str:
        """
        Given an HTTP response, return the decoded data.
        """
        if not response.content:
            return None
        content_type = response.headers.get('content-type')
        decoder = self.get_decoder(content_type)
        return decoder.decode(response)
