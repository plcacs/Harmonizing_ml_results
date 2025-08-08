import http
import requests
from apistar import exceptions
from apistar.client import decoders, encoders
from typing import List, Dict, Any, Optional

class BlockAllCookies(http.cookiejar.CookiePolicy):
    return_ok: Any
    set_ok: Any
    domain_return_ok: Any
    path_return_ok: Any

class BaseTransport:
    schemes: Optional[List[str]]

    def send(self, method: str, url: str, query_params: Optional[Dict[str, Any]] = None, content: Any = None, encoding: Optional[str] = None) -> None:
        raise NotImplementedError()

class HTTPTransport(BaseTransport):
    schemes: List[str]
    default_decoders: List[Any]
    default_encoders: List[Any]

    def __init__(self, auth: Any = None, decoders: Optional[List[Any]] = None, encoders: Optional[List[Any]] = None, headers: Optional[Dict[str, str]] = None, session: Any = None, allow_cookies: bool = True) -> None:
    
    def send(self, method: str, url: str, query_params: Optional[Dict[str, Any]] = None, content: Any = None, encoding: Optional[str] = None) -> Any:
    
    def get_encoder(self, encoding: str) -> Any:
    
    def get_decoder(self, content_type: Optional[str] = None) -> Any:
    
    def get_request_options(self, query_params: Optional[Dict[str, Any]] = None, content: Any = None, encoding: Optional[str] = None) -> Dict[str, Any]:
    
    def decode_response_content(self, response: Any) -> Any:
