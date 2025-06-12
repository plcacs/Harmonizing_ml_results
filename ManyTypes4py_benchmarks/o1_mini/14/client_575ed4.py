from urllib.parse import quote, urljoin, urlparse
import apistar
import typesystem
from apistar import exceptions
from apistar.client import transports
from typing import Any, Dict, Optional, Tuple

class Client:

    document: Any
    transport: transports.HTTPTransport

    def __init__(
        self,
        schema: Any,
        format: Optional[str] = None,
        encoding: Optional[str] = None,
        auth: Optional[Any] = None,
        decoders: Optional[Any] = None,
        encoders: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        session: Optional[Any] = None,
        allow_cookies: bool = True
    ) -> None:
        self.document = apistar.validate(schema, format=format, encoding=encoding)
        self.transport = self.init_transport(auth, decoders, encoders, headers, session, allow_cookies)

    def init_transport(
        self,
        auth: Optional[Any] = None,
        decoders: Optional[Any] = None,
        encoders: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        session: Optional[Any] = None,
        allow_cookies: bool = True
    ) -> transports.HTTPTransport:
        return transports.HTTPTransport(
            auth=auth,
            decoders=decoders,
            encoders=encoders,
            headers=headers,
            session=session,
            allow_cookies=allow_cookies
        )

    def lookup_operation(self, operation_id: str) -> Any:
        for item in self.document.walk_links():
            if item.link.name == operation_id:
                return item.link
        text = f'Operation ID "{operation_id}" not found in schema.'
        message = exceptions.ErrorMessage(text=text, code='invalid-operation')
        raise exceptions.ClientError(messages=[message])

    def get_url(self, link: Any, params: Dict[str, Any]) -> str:
        url = urljoin(self.document.url, link.url)
        scheme = urlparse(url).scheme.lower()
        if not scheme:
            text = f"URL missing scheme '{url}'."
            message = exceptions.ErrorMessage(text=text, code='invalid-url')
            raise exceptions.ClientError(messages=[message])
        if scheme not in self.transport.schemes:
            text = f"Unsupported URL scheme '{scheme}'."
            message = exceptions.ErrorMessage(text=text, code='invalid-url')
            raise exceptions.ClientError(messages=[message])
        for field in link.get_path_fields():
            value = str(params[field.name])
            if f'{{{field.name}}}' in url:
                url = url.replace(f'{{{field.name}}}', quote(value, safe=''))
            elif f'{{+{field.name}}}' in url:
                url = url.replace(f'{{+{field.name}}}', quote(value, safe='/'))
        return url

    def get_query_params(self, link: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            field.name: params[field.name]
            for field in link.get_query_fields()
            if field.name in params
        }

    def get_content_and_encoding(self, link: Any, params: Dict[str, Any]) -> Tuple[Optional[Any], Optional[str]]:
        body_field = link.get_body_field()
        if body_field and body_field.name in params:
            return (params[body_field.name], link.encoding)
        return (None, None)

    def request(self, operation_id: str, **params: Any) -> Any:
        link = self.lookup_operation(operation_id)
        validator = typesystem.Object(
            properties={field.name: typesystem.Any() for field in link.fields},
            required=[field.name for field in link.fields if field.required],
            additional_properties=False
        )
        try:
            validator.validate(params)
        except typesystem.ValidationError as exc:
            raise exceptions.ClientError(messages=exc.messages()) from None
        method: str = link.method
        url: str = self.get_url(link, params)
        query_params: Dict[str, Any] = self.get_query_params(link, params)
        content: Optional[Any]
        encoding: Optional[str]
        content, encoding = self.get_content_and_encoding(link, params)
        return self.transport.send(
            method,
            url,
            query_params=query_params,
            content=content,
            encoding=encoding
        )
