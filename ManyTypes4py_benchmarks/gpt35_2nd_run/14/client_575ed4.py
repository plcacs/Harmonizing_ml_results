from urllib.parse import quote, urljoin, urlparse
import apistar
import typesystem
from apistar import exceptions
from apistar.client import transports

class Client:

    def __init__(self, schema: typesystem.Object, format: str = None, encoding: str = None, auth: str = None, decoders: dict = None, encoders: dict = None, headers: dict = None, session: typesystem.Object = None, allow_cookies: bool = True) -> None:
        self.document: typesystem.Object = apistar.validate(schema, format=format, encoding=encoding)
        self.transport: transports.HTTPTransport = self.init_transport(auth, decoders, encoders, headers, session, allow_cookies)

    def init_transport(self, auth: str = None, decoders: dict = None, encoders: dict = None, headers: dict = None, session: typesystem.Object = None, allow_cookies: bool = True) -> transports.HTTPTransport:
        return transports.HTTPTransport(auth=auth, decoders=decoders, encoders=encoders, headers=headers, session=session, allow_cookies=allow_cookies)

    def lookup_operation(self, operation_id: str) -> apistar.Link:
        for item in self.document.walk_links():
            if item.link.name == operation_id:
                return item.link
        text: str = 'Operation ID "%s" not found in schema.' % operation_id
        message: exceptions.ErrorMessage = exceptions.ErrorMessage(text=text, code='invalid-operation')
        raise exceptions.ClientError(messages=[message])

    def get_url(self, link: apistar.Link, params: dict) -> str:
        url: str = urljoin(self.document.url, link.url)
        scheme: str = urlparse(url).scheme.lower()
        if not scheme:
            text: str = "URL missing scheme '%s'." % url
            message: exceptions.ErrorMessage = exceptions.ErrorMessage(text=text, code='invalid-url')
            raise exceptions.ClientError(messages=[message])
        if scheme not in self.transport.schemes:
            text: str = "Unsupported URL scheme '%s'." % scheme
            message: exceptions.ErrorMessage = exceptions.ErrorMessage(text=text, code='invalid-url')
            raise exceptions.ClientError(messages=[message])
        for field in link.get_path_fields():
            value: str = str(params[field.name])
            if '{%s}' % field.name in url:
                url = url.replace('{%s}' % field.name, quote(value, safe=''))
            elif '{+%s}' % field.name in url:
                url = url.replace('{+%s}' % field.name, quote(value, safe='/'))
        return url

    def get_query_params(self, link: apistar.Link, params: dict) -> dict:
        return {field.name: params[field.name] for field in link.get_query_fields() if field.name in params}

    def get_content_and_encoding(self, link: apistar.Link, params: dict) -> tuple:
        body_field: typesystem.Field = link.get_body_field()
        if body_field and body_field.name in params:
            return (params[body_field.name], link.encoding)
        return (None, None)

    def request(self, operation_id: str, **params: dict) -> dict:
        link: apistar.Link = self.lookup_operation(operation_id)
        validator: typesystem.Object = typesystem.Object(properties={field.name: typesystem.Any() for field in link.fields}, required=[field.name for field in link.fields if field.required], additional_properties=False)
        try:
            validator.validate(params)
        except typesystem.ValidationError as exc:
            raise exceptions.ClientError(messages=exc.messages()) from None
        method: str = link.method
        url: str = self.get_url(link, params)
        query_params: dict = self.get_query_params(link, params)
        content, encoding = self.get_content_and_encoding(link, params)
        return self.transport.send(method, url, query_params=query_params, content=content, encoding=encoding)
