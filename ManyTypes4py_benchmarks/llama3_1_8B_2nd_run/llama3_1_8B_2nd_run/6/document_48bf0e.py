import collections
import re
import typing

LinkInfo = collections.namedtuple('LinkInfo', ['link', 'name', 'sections'])

class Document:
    def __init__(
        self,
        content: typing.Union[typing.List['Link'], typing.List['Section'], typing.List[typing.Any]],
        url: str = '',
        title: str = '',
        description: str = '',
        version: str = ''
    ):
        content = [] if content is None else list(content)
        seen_fields: typing.Set[str] = set()
        seen_sections: typing.Set[str] = set()
        for item in content:
            if isinstance(item, Link):
                msg = 'Link "%s" in Document must have a unique name.'
                assert item.name not in seen_fields, msg % item.name
                seen_fields.add(item.name)
            else:
                msg = 'Section "%s" in Document must have a unique name.'
                assert item.name not in seen_sections, msg % item.name
                seen_sections.add(item.name)
        self.content: typing.List[typing.Union[Link, Section]] = content
        self.url: str = url
        self.title: str = title
        self.description: str = description
        self.version: str = version

    def get_links(self) -> typing.List[Link]:
        return [item for item in self.content if isinstance(item, Link)]

    def get_sections(self) -> typing.List[Section]:
        return [item for item in self.content if isinstance(item, Section)]

    def walk_links(self, previous_sections: typing.Tuple[Section, ...] = ()) -> typing.List[LinkInfo]:
        link_info_list: typing.List[LinkInfo] = []
        sections: typing.Tuple[Section, ...] = previous_sections + (self,)
        for item in self.content:
            if isinstance(item, Link):
                link_info = LinkInfo(link=item, name=item.name, sections=sections)
                link_info_list.append(link_info)
            else:
                link_info_list.extend(item.walk_links(previous_sections=sections))
        return link_info_list

class Section:
    def __init__(
        self,
        name: str,
        content: typing.Union[typing.List[Link], typing.List[Section], typing.List[typing.Any]],
        title: str = '',
        description: str = ''
    ):
        content = [] if content is None else list(content)
        seen_fields: typing.Set[str] = set()
        seen_sections: typing.Set[str] = set()
        for item in content:
            if isinstance(item, Link):
                msg = 'Link "%s" in Section "%s" must have a unique name.'
                assert item.name not in seen_fields, msg % (item.name, name)
                seen_fields.add(item.name)
            else:
                msg = 'Section "%s" in Section "%s" must have a unique name.'
                assert item.name not in seen_sections, msg % (item.name, name)
                seen_sections.add(item.name)
        self.content: typing.List[typing.Union[Link, Section]] = content
        self.name: str = name
        self.title: str = title
        self.description: str = description

    def get_links(self) -> typing.List[Link]:
        return [item for item in self.content if isinstance(item, Link)]

    def get_sections(self) -> typing.List[Section]:
        return [item for item in self.content if isinstance(item, Section)]

    def walk_links(self, previous_sections: typing.Tuple[Section, ...] = ()) -> typing.List[LinkInfo]:
        link_info_list: typing.List[LinkInfo] = []
        sections: typing.Tuple[Section, ...] = previous_sections + (self,)
        for item in self.content:
            if isinstance(item, Link):
                name = ':'.join([section.name for section in sections] + [item.name])
                link_info = LinkInfo(link=item, name=name, sections=sections)
                link_info_list.append(link_info)
            else:
                link_info_list.extend(item.walk_links(previous_sections=sections))
        return link_info_list

class Link:
    """
    Links represent the actions that a client may perform.
    """
    def __init__(
        self,
        url: str,
        method: str,
        handler: typing.Optional[typing.Callable] = None,
        name: str = '',
        encoding: str = '',
        response: typing.Optional['Response'] = None,
        title: str = '',
        description: str = '',
        fields: typing.Union[typing.List['Field'], typing.List[typing.Any]] = None
    ):
        method = method.upper()
        fields = [] if fields is None else list(fields)
        url_path_names: typing.Set[str] = set([item.strip('{}').lstrip('+') for item in re.findall('{[^}]*}', url)])
        path_fields: typing.List[Field] = [field for field in fields if field.location == 'path']
        body_fields: typing.List[Field] = [field for field in fields if field.location == 'body']
        assert method in ('GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS', 'HEAD', 'TRACE')
        assert len(body_fields) < 2
        if body_fields:
            assert encoding
        for field in path_fields:
            assert field.name in url_path_names
        for path_name in url_path_names:
            if path_name not in [field.name for field in path_fields]:
                fields += [Field(name=path_name, location='path', required=True)]
        self.url: str = url
        self.method: str = method
        self.handler: typing.Optional[typing.Callable] = handler
        self.name: str = name if name else handler.__name__ if handler else ''
        self.encoding: str = encoding
        self.response: typing.Optional[Response] = response
        self.title: str = title
        self.description: str = description
        self.fields: typing.List[Field] = fields

    def get_path_fields(self) -> typing.List[Field]:
        return [field for field in self.fields if field.location == 'path']

    def get_query_fields(self) -> typing.List[Field]:
        return [field for field in self.fields if field.location == 'query']

    def get_body_field(self) -> typing.Optional[Field]:
        for field in self.fields:
            if field.location == 'body':
                return field
        return None

    def get_expanded_body(self) -> typing.Optional[typing.Dict[str, typing.Any]]:
        field = self.get_body_field()
        if field is None or not hasattr(field.schema, 'properties'):
            return None
        return field.schema.properties

class Field:
    def __init__(
        self,
        name: str,
        location: str,
        title: str = '',
        description: str = '',
        required: typing.Optional[bool] = None,
        schema: typing.Optional[typing.Dict[str, typing.Any]] = None,
        example: typing.Optional[typing.Any] = None
    ):
        assert location in ('path', 'query', 'body', 'cookie', 'header', 'formData')
        if required is None:
            required = True if location in ('path', 'body') else False
        if location == 'path':
            assert required, "May not set 'required=False' on path fields."
        self.name: str = name
        self.title: str = title
        self.description: str = description
        self.location: str = location
        self.required: bool = required
        self.schema: typing.Optional[typing.Dict[str, typing.Any]] = schema
        self.example: typing.Optional[typing.Any] = example

class Response:
    def __init__(
        self,
        encoding: str,
        status_code: int = 200,
        schema: typing.Optional[typing.Dict[str, typing.Any]] = None
    ):
        self.encoding: str = encoding
        self.status_code: int = status_code
        self.schema: typing.Optional[typing.Dict[str, typing.Any]] = schema
