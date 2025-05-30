import collections
import re
import typing
from typing import Callable, List, Optional, Sequence, Set, Tuple, Union
LinkInfo = collections.namedtuple('LinkInfo', ['link', 'name', 'sections'])

class Document:

    def __init__(self, content=None, url='', title='', description='', version=''):
        content = [] if content is None else list(content)
        seen_fields: Set[str] = set()
        seen_sections: Set[str] = set()
        for item in content:
            if isinstance(item, Link):
                msg = 'Link "%s" in Document must have a unique name.'
                assert item.name not in seen_fields, msg % item.name
                seen_fields.add(item.name)
            else:
                msg = 'Section "%s" in Document must have a unique name.'
                assert item.name not in seen_sections, msg % item.name
                seen_sections.add(item.name)
        self.content: List[Union['Section', 'Link']] = content
        self.url: str = url
        self.title: str = title
        self.description: str = description
        self.version: str = version

    def get_links(self):
        return [item for item in self.content if isinstance(item, Link)]

    def get_sections(self):
        return [item for item in self.content if isinstance(item, Section)]

    def walk_links(self):
        link_info_list: List[LinkInfo] = []
        for item in self.content:
            if isinstance(item, Link):
                link_info = LinkInfo(link=item, name=item.name, sections=())
                link_info_list.append(link_info)
            else:
                link_info_list.extend(item.walk_links())
        return link_info_list

class Section:

    def __init__(self, name, content=None, title='', description=''):
        content = [] if content is None else list(content)
        seen_fields: Set[str] = set()
        seen_sections: Set[str] = set()
        for item in content:
            if isinstance(item, Link):
                msg = 'Link "%s" in Section "%s" must have a unique name.'
                assert item.name not in seen_fields, msg % (item.name, name)
                seen_fields.add(item.name)
            else:
                msg = 'Section "%s" in Section "%s" must have a unique name.'
                assert item.name not in seen_sections, msg % (item.name, name)
                seen_sections.add(item.name)
        self.content: List[Union['Section', 'Link']] = content
        self.name: str = name
        self.title: str = title
        self.description: str = description

    def get_links(self):
        return [item for item in self.content if isinstance(item, Link)]

    def get_sections(self):
        return [item for item in self.content if isinstance(item, Section)]

    def walk_links(self, previous_sections=()):
        link_info_list: List[LinkInfo] = []
        sections: Tuple['Section', ...] = previous_sections + (self,)
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

    def __init__(self, url, method, handler=None, name='', encoding='', response=None, title='', description='', fields=None):
        method = method.upper()
        fields = [] if fields is None else list(fields)
        url_path_names: Set[str] = set([item.strip('{}').lstrip('+') for item in re.findall('{[^}]*}', url)])
        path_fields: List['Field'] = [field for field in fields if field.location == 'path']
        body_fields: List['Field'] = [field for field in fields if field.location == 'body']
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
        self.handler: Optional[Callable] = handler
        self.name: str = name if name else handler.__name__ if handler else ''
        self.encoding: str = encoding
        self.response: Optional['Response'] = response
        self.title: str = title
        self.description: str = description
        self.fields: List['Field'] = fields

    def get_path_fields(self):
        return [field for field in self.fields if field.location == 'path']

    def get_query_fields(self):
        return [field for field in self.fields if field.location == 'query']

    def get_body_field(self):
        for field in self.fields:
            if field.location == 'body':
                return field
        return None

    def get_expanded_body(self):
        field = self.get_body_field()
        if field is None or not hasattr(field.schema, 'properties'):
            return None
        return field.schema.properties

class Field:

    def __init__(self, name, location, title='', description='', required=None, schema=None, example=None):
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
        self.schema: Optional[typing.Any] = schema
        self.example: Optional[typing.Any] = example

class Response:

    def __init__(self, encoding, status_code=200, schema=None):
        self.encoding: str = encoding
        self.status_code: int = status_code
        self.schema: Optional[typing.Any] = schema