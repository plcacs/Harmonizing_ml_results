import collections
import re
import typing
from typing import Callable, List, Optional, Sequence, Set, Tuple, Union

LinkInfo = collections.namedtuple("LinkInfo", ["link", "name", "sections"])


class Document:
    def __init__(
        self,
        content: Optional[Sequence[Union["Section", "Link"]]] = None,
        url: str = "",
        title: str = "",
        description: str = "",
        version: str = "",
    ) -> None:
        content = [] if (content is None) else list(content)

        # Ensure all names within a document are unique.
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

        self.content: List[Union["Section", "Link"]] = content
        self.url: str = url
        self.title: str = title
        self.description: str = description
        self.version: str = version

    def get_links(self) -> List["Link"]:
        return [item for item in self.content if isinstance(item, Link)]

    def get_sections(self) -> List["Section"]:
        return [item for item in self.content if isinstance(item, Section)]

    def walk_links(self) -> List[LinkInfo]:
        link_info_list: List[LinkInfo] = []
        for item in self.content:
            if isinstance(item, Link):
                link_info = LinkInfo(link=item, name=item.name, sections=())
                link_info_list.append(link_info)
            else:
                link_info_list.extend(item.walk_links())
        return link_info_list


class Section:
    def __init__(
        self,
        name: str,
        content: Optional[Sequence[Union["Section", "Link"]]] = None,
        title: str = "",
        description: str = "",
    ) -> None:
        content = [] if (content is None) else list(content)

        # Ensure all names within a section are unique.
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

        self.content: List[Union["Section", "Link"]] = content
        self.name: str = name
        self.title: str = title
        self.description: str = description

    def get_links(self) -> List["Link"]:
        return [item for item in self.content if isinstance(item, Link)]

    def get_sections(self) -> List["Section"]:
        return [item for item in self.content if isinstance(item, Section)]

    def walk_links(self, previous_sections: Tuple["Section", ...] = ()) -> List[LinkInfo]:
        link_info_list: List[LinkInfo] = []
        sections: Tuple["Section", ...] = previous_sections + (self,)
        for item in self.content:
            if isinstance(item, Link):
                name = ":".join([section.name for section in sections] + [item.name])
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
        handler: Optional[Callable] = None,
        name: str = "",
        encoding: str = "",
        response: Optional["Response"] = None,
        title: str = "",
        description: str = "",
        fields: Optional[Sequence["Field"]] = None,
    ) -> None:
        method = method.upper()
        fields = [] if (fields is None) else list(fields)

        url_path_names: Set[str] = set(
            [item.strip("{}").lstrip("+") for item in re.findall("{[^}]*}", url)]
        )
        path_fields: List["Field"] = [field for field in fields if field.location == "path"]
        body_fields: List["Field"] = [field for field in fields if field.location == "body"]

        assert method in (
            "GET",
            "POST",
            "PUT",
            "PATCH",
            "DELETE",
            "OPTIONS",
            "HEAD",
            "TRACE",
        )
        assert len(body_fields) < 2
        if body_fields:
            assert encoding
        for field in path_fields:
            assert field.name in url_path_names

        # Add in path fields for any "{param}" items that don't already have
        # a corresponding path field.
        for path_name in url_path_names:
            if path_name not in [field.name for field in path_fields]:
                fields += [Field(name=path_name, location="path", required=True)]

        self.url: str = url
        self.method: str = method
        self.handler: Optional[Callable] = handler
        self.name: str = name if name else handler.__name__ if handler else ""
        self.encoding: str = encoding
        self.response: Optional["Response"] = response
        self.title: str = title
        self.description: str = description
        self.fields: List["Field"] = fields

    def get_path_fields(self) -> List["Field"]:
        return [field for field in self.fields if field.location == "path"]

    def get_query_fields(self) -> List["Field"]:
        return [field for field in self.fields if field.location == "query"]

    def get_body_field(self) -> Optional["Field"]:
        for field in self.fields:
            if field.location == "body":
                return field
        return None

    def get_expanded_body(self) -> Optional[typing.Any]:
        field = self.get_body_field()
        if field is None or not hasattr(field.schema, "properties"):
            return None
        return field.schema.properties


class Field:
    def __init__(
        self,
        name: str,
        location: str,
        title: str = "",
        description: str = "",
        required: Optional[bool] = None,
        schema: Optional[typing.Any] = None,
        example: Optional[typing.Any] = None,
    ) -> None:
        assert location in ("path", "query", "body", "cookie", "header", "formData")
        if required is None:
            required = True if location in ("path", "body") else False
        if location == "path":
            assert required, "May not set 'required=False' on path fields."

        self.name: str = name
        self.title: str = title
        self.description: str = description
        self.location: str = location
        self.required: bool = required
        self.schema: Optional[typing.Any] = schema
        self.example: Optional[typing.Any] = example


class Response:
    def __init__(
        self, encoding: str, status_code: int = 200, schema: Optional[typing.Any] = None
    ) -> None:
        self.encoding: str = encoding
        self.status_code: int = status_code
        self.schema: Optional[typing.Any] = schema
