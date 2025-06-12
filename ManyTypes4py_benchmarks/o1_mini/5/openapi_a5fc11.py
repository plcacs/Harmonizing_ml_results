import re
from urllib.parse import urljoin
from typing import Any, Dict, List, Optional, Union
import typesystem
from apistar.document import Document, Field, Link, Section
from apistar.schemas.jsonschema import JSON_SCHEMA

SCHEMA_REF = typesystem.Object(properties={'$ref': typesystem.String(pattern='^#/components/schemas/')})
REQUESTBODY_REF = typesystem.Object(properties={'$ref': typesystem.String(pattern='^#/components/requestBodies/')})
RESPONSE_REF = typesystem.Object(properties={'$ref': typesystem.String(pattern='^#/components/responses/')})
definitions = typesystem.SchemaDefinitions()
OPEN_API = typesystem.Object(
    title='OpenAPI',
    properties={
        'openapi': typesystem.String(),
        'info': typesystem.Reference('Info', definitions=definitions),
        'servers': typesystem.Array(items=typesystem.Reference('Server', definitions=definitions)),
        'paths': typesystem.Reference('Paths', definitions=definitions),
        'components': typesystem.Reference('Components', definitions=definitions),
        'security': typesystem.Array(items=typesystem.Reference('SecurityRequirement', definitions=definitions)),
        'tags': typesystem.Array(items=typesystem.Reference('Tag', definitions=definitions)),
        'externalDocs': typesystem.Reference('ExternalDocumentation', definitions=definitions)
    },
    pattern_properties={'^x-': typesystem.Any()},
    additional_properties=False,
    required=['openapi', 'info', 'paths']
)
definitions['Info'] = typesystem.Object(
    properties={
        'title': typesystem.String(allow_blank=True),
        'description': typesystem.Text(allow_blank=True),
        'termsOfService': typesystem.String(format='url'),
        'contact': typesystem.Reference('Contact', definitions=definitions),
        'license': typesystem.Reference('License', definitions=definitions),
        'version': typesystem.String(allow_blank=True)
    },
    pattern_properties={'^x-': typesystem.Any()},
    additional_properties=False,
    required=['title', 'version']
)
definitions['Contact'] = typesystem.Object(
    properties={
        'name': typesystem.String(allow_blank=True),
        'url': typesystem.String(format='url'),
        'email': typesystem.String(format='email')
    },
    pattern_properties={'^x-': typesystem.Any()},
    additional_properties=False
)
definitions['License'] = typesystem.Object(
    properties={
        'name': typesystem.String(),
        'url': typesystem.String(format='url')
    },
    required=['name'],
    pattern_properties={'^x-': typesystem.Any()},
    additional_properties=False
)
definitions['Server'] = typesystem.Object(
    properties={
        'url': typesystem.String(),
        'description': typesystem.Text(allow_blank=True),
        'variables': typesystem.Object(additional_properties=typesystem.Reference('ServerVariable', definitions=definitions))
    },
    pattern_properties={'^x-': typesystem.Any()},
    additional_properties=False,
    required=['url']
)
definitions['ServerVariable'] = typesystem.Object(
    properties={
        'enum': typesystem.Array(items=typesystem.String()),
        'default': typesystem.String(),
        'description': typesystem.Text(allow_blank=True)
    },
    pattern_properties={'^x-': typesystem.Any()},
    additional_properties=False,
    required=['default']
)
definitions['Paths'] = typesystem.Object(
    pattern_properties={
        '^/': typesystem.Reference('Path', definitions=definitions),
        '^x-': typesystem.Any()
    },
    additional_properties=False
)
definitions['Path'] = typesystem.Object(
    properties={
        'summary': typesystem.String(allow_blank=True),
        'description': typesystem.Text(allow_blank=True),
        'get': typesystem.Reference('Operation', definitions=definitions),
        'put': typesystem.Reference('Operation', definitions=definitions),
        'post': typesystem.Reference('Operation', definitions=definitions),
        'delete': typesystem.Reference('Operation', definitions=definitions),
        'options': typesystem.Reference('Operation', definitions=definitions),
        'head': typesystem.Reference('Operation', definitions=definitions),
        'patch': typesystem.Reference('Operation', definitions=definitions),
        'trace': typesystem.Reference('Operation', definitions=definitions),
        'servers': typesystem.Array(items=typesystem.Reference('Server', definitions=definitions)),
        'parameters': typesystem.Array(items=typesystem.Reference('Parameter', definitions=definitions))
    },
    pattern_properties={'^x-': typesystem.Any()},
    additional_properties=False
)
definitions['Operation'] = typesystem.Object(
    properties={
        'tags': typesystem.Array(items=typesystem.String()),
        'summary': typesystem.String(allow_blank=True),
        'description': typesystem.Text(allow_blank=True),
        'externalDocs': typesystem.Reference('ExternalDocumentation', definitions=definitions),
        'operationId': typesystem.String(),
        'parameters': typesystem.Array(items=typesystem.Reference('Parameter', definitions=definitions)),
        'requestBody': REQUESTBODY_REF | typesystem.Reference('RequestBody', definitions=definitions),
        'responses': typesystem.Reference('Responses', definitions=definitions),
        'deprecated': typesystem.Boolean(),
        'security': typesystem.Array(typesystem.Reference('SecurityRequirement', definitions=definitions)),
        'servers': typesystem.Array(items=typesystem.Reference('Server', definitions=definitions))
    },
    pattern_properties={'^x-': typesystem.Any()},
    additional_properties=False
)
definitions['ExternalDocumentation'] = typesystem.Object(
    properties={
        'description': typesystem.Text(allow_blank=True),
        'url': typesystem.String(format='url')
    },
    pattern_properties={'^x-': typesystem.Any()},
    additional_properties=False,
    required=['url']
)
definitions['Parameter'] = typesystem.Object(
    properties={
        'name': typesystem.String(),
        'in': typesystem.Choice(choices=['query', 'header', 'path', 'cookie']),
        'description': typesystem.Text(allow_blank=True),
        'required': typesystem.Boolean(),
        'deprecated': typesystem.Boolean(),
        'allowEmptyValue': typesystem.Boolean(),
        'style': typesystem.Choice(choices=['matrix', 'label', 'form', 'simple', 'spaceDelimited', 'pipeDelimited', 'deepObject']),
        'schema': JSON_SCHEMA | SCHEMA_REF,
        'example': typesystem.Any()
    },
    pattern_properties={'^x-': typesystem.Any()},
    additional_properties=False,
    required=['name', 'in']
)
definitions['RequestBody'] = typesystem.Object(
    properties={
        'description': typesystem.String(allow_blank=True),
        'content': typesystem.Object(additional_properties=typesystem.Reference('MediaType', definitions=definitions)),
        'required': typesystem.Boolean()
    },
    pattern_properties={'^x-': typesystem.Any()},
    additional_properties=False
)
definitions['Responses'] = typesystem.Object(
    properties={
        'default': typesystem.Reference('Response', definitions=definitions) | RESPONSE_REF
    },
    pattern_properties={
        '^([1-5][0-9][0-9]|[1-5]XX)$': typesystem.Reference('Response', definitions=definitions) | RESPONSE_REF,
        '^x-': typesystem.Any()
    },
    additional_properties=False
)
definitions['Response'] = typesystem.Object(
    properties={
        'description': typesystem.String(allow_blank=True),
        'content': typesystem.Object(additional_properties=typesystem.Reference('MediaType', definitions=definitions)),
        'headers': typesystem.Object(additional_properties=typesystem.Reference('Header', definitions=definitions))
    },
    pattern_properties={'^x-': typesystem.Any()},
    additional_properties=False
)
definitions['MediaType'] = typesystem.Object(
    properties={
        'schema': JSON_SCHEMA | SCHEMA_REF,
        'example': typesystem.Any()
    },
    pattern_properties={'^x-': typesystem.Any()},
    additional_properties=False
)
definitions['Header'] = typesystem.Object(
    properties={
        'description': typesystem.Text(),
        'required': typesystem.Boolean(),
        'deprecated': typesystem.Boolean(),
        'allowEmptyValue': typesystem.Boolean(),
        'style': typesystem.Choice(choices=['matrix', 'label', 'form', 'simple', 'spaceDelimited', 'pipeDelimited', 'deepObject']),
        'schema': JSON_SCHEMA | SCHEMA_REF,
        'example': typesystem.Any()
    },
    pattern_properties={'^x-': typesystem.Any()},
    additional_properties=False
)
definitions['Components'] = typesystem.Object(
    properties={
        'schemas': typesystem.Object(additional_properties=JSON_SCHEMA),
        'responses': typesystem.Object(additional_properties=typesystem.Reference('Response', definitions=definitions)),
        'parameters': typesystem.Object(additional_properties=typesystem.Reference('Parameter', definitions=definitions)),
        'requestBodies': typesystem.Object(additional_properties=typesystem.Reference('RequestBody', definitions=definitions)),
        'securitySchemes': typesystem.Object(additional_properties=typesystem.Reference('SecurityScheme', definitions=definitions))
    },
    pattern_properties={'^x-': typesystem.Any()},
    additional_properties=False
)
definitions['Tag'] = typesystem.Object(
    properties={
        'name': typesystem.String(),
        'description': typesystem.Text(allow_blank=True),
        'externalDocs': typesystem.Reference('ExternalDocumentation', definitions=definitions)
    },
    pattern_properties={'^x-': typesystem.Any()},
    additional_properties=False,
    required=['name']
)
definitions['SecurityRequirement'] = typesystem.Object(additional_properties=typesystem.Array(items=typesystem.String()))
definitions['SecurityScheme'] = typesystem.Object(
    properties={
        'type': typesystem.Choice(choices=['apiKey', 'http', 'oauth2', 'openIdConnect']),
        'description': typesystem.Text(allow_blank=True),
        'name': typesystem.String(),
        'in': typesystem.Choice(choices=['query', 'header', 'cookie']),
        'scheme': typesystem.String(),
        'bearerFormat': typesystem.String(),
        'flows': typesystem.Any(),
        'openIdConnectUrl': typesystem.String(format='url')
    },
    pattern_properties={'^x-': typesystem.Any()},
    additional_properties=False,
    required=['type']
)
METHODS: List[str] = ['get', 'put', 'post', 'delete', 'options', 'head', 'patch', 'trace']


def lookup(
    value: Union[Dict[str, Any], List[Any]],
    keys: List[Union[str, int]],
    default: Any = None
) -> Any:
    for key in keys:
        try:
            value = value[key]
        except (KeyError, IndexError, TypeError):
            return default
    return value


def _simple_slugify(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    text = text.lower()
    text = re.sub('[^a-z0-9]+', '_', text)
    text = re.sub('[_]+', '_', text)
    return text.strip('_')


class OpenAPI:

    def load(self, data: Dict[str, Any]) -> Document:
        title: Optional[str] = lookup(data, ['info', 'title'])
        description: Optional[str] = lookup(data, ['info', 'description'])
        version: Optional[str] = lookup(data, ['info', 'version'])
        base_url: Optional[str] = lookup(data, ['servers', 0, 'url'])
        schema_definitions: typesystem.SchemaDefinitions = self.get_schema_definitions(data)
        content: List[Union[Link, Section]] = self.get_content(data, base_url, schema_definitions)
        return Document(title=title, description=description, version=version, url=base_url, content=content)

    def get_schema_definitions(self, data: Dict[str, Any]) -> typesystem.SchemaDefinitions:
        definitions: typesystem.SchemaDefinitions = typesystem.SchemaDefinitions()
        schemas: Dict[str, Any] = lookup(data, ['components', 'schemas'], {})
        for key, value in schemas.items():
            ref: str = f'#/components/schemas/{key}'
            definitions[ref] = typesystem.from_json_schema(value, definitions=definitions)
        return definitions

    def get_content(
        self,
        data: Dict[str, Any],
        base_url: Optional[str],
        schema_definitions: typesystem.SchemaDefinitions
    ) -> List[Union[Link, Section]]:
        """
        Return all the links in the document, layed out by tag and operationId.
        """
        links_by_tag: Dict[str, List[Link]] = {}
        links: List[Link] = []
        paths: Dict[str, Any] = data.get('paths', {})
        for path, path_info in paths.items():
            operations: Dict[str, Any] = {key: path_info[key] for key in path_info if key in METHODS}
            for operation, operation_info in operations.items():
                tag: Optional[str] = lookup(operation_info, ['tags', 0])
                link: Optional[Link] = self.get_link(base_url, path, path_info, operation, operation_info, schema_definitions)
                if link is None:
                    continue
                if tag is None:
                    links.append(link)
                elif tag not in links_by_tag:
                    links_by_tag[tag] = [link]
                else:
                    links_by_tag[tag].append(link)
        sections: List[Section] = [
            Section(name=_simple_slugify(tag), title=tag.title(), content=links)
            for tag, links in links_by_tag.items()
        ]
        return links + sections

    def get_link(
        self,
        base_url: Optional[str],
        path: str,
        path_info: Dict[str, Any],
        operation: str,
        operation_info: Dict[str, Any],
        schema_definitions: typesystem.SchemaDefinitions
    ) -> Optional[Link]:
        """
        Return a single link in the document.
        """
        name: Optional[str] = operation_info.get('operationId')
        title: Optional[str] = operation_info.get('summary')
        description: Optional[str] = operation_info.get('description')
        if name is None:
            name = _simple_slugify(title)
            if not name:
                return None
        base_url = lookup(path_info, ['servers', 0, 'url'], default=base_url)
        base_url = lookup(operation_info, ['servers', 0, 'url'], default=base_url)
        parameters: List[Dict[str, Any]] = path_info.get('parameters', []) + operation_info.get('parameters', [])
        fields: List[Field] = [self.get_field(parameter, schema_definitions) for parameter in parameters]
        body_schema: Optional[Dict[str, Any]] = lookup(operation_info, ['requestBody', 'content', 'application/json', 'schema'])
        encoding: Optional[str] = None
        if body_schema:
            encoding = 'application/json'
            if '$ref' in body_schema:
                ref: str = body_schema['$ref']
                schema: Optional[typesystem.Object] = schema_definitions.get(ref)
                field_name: str = ref[len('#/components/schemas/'):].lower()
            else:
                schema = typesystem.from_json_schema(body_schema, definitions=schema_definitions)
                field_name = 'body'
            field_name = lookup(operation_info, ['requestBody', 'x-name'], default=field_name)
            fields += [Field(name=field_name, location='body', schema=schema)]
        url: str = urljoin(base_url or '', path)
        return Link(
            name=name,
            url=url,
            method=operation,
            title=title,
            description=description,
            fields=fields,
            encoding=encoding
        )

    def get_field(
        self,
        parameter: Dict[str, Any],
        schema_definitions: typesystem.SchemaDefinitions
    ) -> Field:
        """
        Return a single field in a link.
        """
        name: str = parameter.get('name')
        location: str = parameter.get('in')
        description: Optional[str] = parameter.get('description')
        required: bool = parameter.get('required', False)
        schema: Optional[Any] = parameter.get('schema')
        example: Any = parameter.get('example')
        if schema is not None:
            if '$ref' in schema:
                ref: str = schema['$ref']
                schema = schema_definitions.get(ref)
            else:
                schema = typesystem.from_json_schema(schema, definitions=schema_definitions)
        return Field(
            name=name,
            location=location,
            description=description,
            required=required,
            schema=schema,
            example=example
        )
