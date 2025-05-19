import copy
import json
import re
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import markdown
from markdown.extensions import Extension
from markdown.preprocessors import Preprocessor
from typing_extensions import override
from zerver.lib.markdown.priorities import PREPROCESSOR_PRIORITIES
from zerver.openapi.openapi import check_deprecated_consistency, get_openapi_return_values
from .api_arguments_table_generator import generate_data_type
from re import Pattern

REGEXP: Pattern = re.compile(r'\{generate_return_values_table\|\s*(.+?)\s*\|\s*(.+)\s*\}')
EVENT_HEADER_TEMPLATE: str = '\n<div class="api-event-header">\n    <h3 id="{id}"><span class="api-event-name">{event}</span></h3>\n</div>\n'.strip()
OP_TEMPLATE: str = '<span class="api-event-op">op: {op_type}</span>'
EVENTS_TABLE_TEMPLATE: str = '\n<div class="api-events-table">\n{events_list}\n</div>\n<hr>\n'.strip()
TABLE_OPS_TEMPLATE: str = '\n<div class="api-event-type">{event_name}:</div>\n<div class="api-event-ops">\n{ops}\n</div>\n'.strip()
TABLE_LINK_TEMPLATE: str = '\n<div class="api-event-link">\n    <a href="#{url}">{link_name}</a>\n</div>\n'.strip()

@dataclass
class EventData:
    type: str
    description: str
    properties: Dict[str, Any]
    example: str
    op_type: Optional[str] = None

class MarkdownReturnValuesTableGenerator(Extension):

    @override
    def extendMarkdown(self, md: markdown.Markdown) -> None:
        md.preprocessors.register(
            APIReturnValuesTablePreprocessor(md, self.getConfigs()),
            'generate_return_values',
            PREPROCESSOR_PRIORITIES['generate_return_values']
        )

class APIReturnValuesTablePreprocessor(Preprocessor):
    config: Dict[str, Any]

    def __init__(self, md: markdown.Markdown, config: Dict[str, Any]) -> None:
        super().__init__(md)
        self.config = config

    @override
    def run(self, lines: List[str]) -> List[str]:
        done = False
        while not done:
            for line in lines:
                loc: int = lines.index(line)
                match: Optional[re.Match] = REGEXP.search(line)
                if not match:
                    continue
                doc_name: str = match.group(2)
                endpoint_method: List[str] = doc_name.rsplit(':', 1)
                if len(endpoint_method) != 2:
                    continue
                endpoint, method = endpoint_method
                return_values: Dict[str, Any] = get_openapi_return_values(endpoint, method)
                if doc_name == '/events:get':
                    return_values = copy.deepcopy(return_values)
                    events: Optional[Dict[str, Any]] = return_values['events'].pop('items', None)
                    text: List[str] = self.render_table(return_values, 0)
                    text.append('\n\n## Events by `type`\n\n')
                    if events is not None:
                        text += self.render_events(events)
                else:
                    text = self.render_table(return_values, 0)
                if len(text) > 0:
                    text = ['#### Return values', *text]
                line_split: List[str] = REGEXP.split(line, maxsplit=0)
                preceding: str = line_split[0]
                following: str = line_split[-1]
                text = [preceding, *text, following]
                lines = lines[:loc] + text + lines[loc + 1:]
                break
            else:
                done = True
        return lines

    def render_desc(
        self,
        description: str,
        spacing: int,
        data_type: str,
        return_value: Optional[str] = None
    ) -> str:
        description = description.replace('\n', '\n' + (spacing + 4) * ' ')
        if return_value is None:
            arr: List[str] = description.split(': ', 1)
            if len(arr) == 1 or '\n' in arr[0]:
                return spacing * ' ' + '* ' + description
            key_name, key_description = arr
            return (
                spacing * ' ' + '* ' + key_name + ': ' +
                '<span class="api-field-type">' + data_type + '</span>\n\n' +
                (spacing + 4) * ' ' + key_description
            )
        return (
            spacing * ' ' + '* `' + return_value + '`: ' +
            '<span class="api-field-type">' + data_type + '</span>\n\n' +
            (spacing + 4) * ' ' + description
        )

    def render_oneof_block(self, object_schema: Dict[str, Any], spacing: int) -> List[str]:
        ans: List[str] = []
        block_spacing: int = spacing
        for element in object_schema.get('oneOf', []):
            current_spacing: int = block_spacing
            if 'description' not in element:
                current_spacing -= 4
            else:
                data_type: str = generate_data_type(element)
                ans.append(self.render_desc(element['description'], current_spacing, data_type))
            if 'properties' in element:
                ans += self.render_table(element['properties'], current_spacing + 4)
            if element.get('additionalProperties', False):
                additional_properties: Any = element['additionalProperties']
                if isinstance(additional_properties, dict):
                    if 'description' in additional_properties:
                        data_type = generate_data_type(additional_properties)
                        ans.append(
                            self.render_desc(
                                additional_properties['description'],
                                current_spacing + 4,
                                data_type
                            )
                        )
                    if 'properties' in additional_properties:
                        ans += self.render_table(
                            additional_properties['properties'],
                            current_spacing + 8
                        )
        return ans

    def render_table(self, return_values: Dict[str, Any], spacing: int) -> List[str]:
        IGNORE: List[str] = ['result', 'msg', 'ignored_parameters_unsupported']
        ans: List[str] = []
        for return_value, schema in return_values.items():
            if return_value in IGNORE:
                continue
            if 'oneOf' in schema:
                data_type: str = generate_data_type(schema)
                ans.append(self.render_desc(schema.get('description', ''), spacing, data_type, return_value))
                ans += self.render_oneof_block(schema, spacing + 4)
                continue
            description: str = schema.get('description', '')
            data_type: str = generate_data_type(schema)
            check_deprecated_consistency(schema.get('deprecated', False), description)
            ans.append(self.render_desc(description, spacing, data_type, return_value))
            if 'properties' in schema:
                ans += self.render_table(schema['properties'], spacing + 4)
            if schema.get('additionalProperties', False):
                additional_properties = schema['additionalProperties']
                if isinstance(additional_properties, dict):
                    if 'description' in additional_properties:
                        data_type = generate_data_type(additional_properties)
                        ans.append(
                            self.render_desc(
                                additional_properties['description'],
                                spacing + 4,
                                data_type
                            )
                        )
                    if 'properties' in additional_properties:
                        ans += self.render_table(
                            additional_properties['properties'],
                            spacing + 8
                        )
                    elif 'oneOf' in additional_properties:
                        ans += self.render_oneof_block(
                            additional_properties,
                            spacing + 8
                        )
                    elif additional_properties.get('additionalProperties', False):
                        nested_additional = additional_properties['additionalProperties']
                        if isinstance(nested_additional, dict):
                            data_type = generate_data_type(nested_additional)
                            if 'description' in nested_additional:
                                ans.append(
                                    self.render_desc(
                                        nested_additional['description'],
                                        spacing + 8,
                                        data_type
                                    )
                                )
                            if 'properties' in nested_additional:
                                ans += self.render_table(
                                    nested_additional['properties'],
                                    spacing + 12
                                )
            if 'items' in schema:
                items = schema['items']
                if 'properties' in items:
                    ans += self.render_table(items['properties'], spacing + 4)
                elif 'oneOf' in items:
                    ans += self.render_oneof_block(items, spacing + 4)
        return ans

    def generate_event_strings(self, event_data: EventData) -> List[str]:
        event_strings: List[str] = []
        if event_data.op_type is None:
            event_strings.append(
                EVENT_HEADER_TEMPLATE.format(
                    id=event_data.type,
                    event=event_data.type
                )
            )
        else:
            op_detail: str = OP_TEMPLATE.format(op_type=event_data.op_type)
            event_strings.append(
                EVENT_HEADER_TEMPLATE.format(
                    id=f'{event_data.type}-{event_data.op_type}',
                    event=f'{event_data.type} {op_detail}'
                )
            )
        event_strings.append(f'\n{event_data.description}\n\n\n')
        event_strings += self.render_table(event_data.properties, 0)
        event_strings.append('**Example**')
        event_strings.append('\n