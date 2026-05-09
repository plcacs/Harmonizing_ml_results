from collections.abc import Sequence
from typing import Any, Dict, List, Optional, Union

class ExposureParser(YamlReader):
    def __init__(self, schema_parser: SchemaParser, yaml: YamlReader) -> None:
        super().__init__(schema_parser, yaml, NodeType.Exposure.pluralize())
        self.schema_parser: SchemaParser = schema_parser
        self.yaml: YamlReader = yaml

    # ... (rest of the method)

class MetricParser(YamlReader):
    def __init__(self, schema_parser: SchemaParser, yaml: YamlReader) -> None:
        super().__init__(schema_parser, yaml, NodeType.Metric.pluralize())
        self.schema_parser: SchemaParser = schema_parser
        self.yaml: YamlReader = yaml

    # ... (rest of the method)

class GroupParser(YamlReader):
    def __init__(self, schema_parser: SchemaParser, yaml: YamlReader) -> None:
        super().__init__(schema_parser, yaml, NodeType.Group.pluralize())
        self.schema_parser: SchemaParser = schema_parser
        self.yaml: YamlReader = yaml

    # ... (rest of the method)

class SemanticModelParser(YamlReader):
    def __init__(self, schema_parser: SchemaParser, yaml: YamlReader) -> None:
        super().__init__(schema_parser, yaml, 'semantic_models')
        self.schema_parser: SchemaParser = schema_parser
        self.yaml: YamlReader = yaml

    # ... (rest of the method)

class SavedQueryParser(YamlReader):
    def __init__(self, schema_parser: SchemaParser, yaml: YamlReader) -> None:
        super().__init__(schema_parser, yaml, 'saved_queries')
        self.schema_parser: SchemaParser = schema_parser
        self.yaml: YamlReader = yaml

    # ... (rest of the method)
