from typing import Any, Dict, Generic, List, Optional, Tuple
from dbt.artifacts.resources import NodeVersion
from dbt.clients.jinja import GENERIC_TEST_KWARGS_NAME, get_rendered
from dbt.contracts.graph.nodes import UnpatchedSourceDefinition
from dbt.contracts.graph.unparsed import UnparsedModelUpdate, UnparsedNodeUpdate
from dbt.exceptions import CustomMacroPopulatingConfigValueError, SameKeyNestedError, TagNotStringError, TagsNotListOfStringsError, TestArgIncludesModelError, TestArgsNotDictError, TestDefinitionDictLengthError, TestTypeError, UnexpectedTestNamePatternError
from dbt.parser.common import Testable
from dbt.utils import md5
from dbt_common.exceptions.macros import UndefinedMacroError

def synthesize_generic_test_names(test_type: str, test_name: str, args: Dict[str, Any]) -> Tuple[str, str]:
    # ...

class TestBuilder(Generic[Testable]):
    # ...

    def __init__(self, 
                 data_test: Any, 
                 target: Any, 
                 package_name: str, 
                 render_ctx: Any, 
                 column_name: Optional[str] = None, 
                 version: Optional[NodeVersion] = None) -> None:
        # ...

    def _process_legacy_args(self) -> Dict[str, Any]:
        # ...

    def _render_values(self, config: Dict[str, Any]) -> Dict[str, Any]:
        # ...

    def _bad_type(self) -> TypeError:
        # ...

    @staticmethod
    def extract_test_args(data_test: Dict[str, Any], name: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        # ...

    def tags(self) -> List[str]:
        # ...

    def macro_name(self) -> str:
        # ...

    def get_synthetic_test_names(self) -> Tuple[str, str]:
        # ...

    def construct_config(self) -> str:
        # ...

    def build_raw_code(self) -> str:
        # ...

    def build_model_str(self) -> str:
        # ...
