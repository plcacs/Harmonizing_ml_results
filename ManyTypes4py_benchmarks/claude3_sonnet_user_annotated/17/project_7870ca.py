import os
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, Dict, List, Mapping, Optional, TypeVar, Union, Set

from typing_extensions import Protocol, runtime_checkable

from dbt import deprecations
from dbt.adapters.contracts.connection import QueryComment
from dbt.clients.yaml_helper import load_yaml_text
from dbt.config.selectors import SelectorDict
from dbt.config.utils import normalize_warn_error_options
from dbt.constants import (
    DBT_PROJECT_FILE_NAME,
    DEPENDENCIES_FILE_NAME,
    PACKAGE_LOCK_HASH_KEY,
    PACKAGES_FILE_NAME,
)
from dbt.contracts.project import PackageConfig
from dbt.contracts.project import Project as ProjectContract
from dbt.contracts.project import ProjectFlags, ProjectPackageMetadata, SemverString
from dbt.exceptions import (
    DbtExclusivePropertyUseError,
    DbtProjectError,
    DbtRuntimeError,
    ProjectContractBrokenError,
    ProjectContractError,
)
from dbt.flags import get_flags
from dbt.graph import SelectionSpec
from dbt.node_types import NodeType
from dbt.utils import MultiDict, coerce_dict_str, md5
from dbt.version import get_installed_version
from dbt_common.clients.system import load_file_contents, path_exists
from dbt_common.dataclass_schema import ValidationError
from dbt_common.exceptions import SemverError
from dbt_common.helper_types import NoValue
from dbt_common.semver import VersionSpecifier, versions_compatible

from .renderer import DbtProjectYamlRenderer, PackageRenderer
from .selectors import (
    SelectorConfig,
    selector_config_from_data,
    selector_data_from_root,
)
