from dataclasses import dataclass
from typing import Iterable, Iterator, List, Tuple, Union
from dbt.context.context_config import ContextConfig
from dbt.contracts.files import FilePath
from dbt.contracts.graph.nodes import HookNode
from dbt.node_types import NodeType, RunHookType
from dbt.parser.base import SimpleParser
from dbt.parser.search import FileBlock
from dbt.utils import get_pseudo_hook_path
from dbt_common.exceptions import DbtInternalError

@dataclass
class HookBlock(FileBlock):
    """Represents a hook block in a dbt project."""

    @property
    def contents(self) -> str:
        """Returns the contents of the hook block."""
        return self.value

    @property
    def name(self) -> str:
        """Returns the name of the hook block."""
        return '{}-{!s}-{!s}'.format(self.project, self.hook_type, self.index)

class HookSearcher(Iterable[HookBlock]):
    """Searches for hook blocks in a dbt project."""

    def __init__(self, project: ContextConfig, source_file: str, hook_type: RunHookType):
        """Initializes the hook searcher.

        Args:
            project: The dbt project configuration.
            source_file: The source file to search for hooks.
            hook_type: The type of hook to search for.
        """
        self.project = project
        self.source_file = source_file
        self.hook_type = hook_type

    def _hook_list(self, hooks: Union[List[str], Tuple[str], str]) -> List[str]:
        """Converts hooks to a list if necessary."""
        if isinstance(hooks, tuple):
            hooks = list(hooks)
        elif not isinstance(hooks, list):
            hooks = [hooks]
        return hooks

    def get_hook_defs(self) -> List[str]:
        """Returns the hook definitions for the given hook type."""
        if self.hook_type == RunHookType.Start:
            hooks = self.project.on_run_start
        elif self.hook_type == RunHookType.End:
            hooks = self.project.on_run_end
        else:
            raise DbtInternalError('hook_type must be one of "{}" or "{}" (got {})'.format(RunHookType.Start, RunHookType.End, self.hook_type))
        return self._hook_list(hooks)

    def __iter__(self) -> Iterator[HookBlock]:
        """Iterates over the hook blocks."""
        hooks = self.get_hook_defs()
        for index, hook in enumerate(hooks):
            yield HookBlock(file=self.source_file, project=self.project.project_name, value=hook, index=index, hook_type=self.hook_type)

class HookParser(SimpleParser[HookBlock, HookNode]):
    """Parses hook blocks into hook nodes."""

    def get_path(self) -> FilePath:
        """Returns the path to the hook file."""
        path = FilePath(project_root=self.project.project_root, searched_path='.', relative_path='dbt_project.yml', modification_time=0.0)
        return path

    def parse_from_dict(self, dct: dict, validate: bool = True) -> HookNode:
        """Parses a hook node from a dictionary."""
        if validate:
            HookNode.validate(dct)
        return HookNode.from_dict(dct)

    @classmethod
    def get_compiled_path(cls, block: HookBlock) -> str:
        """Returns the compiled path for the hook block."""
        return get_pseudo_hook_path(block.name)

    def _create_parsetime_node(self, block: HookBlock, path: FilePath, config: ContextConfig, fqn: str, name: str = None, **kwargs) -> HookNode:
        """Creates a parse-time hook node."""
        return super()._create_parsetime_node(block=block, path=path, config=config, fqn=fqn, index=block.index, name=name, tags=[str(block.hook_type)])

    @property
    def resource_type(self) -> NodeType:
        """Returns the resource type for the hook parser."""
        return NodeType.Operation

    def parse_file(self, block: HookBlock) -> None:
        """Parses a hook file."""
        for hook_type in RunHookType:
            for hook in HookSearcher(self.project, block.file, hook_type):
                self.parse_node(hook)
