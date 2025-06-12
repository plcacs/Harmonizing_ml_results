from dataclasses import dataclass
from typing import Iterable, Iterator, List, Tuple, Union, Any, Dict, Optional
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
    index: int
    hook_type: RunHookType

    @property
    def contents(self) -> str:
        return self.value

    @property
    def name(self) -> str:
        return '{}-{!s}-{!s}'.format(self.project, self.hook_type, self.index)

class HookSearcher(Iterable[HookBlock]):
    def __init__(self, project: Any, source_file: FilePath, hook_type: RunHookType) -> None:
        self.project = project
        self.source_file = source_file
        self.hook_type = hook_type

    def _hook_list(self, hooks: Union[Tuple[Any, ...], List[Any], Any]) -> List[Any]:
        if isinstance(hooks, tuple):
            hooks = list(hooks)
        elif not isinstance(hooks, list):
            hooks = [hooks]
        return hooks

    def get_hook_defs(self) -> List[Any]:
        if self.hook_type == RunHookType.Start:
            hooks = self.project.on_run_start
        elif self.hook_type == RunHookType.End:
            hooks = self.project.on_run_end
        else:
            raise DbtInternalError('hook_type must be one of "{}" or "{}" (got {})'.format(RunHookType.Start, RunHookType.End, self.hook_type))
        return self._hook_list(hooks)

    def __iter__(self) -> Iterator[HookBlock]:
        hooks = self.get_hook_defs()
        for index, hook in enumerate(hooks):
            yield HookBlock(file=self.source_file, project=self.project.project_name, value=hook, index=index, hook_type=self.hook_type)

class HookParser(SimpleParser[HookBlock, HookNode]):
    def get_path(self) -> FilePath:
        path = FilePath(project_root=self.project.project_root, searched_path='.', relative_path='dbt_project.yml', modification_time=0.0)
        return path

    def parse_from_dict(self, dct: Dict[str, Any], validate: bool = True) -> HookNode:
        if validate:
            HookNode.validate(dct)
        return HookNode.from_dict(dct)

    @classmethod
    def get_compiled_path(cls, block: HookBlock) -> str:
        return get_pseudo_hook_path(block.name)

    def _create_parsetime_node(
        self, 
        block: HookBlock, 
        path: FilePath, 
        config: ContextConfig, 
        fqn: List[str], 
        name: Optional[str] = None, 
        **kwargs: Any
    ) -> HookNode:
        return super()._create_parsetime_node(block=block, path=path, config=config, fqn=fqn, index=block.index, name=name, tags=[str(block.hook_type)])

    @property
    def resource_type(self) -> NodeType:
        return NodeType.Operation

    def parse_file(self, block: FileBlock) -> None:
        for hook_type in RunHookType:
            for hook in HookSearcher(self.project, block.file, hook_type):
                self.parse_node(hook)
