from typing import List, Dict, Any

class AsyncTree:
    cache: Dict[int, bool]
    use_task_groups: bool

    async def mock_io_call(self) -> None: ...

    async def workload_func(self) -> Any: ...

    async def recurse_with_gather(self, recurse_level: int) -> None: ...

    async def recurse_with_task_group(self, recurse_level: int) -> None: ...

    async def run(self) -> None: ...

class EagerMixin:
    async def run(self) -> None: ...

class NoneAsyncTree(AsyncTree):
    async def workload_func(self) -> None: ...

class EagerAsyncTree(EagerMixin, NoneAsyncTree):
    pass

class IOAsyncTree(AsyncTree):
    async def workload_func(self) -> None: ...

class EagerIOAsyncTree(EagerMixin, IOAsyncTree):
    pass

class MemoizationAsyncTree(AsyncTree):
    async def workload_func(self) -> Any: ...

class EagerMemoizationAsyncTree(EagerMixin, MemoizationAsyncTree):
    pass

class CpuIoMixedAsyncTree(MemoizationAsyncTree):
    async def workload_func(self) -> Any: ...

class EagerCpuIoMixedAsyncTree(EagerMixin, CpuIoMixedAsyncTree):
    pass

def add_metadata(runner: Any) -> None: ...

def add_cmdline_args(cmd: List[str], args: Any) -> None: ...

def add_parser_args(parser: Any) -> None: ...
