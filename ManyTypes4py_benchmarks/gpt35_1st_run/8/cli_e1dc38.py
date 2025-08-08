from __future__ import annotations
from typing import Any

def cli() -> None:
    pass

def info() -> None:
    pass

def project_commands() -> None:
    pass

def global_commands() -> None:
    pass

def _init_plugins() -> None:
    pass

class KedroCLI(CommandCollection):
    def __init__(self, project_path: Path) -> None:
        pass

    def main(self, args=None, prog_name=None, complete_var=None, standalone_mode=True, **extra) -> None:
        pass

    @property
    def global_groups(self) -> Sequence:
        pass

    @property
    def project_groups(self) -> Sequence:
        pass

def main() -> None:
    pass
