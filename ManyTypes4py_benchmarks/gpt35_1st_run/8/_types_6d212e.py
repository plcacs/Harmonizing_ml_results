def SettingsOption(setting: Setting, *args, **kwargs) -> typer.Option:
def SettingsArgument(setting: Setting, *args, **kwargs) -> typer.Argument:
def with_deprecated_message(warning: str) -> Callable[[Callable], Callable]:
def decorator(fn: Callable) -> Callable:
def wrapper(*args, **kwargs) -> Any:
def PrefectTyper(typer.Typer) -> None:
def __init__(self, *args, deprecated: bool = False, deprecated_start_date: Optional[str] = None, deprecated_help: str = '', deprecated_name: str = '', **kwargs) -> None:
def add_typer(self, typer_instance: typer.Typer, *args, no_args_is_help: bool = True, aliases: Optional[List[str]] = None, **kwargs) -> Any:
def command(self, name: Optional[str] = None, *args, aliases: Optional[List[str]] = None, deprecated: bool = False, deprecated_start_date: Optional[str] = None, deprecated_help: str = '', deprecated_name: str = '', **kwargs) -> Callable[[Callable], Callable]:
def wrapper(original_fn: Callable) -> Callable:
def sync_fn(*args, **kwargs) -> Any:
def wrapped_fn(*args, **kwargs) -> Any:
def setup_console(self, soft_wrap: bool, prompt: bool) -> None:
