from typing import Any, Callable, List, MutableSequence, Sequence, Union

class argument:
    args: List
    kwargs: dict
    argument: Any

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.argument = click.argument(*self.args, **self.kwargs)

    def __call__(self, fun: Callable) -> Any:
        return self.argument(fun)

    def __repr__(self) -> str:
        return reprcall(type(self).__name__, self.args, self.kwargs)

class option:
    args: List
    kwargs: dict
    option: Any

    def __init__(self, *args, show_default: bool = True, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.option = click.option(*args, show_default=show_default, **kwargs)

    def __call__(self, fun: Callable) -> Any:
        return self.option(fun)

    def __repr__(self) -> str:
        return reprcall(type(self).__name__, self.args, self.kwargs)

OptionDecorator = Callable[[Any], Any]
OptionSequence = Sequence[OptionDecorator]
OptionList = MutableSequence[OptionDecorator]
