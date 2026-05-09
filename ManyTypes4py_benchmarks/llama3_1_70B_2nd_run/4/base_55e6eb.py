class argument:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.argument = click.argument(*self.args, **self.kwargs)

    def __call__(self, fun: Callable[..., Any]) -> Callable[..., Any]:
        return self.argument(fun)

    def __repr__(self) -> str:
        return reprcall(type(self).__name__, self.args, self.kwargs)


class option:
    def __init__(self, *args: Any, show_default: bool = True, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.option = click.option(*args, show_default=show_default, **kwargs)

    def __call__(self, fun: Callable[..., Any]) -> Callable[..., Any]:
        return self.option(fun)

    def __repr__(self) -> str:
        return reprcall(type(self).__name__, self.args, self.kwargs)


class State:
    app: Any = None
    quiet: bool = False
    debug: bool = False
    workdir: str = None
    datadir: str = None
    json: bool = False
    no_color: bool = False
    loop: str = None
    logfile: str = None
    loglevel: str = None
    blocking_timeout: float = None
    console_port: int = None


class _FaustRootContextT(click.Context):
    pass


def find_app(app: str, *, symbol_by_name: Callable[[str], Any] = symbol_by_name, imp: Callable[[str], ModuleType] = import_from_cwd, attr_name: str = 'app') -> Any:
    try:
        val = symbol_by_name(app, imp=imp)
    except AttributeError:
        val = imp(app)
    if isinstance(val, ModuleType) and ':' not in app:
        found = getattr(val, attr_name)
        if isinstance(found, ModuleType):
            raise AttributeError(f'Looks like module, not app: -A {app}')
        val = found
    return prepare_app(val, app)


def prepare_app(app: Any, name: str) -> Any:
    app.finalize()
    if app.conf._origin is None:
        app.conf._origin = name
    app.worker_init()
    if app.conf.autodiscover:
        app.discover()
    app.worker_init_post_autodiscover()
    if 1:
        main = sys.modules.get('__main__')
        if main is not None and 'cProfile.py' in getattr(main, '__file__', ''):
            from ..models import registry
            registry.update({(app.conf.origin or '') + k[8:]: v for k, v in registry.items() if k.startswith('cProfile.')})
        return app


def _apply_options(options: List[OptionDecorator]) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def _inner(fun: Callable[..., Any]) -> Callable[..., Any]:
        for opt in options:
            fun = opt(fun)
        return fun
    return _inner


class _Group(click.Group):
    def get_help(self, ctx: click.Context) -> str:
        self._maybe_import_app()
        return super().get_help(ctx)

    def get_usage(self, ctx: click.Context) -> str:
        self._maybe_import_app()
        return super().get_usage(ctx)

    def _maybe_import_app(self, argv: List[str] = sys.argv) -> None:
        workdir = self._extract_param(argv, '-W', '--workdir')
        if workdir:
            os.chdir(Path(workdir).absolute())
        appstr = self._extract_param(argv, '-A', '--app')
        if appstr is not None:
            find_app(appstr)

    def _extract_param(self, argv: List[str], shortopt: str, longopt: str) -> Optional[str]:
        for i, arg in enumerate(argv):
            if arg == shortopt:
                try:
                    return argv[i + 1]
                except IndexError:
                    raise click.UsageError(f'Missing argument for {shortopt}')
            elif arg.startswith(longopt):
                if '=' in arg:
                    _, _, value = arg.partition('=')
                    return value
                else:
                    try:
                        return argv[i + 1]
                    except IndexError:
                        raise click.UsageError(f'Missing argument for {longopt}')
        return None

    def make_context(self, info_name: str, args: List[str], app: Any = None, parent: Optional[click.Context] = None, stdout: Optional[IO] = None, stderr: Optional[IO] = None, side_effects: bool = True, **extra: Any) -> click.Context:
        ctx = super().make_context(info_name, args, **extra)
        self._maybe_import_app()
        root = cast(_FaustRootContextT, ctx.find_root())
        root.app = app
        root.stdout = stdout
        root.stderr = stderr
        root.side_effects = side_effects
        return ctx


@click.group(cls=_Group)
@_apply_options(builtin_options)
@click.pass_context
def cli(ctx: click.Context, *args: Any, **kwargs: Any) -> None:
    return _prepare_cli(ctx, *args, **kwargs)


def _prepare_cli(ctx: click.Context, app: Any, quiet: bool, debug: bool, workdir: str, datadir: str, json: bool, no_color: bool, loop: str) -> None:
    state = ctx.ensure_object(State)
    state.app = app
    state.quiet = quiet
    state.debug = debug
    state.workdir = workdir
    state.datadir = datadir
    state.json = json
    state.no_color = no_color
    state.loop = loop
    root = cast(_FaustRootContextT, ctx.find_root())
    if root.side_effects:
        if workdir:
            os.environ['F_WORKDIR'] = workdir
            os.chdir(Path(workdir).absolute())
        if datadir:
            os.environ['F_DATADIR'] = datadir
        if not no_color and terminal.isatty(sys.stdout):
            enable_all_colors()
        else:
            disable_all_colors()
        if json:
            disable_all_colors()


class Command(abc.ABC):
    UsageError = click.UsageError
    abstract: bool = True
    _click: Optional[click.Command] = None
    daemon: bool = False
    redirect_stdouts: Optional[bool] = None
    redirect_stdouts_level: Optional[str] = None
    builtin_options: List[OptionDecorator] = builtin_options
    options: Optional[List[OptionDecorator]] = None
    prog_name: str = ''

    @classmethod
    def as_click_command(cls) -> click.Command:
        @click.pass_context
        @wraps(cls)
        def _inner(ctx: click.Context, *args: Any, **kwargs: Any) -> None:
            cmd = cls(ctx, *args, **kwargs)
            with exiting(print_exception=True, file=sys.stderr):
                cmd()
        return _apply_options(cls.options or [])(cli.command(help=cls.__doc__)(_inner))

    def __init_subclass__(cls) -> None:
        if cls.abstract:
            cls.abstract = False
        else:
            cls._click = cls.as_click_command()
        _apply_options(cls.builtin_options)(cls._parse)
        _apply_options(cls.options or [])(cls._parse)

    @classmethod
    def parse(cls, argv: List[str]) -> Dict[str, Any]:
        return cls._parse(argv, standalone_mode=False)

    @staticmethod
    @click.command()
    def _parse(**kwargs: Any) -> Dict[str, Any]:
        return kwargs

    def __init__(self, ctx: click.Context, *args: Any, **kwargs: Any) -> None:
        self.ctx = ctx
        root = cast(_FaustRootContextT, self.ctx.find_root())
        self.state = ctx.ensure_object(State)
        self.debug = self.state.debug
        self.quiet = self.state.quiet
        self.workdir = self.state.workdir
        self.datadir = self.state.datadir
        self.json = self.state.json
        self.no_color = self.state.no_color
        self.logfile = self.state.logfile
        self.stdout = root.stdout or sys.stdout
        self.stderr = root.stderr or sys.stderr
        self.args = args
        self.kwargs = kwargs
        self.prog_name = root.command_path
        self._loglevel = self.state.loglevel
        self._blocking_timeout = self.state.blocking_timeout
        self._console_port = self.state.console_port

    async def run(self, *args: Any, **kwargs: Any) -> None:
        ...

    async def execute(self, *args: Any, **kwargs: Any) -> None:
        try:
            await self.run(*args, **kwargs)
        finally:
            await self.on_stop()

    async def on_stop(self) -> None:
        ...

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        self.run_using_worker(*args, **kwargs)

    def run_using_worker(self, *args: Any, **kwargs: Any) -> None:
        loop = asyncio.get_event_loop()
        args = self.args + args
        kwargs = {**self.kwargs, **kwargs}
        service = self.as_service(loop, *args, **kwargs)
        worker = self.worker_for_service(service, loop)
        self.on_worker_created(worker)
        raise worker.execute_from_commandline()

    def on_worker_created(self, worker: Worker) -> None:
        ...

    def as_service(self, loop: asyncio.AbstractEventLoop, *args: Any, **kwargs: Any) -> Service:
        return Service.from_awaitable(self.execute(*args, **kwargs), name=type(self).__name__, loop=loop or asyncio.get_event_loop())

    def worker_for_service(self, service: Service, loop: Optional[asyncio.AbstractEventLoop] = None) -> Worker:
        return self._Worker(service, debug=self.debug, quiet=self.quiet, stdout=self.stdout, stderr=self.stderr, loglevel=self.loglevel, logfile=self.logfile, blocking_timeout=self.blocking_timeout, console_port=self.console_port, redirect_stdouts=self.redirect_stdouts or False, redirect_stdouts_level=self.redirect_stdouts_level, loop=loop or asyncio.get_event_loop(), daemon=self.daemon)

    @property
    def _Worker(self) -> Type[Worker]:
        return Worker

    def tabulate(self, data: List[Any], headers: Optional[List[Any]] = None, wrap_last_row: bool = True, title: str = '', title_color: str = 'blue', **kwargs: Any) -> str:
        if self.json:
            return self._tabulate_json(data, headers=headers)
        if headers:
            data = [headers] + list(data)
        title = self.bold(self.color(title_color, title))
        table = self.table(data, title=title, **kwargs)
        if wrap_last_row:
            data = [list(item[:-1]) + [self._table_wrap(table, item[-1])] for item in data]
        return table.table

    def _tabulate_json(self, data: List[Any], headers: Optional[List[Any]] = None) -> str:
        if headers:
            return json.dumps([dict(zip(headers, row)) for row in data])
        return json.dumps(data)

    def table(self, data: List[Any], title: str = '', **kwargs: Any) -> terminaltables.SingleTable:
        return terminal.table(data, title=title, target=sys.stdout, **kwargs)

    def color(self, name: str, text: str) -> str:
        return Color(f'{{{name}}}{text}{{/{name}}}')

    def dark(self, text: str) -> str:
        return self.color('autoblack', text)

    def bold(self, text: str) -> str:
        return self.color('b', text)

    def bold_tail(self, text: str, *, sep: str = '.') -> str:
        head, fsep, tail = text.rpartition(sep)
        return fsep.join([head, self.bold(tail)])

    def _table_wrap(self, table: terminaltables.SingleTable, text: str) -> str:
        max_width = max(table.column_max_width(1), 10)
        return '\n'.join(wrap(text, max_width))

    def say(self, message: str, file: Optional[IO] = None, err: Optional[bool] = None, **kwargs: Any) -> None:
        if not self.quiet:
            echo(message, file=file or self.stdout, err=cast(bool, err or self.stderr), **kwargs)

    def carp(self, s: str, **kwargs: Any) -> None:
        if self.debug:
            self.say(f'#-- {s}', **kwargs)

    def dumps(self, obj: Any) -> str:
        return json.dumps(obj)

    @property
    def loglevel(self) -> str:
        return self._loglevel or DEFAULT_LOGLEVEL

    @loglevel.setter
    def loglevel(self, level: str) -> None:
        self._loglevel = level

    @property
    def blocking_timeout(self) -> float:
        return self._blocking_timeout or 0.0

    @blocking_timeout.setter
    def blocking_timeout(self, timeout: float) -> None:
        self._blocking_timeout = timeout

    @property
    def console_port(self) -> int:
        return self._console_port or CONSOLE_PORT

    @console_port.setter
    def console_port(self, port: int) -> None:
        self._console_port = port


class AppCommand(Command):
    abstract: bool = True
    require_app: bool = True

    @classmethod
    def from_handler(cls, *options: OptionDecorator, **kwargs: Any) -> Callable[[Callable[..., Awaitable[Any]]], Type['AppCommand']]:
        def _inner(fun: Callable[..., Awaitable[Any]]) -> Type['AppCommand']:
            target = fun
            if not inspect.signature(fun).parameters:
                target = staticmethod(fun)
            fields = {'run': target, '__doc__': fun.__doc__, '__name__': fun.__name__, '__qualname__': fun.__qualname__, '__module__': fun.__module__, '__wrapped__': fun, 'options': options}
            return type(fun.__name__, (cls,), {**fields, **kwargs})
        return _inner

    def __init__(self, ctx: click.Context, *args: Any, key_serializer: Optional[CodecArg] = None, value_serializer: Optional[CodecArg] = None, **kwargs: Any) -> None:
        super().__init__(ctx)
        self.app = self._finalize_app(getattr(ctx.find_root(), 'app', None))
        self.args = args
        self.kwargs = kwargs
        self.key_serializer = key_serializer or self.app.conf.key_serializer
        self.value_serializer = value_serializer or self.app.conf.value_serializer

    def _finalize_app(self, app: Any) -> Any:
        if app is not None:
            return self._finalize_concrete_app(app)
        else:
            return self._app_from_str(self.state.app)

    def _app_from_str(self, appstr: Optional[str] = None) -> Any:
        if appstr:
            return find_app(appstr)
        else:
            if self.require_app:
                raise self.UsageError('Need to specify app using -A parameter')
            return None

    def _finalize_concrete_app(self, app: Any) -> Any:
        app.finalize()
        origin = app.conf.origin
        if sys.argv:
            origin = self._detect_main_package(sys.argv)
        return prepare_app(app, origin)

    def _detect_main_package(self, argv: List[str]) -> str:
        prog = Path(argv[0]).absolute()
        paths = []
        p = prog.parent
        while p:
            if not (p / '__init__.py').is_file():
                break
            paths.append(p)
            p = p.parent
        package = '.'.join([p.name for p in paths] + [prog.with_suffix('').name])
        if package.endswith('.__main__'):
            package = package[:-9]
        return package

    async def on_stop(self) -> None:
        await super().on_stop()
        app = cast(_App, self.app)
        if app._producer is not None and app._producer.started:
            await app._producer.stop()
        if app.started:
            await app.stop()
        if app._http_client is not None:
            await app._maybe_close_http_client()

    def to_key(self, typ: str, key: str) -> Any:
        return self.to_model(typ, key, self.key_serializer)

    def to_value(self, typ: str, value: str) -> Any:
        return self.to_model(typ, value, self.value_serializer)

    def to_model(self, typ: str, value: str, serializer: CodecArg) -> Any:
        if typ:
            model = self.import_relative_to_app(typ)
            return model.loads(want_bytes(value), serializer=serializer)
        return want_bytes(value)

    def import_relative_to_app(self, attr: str) -> Type[ModelT]:
        try:
            return symbol_by_name(attr)
        except ImportError as original_exc:
            if not self.app.conf.origin:
                raise
            root, _, _ = self.app.conf.origin.partition(':')
            try:
                return symbol_by_name(f'{root}.models.{attr}')
            except ImportError:
                try:
                    return symbol_by_name(f'{root}.{attr}')
                except ImportError:
                    raise original_exc from original_exc

    def to_topic(self, entity: str) -> Any:
        if not entity:
            raise self.UsageError('Missing topic/@agent name')
        if entity.startswith('@'):
            return self.import_relative_to_app(entity[1:])
        return self.app.topic(entity)

    def abbreviate_fqdn(self, name: str, *, prefix: str = '') -> str:
        if self.app.conf.origin:
            return text.abbr_fqdn(self.app.conf.origin, name, prefix=prefix)
        return ''

    @property
    def blocking_timeout(self) -> float:
        return self._blocking_timeout or self.app.conf.blocking_timeout

    @blocking_timeout.setter
    def blocking_timeout(self, timeout: float) -> None:
        self._blocking_timeout = timeout


def call_command(command: str, args: Optional[List[str]] = None, stdout: Optional[IO] = None, stderr: Optional[IO] = None, side_effects: bool = False, **kwargs: Any) -> Tuple[int, IO, IO]:
    exitcode = 0
    if stdout is None:
        stdout = io.StringIO()
    if stderr is None:
        stderr = io.StringIO()
    try:
        cli(args=[command] + (args or []), side_effects=side_effects, stdout=stdout, stderr=stderr, **kwargs)
    except SystemExit as exc:
        exitcode = exc.code
    return (exitcode, stdout, stderr)
