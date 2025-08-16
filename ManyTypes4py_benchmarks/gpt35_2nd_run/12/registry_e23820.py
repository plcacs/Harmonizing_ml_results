    def loads_key(self, typ: Type[ModelT], key: Any, *, serializer: Optional[CodecArg] = None) -> K:
    def _loads(self, serializer: CodecArg, data: Any) -> Any:
    def _serializer(self, typ: Type, *alt: CodecArg) -> Optional[CodecArg]:
    def loads_value(self, typ: Type[ModelT], value: Any, *, serializer: Optional[CodecArg] = None) -> V:
    def _prepare_payload(self, typ: Type, value: Any) -> Any:
    def dumps_key(self, typ: ModelArg, key: Any, *, serializer: Optional[CodecArg] = None, skip: IsInstanceArg = (bytes,)) -> Optional[bytes]:
    def dumps_value(self, typ: ModelArg, value: Any, *, serializer: Optional[CodecArg] = None, skip: IsInstanceArg = (bytes,)) -> bytes:
    @cached_property
    def Model(self) -> Type[Model]:
