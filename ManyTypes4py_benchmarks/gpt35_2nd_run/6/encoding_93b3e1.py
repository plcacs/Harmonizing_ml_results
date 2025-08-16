TypeDict = Dict[str, Any]

def typed_dict_to_dict(typ: Type) -> TypeDict:
    ...

def type_to_dict(typ: Type) -> TypeDict:
    ...

def typed_dict_from_dict(d: TypeDict) -> TypedDict:
    ...

def type_from_dict(d: TypeDict) -> Type:
    ...

def type_to_json(typ: Type) -> str:
    ...

def type_from_json(typ_json: str) -> Type:
    ...

def arg_types_to_json(arg_types: Dict[str, Type]) -> str:
    ...

def arg_types_from_json(arg_types_json: str) -> Dict[str, Type]:
    ...

TypeEncoder = Callable[[Type], str]

def maybe_encode_type(encode: TypeEncoder, typ: Optional[Type]) -> Optional[str]:
    ...

TypeDecoder = Callable[[str], Type]

def maybe_decode_type(decode: TypeDecoder, encoded: Optional[str]) -> Optional[Type]:

CallTraceRowT = TypeVar('CallTraceRowT', bound='CallTraceRow')

class CallTraceRow(CallTraceThunk):
    def __init__(self, module: str, qualname: str, arg_types: str, return_type: str, yield_type: str):
        ...

    @classmethod
    def from_trace(cls, trace: CallTrace) -> 'CallTraceRow':
        ...

    def to_trace(self) -> CallTrace:
        ...

    def __eq__(self, other: Any) -> bool:
        ...

def serialize_traces(traces: Iterable[CallTrace]) -> Iterable[CallTraceRow]:
    ...
