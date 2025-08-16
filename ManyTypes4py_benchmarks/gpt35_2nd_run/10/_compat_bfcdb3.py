def _regenerate_error_with_loc(*, errors: List[Dict[str, Any]], loc_prefix: Tuple[str]) -> List[Dict[str, Any]]:
    updated_loc_errors = [{**err, 'loc': loc_prefix + err.get('loc', ())} for err in _normalize_errors(errors)]
    return updated_loc_errors

def _annotation_is_sequence(annotation: Type) -> bool:
    if lenient_issubclass(annotation, (str, bytes)):
        return False
    return lenient_issubclass(annotation, sequence_types)

def field_annotation_is_sequence(annotation: Type) -> bool:
    origin = get_origin(annotation)
    if origin is Union or origin is UnionType:
        for arg in get_args(annotation):
            if field_annotation_is_sequence(arg):
                return True
        return False
    return _annotation_is_sequence(annotation) or _annotation_is_sequence(get_origin(annotation))

def value_is_sequence(value: Any) -> bool:
    return isinstance(value, sequence_types) and (not isinstance(value, (str, bytes)))

def _annotation_is_complex(annotation: Type) -> bool:
    return lenient_issubclass(annotation, (BaseModel, Mapping, UploadFile)) or _annotation_is_sequence(annotation) or is_dataclass(annotation)

def field_annotation_is_complex(annotation: Type) -> bool:
    origin = get_origin(annotation)
    if origin is Union or origin is UnionType:
        return any((field_annotation_is_complex(arg) for arg in get_args(annotation)))
    return _annotation_is_complex(annotation) or _annotation_is_complex(origin) or hasattr(origin, '__pydantic_core_schema__') or hasattr(origin, '__get_pydantic_core_schema__')

def field_annotation_is_scalar(annotation: Type) -> bool:
    return annotation is Ellipsis or not field_annotation_is_complex(annotation)

def field_annotation_is_scalar_sequence(annotation: Type) -> bool:
    origin = get_origin(annotation)
    if origin is Union or origin is UnionType:
        at_least_one_scalar_sequence = False
        for arg in get_args(annotation):
            if field_annotation_is_scalar_sequence(arg):
                at_least_one_scalar_sequence = True
                continue
            elif not field_annotation_is_scalar(arg):
                return False
        return at_least_one_scalar_sequence
    return field_annotation_is_sequence(annotation) and all((field_annotation_is_scalar(sub_annotation) for sub_annotation in get_args(annotation)))

def is_bytes_or_nonable_bytes_annotation(annotation: Type) -> bool:
    if lenient_issubclass(annotation, bytes):
        return True
    origin = get_origin(annotation)
    if origin is Union or origin is UnionType:
        for arg in get_args(annotation):
            if lenient_issubclass(arg, bytes):
                return True
    return False

def is_uploadfile_or_nonable_uploadfile_annotation(annotation: Type) -> bool:
    if lenient_issubclass(annotation, UploadFile):
        return True
    origin = get_origin(annotation)
    if origin is Union or origin is UnionType:
        for arg in get_args(annotation):
            if lenient_issubclass(arg, UploadFile):
                return True
    return False

def is_bytes_sequence_annotation(annotation: Type) -> bool:
    origin = get_origin(annotation)
    if origin is Union or origin is UnionType:
        at_least_one = False
        for arg in get_args(annotation):
            if is_bytes_sequence_annotation(arg):
                at_least_one = True
                continue
        return at_least_one
    return field_annotation_is_sequence(annotation) and all((is_bytes_or_nonable_bytes_annotation(sub_annotation) for sub_annotation in get_args(annotation)))

def is_uploadfile_sequence_annotation(annotation: Type) -> bool:
    origin = get_origin(annotation)
    if origin is Union or origin is UnionType:
        at_least_one = False
        for arg in get_args(annotation):
            if is_uploadfile_sequence_annotation(arg):
                at_least_one = True
                continue
        return at_least_one
    return field_annotation_is_sequence(annotation) and all((is_uploadfile_or_nonable_uploadfile_annotation(sub_annotation) for sub_annotation in get_args(annotation)))
