def _extract_get_pydantic_json_schema(tp: Any) -> GetJsonSchemaFunction | None:
    """Extract `__get_pydantic_json_schema__` from a type, handling the deprecated `__modify_schema__`."""
    js_modify_function: GetJsonSchemaFunction | None = getattr(tp, '__get_pydantic_json_schema__', None)

    if hasattr(tp, '__modify_schema__'):
        BaseModel = import_cached_base_model()

        has_custom_v2_modify_js_func: bool = (
            js_modify_function is not None
            and BaseModel.__get_pydantic_json_schema__.__func__  # type: ignore
            not in (js_modify_function, getattr(js_modify_function, '__func__', None))
        )

        if not has_custom_v2_modify_js_func:
            cls_name: str | None = getattr(tp, '__name__', None)
            raise PydanticUserError(
                f'The `__modify_schema__` method is not supported in Pydantic v2. '
                f'Use `__get_pydantic_json_schema__` instead{f" in class `{cls_name}`" if cls_name else ""}.',
                code='custom-json-schema',
            )

    # handle GenericAlias' but ignore Annotated which "lies" about its origin (in this case it would be `int`)
    if hasattr(tp, '__origin__') and not _typing_extra.is_annotated(tp):
        # Generic aliases proxy attribute access to the origin, *except* dunder attributes,
        # such as `__get_pydantic_json_schema__`, hence the explicit check.
        return _extract_get_pydantic_json_schema(tp.__origin__)

    if js_modify_function is None:
        return None

    return js_modify_function
