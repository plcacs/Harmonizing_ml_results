def validate_with_model(data: dict, model: BaseModel) -> None:
    allowed_fields = set(model.__fields__.keys())
    if not set(data.keys()).issubset(allowed_fields):
        raise ValueError(f'Extra fields not allowed: {set(data.keys()) - allowed_fields}')
    model.validate(data, strict=True)

def make_checker(base_model: BaseModel) -> Callable:
    def f(label: str, event: dict) -> None:
        try:
            validate_with_model(event, base_model)
        except Exception as e:
            print(f'\nFAILURE:\n\nThe event below fails the check to make sure it has the\ncorrect "shape" of data:\n\n    {label}\n\nOften this is a symptom that the following type definition\nis either broken or needs to be updated due to other\nchanges that you have made:\n\n    {base_model}\n\nA traceback should follow to help you debug this problem.\n\nHere is the event:\n')
            PrettyPrinter(indent=4).pprint(event)
            raise e
    return f

def check_alert_words(var_name: str, event: dict) -> None:
    _check_alert_words(var_name, event)

def check_attachment_add(var_name: str, event: dict) -> None:
    _check_attachment_add(var_name, event)

# Add type annotations for the remaining check functions
