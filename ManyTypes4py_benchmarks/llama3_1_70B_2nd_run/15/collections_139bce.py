class AutoEnum(str, Enum):
    """
    An enum class that automatically generates value from variable names.

    This guards against common errors where variable names are updated but values are
    not.

    In addition, because AutoEnums inherit from `str`, they are automatically
    JSON-serializable.

    See https://docs.python.org/3/library/enum.html#using-automatic-values

    Example:
        