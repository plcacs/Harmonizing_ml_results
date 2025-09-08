class Singleton(object):
    """Singleton Class
    This is a class to make some class being a Singleton class.
    Such as database class or config class.

    usage:
        class xxx(Singleton):
            def __init__(self):
                if hasattr(self, '_init'):
                    return
                self._init = True
                other init method
    """

    _instance: object

    def __new__(cls, *args: object, **kwargs: object) -> object:
        if not hasattr(cls, '_instance'):
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance
