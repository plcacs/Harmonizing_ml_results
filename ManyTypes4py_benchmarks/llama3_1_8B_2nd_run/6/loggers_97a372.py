        import logging
        from prefect.logging import LogEavesdropper

        with LogEavesdropper("my_logger") as eavesdropper:
            logging.getLogger("my_logger").info("Hello, world!")
            logging.getLogger("my_logger.child_module").info("Another one!")

        print(eavesdropper.text())

        # Outputs: "Hello, world!
Another one!"
    """

    def __init__(self, eavesdrop_on, level=logging.NOTSET):
        """
        Args:
            eavesdrop_on (str): the name of the logger to eavesdrop on
            level (int): the minimum log level to eavesdrop on; if omitted, all levels
                are captured
        """
        super().__init__(level=level)
        self.eavesdrop_on = eavesdrop_on
        self._target_logger = None
        self.formatter = logging.Formatter('[%(levelname)s]: %(message)s')

    def __enter__(self):
        self._target_logger = logging.getLogger(self.eavesdrop_on)
        self._original_level = self._target_logger.level
        self._target_logger.level = self.level
        self._target_logger.addHandler(self)
        self._lines = []
        return self

    def __exit__(self, *_):
        if self._target_logger:
            self._target_logger.removeHandler(self)
            self._target_logger.level = self._original_level

    def emit(self, record):
        """The logging.Handler implementation, not intended to be called directly."""
        self._lines.append(self.format(record))

    def text(self):
        """Return the collected logs as a single newline-delimited string"""
        return '\n'.join(self._lines)