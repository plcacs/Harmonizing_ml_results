class DatasetError(Exception):
    """``DatasetError`` raised by ``AbstractDataset`` implementations
    in case of failure of input/output methods.

    ``AbstractDataset`` implementations should provide instructive
    information in case of failure.
    """
    pass

class DatasetNotFoundError(DatasetError):
    """``DatasetNotFoundError`` raised by 