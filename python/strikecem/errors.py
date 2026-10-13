class StrikeCemError(Exception):
    """Base exception for all StrikeCEM errors."""
    pass

class InvalidArgumentError(StrikeCemError):
    """Raised when an invalid argument is passed to the StrikeCEM API."""
    pass

class DatabaseOpenError(StrikeCemError):
    """Raised when a database cannot be opened or file is missing."""
    pass

class DatabaseCorruptError(StrikeCemError):
    """Raised when a database file is corrupt or has invalid headers/checksums."""
    pass

class DatabaseIncompleteError(StrikeCemError):
    """Raised when a database is incomplete or missing chunks."""
    pass

class DesignMismatchError(StrikeCemError):
    """Raised when design ID or revision does not match the expected values."""
    pass

class ChannelNotFoundError(StrikeCemError):
    """Raised when a requested frequency or polarization channel is not found."""
    pass

class InternalError(StrikeCemError):
    """Raised when an internal solver or query engine error occurs."""
    pass

STATUS_MAP = {
    1: InvalidArgumentError,
    2: DatabaseOpenError,
    3: DatabaseCorruptError,
    4: DatabaseIncompleteError,
    5: DesignMismatchError,
    6: ChannelNotFoundError,
    7: InternalError,
}

def raise_for_status(status_code: int, default_msg: str = ""):
    if status_code == 0:
        return
    exc_cls = STATUS_MAP.get(status_code, StrikeCemError)
    msg = default_msg or f"StrikeCEM API error code {status_code}"
    raise exc_cls(msg)
