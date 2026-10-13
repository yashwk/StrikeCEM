"""StrikeCEM Python SDK: Real-Time Radar Cross Section (RCS) Query Engine."""

from .bindings import find_library
from .database import Database, QueryResult, QueryRequest, InterpolationMode
from .errors import (
    StrikeCemError,
    InvalidArgumentError,
    DatabaseOpenError,
    DatabaseCorruptError,
    DatabaseIncompleteError,
    DesignMismatchError,
    ChannelNotFoundError,
    InternalError,
)

__version__ = "0.1.0"
__all__ = [
    "Database",
    "QueryResult",
    "QueryRequest",
    "InterpolationMode",
    "find_library",
    "StrikeCemError",
    "InvalidArgumentError",
    "DatabaseOpenError",
    "DatabaseCorruptError",
    "DatabaseIncompleteError",
    "DesignMismatchError",
    "ChannelNotFoundError",
    "InternalError",
]
