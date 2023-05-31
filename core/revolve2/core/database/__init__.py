"""Classes and interfaces for working with databases."""

from ._incompatible_error import IncompatibleError
from ._serializer import Serializer
from ._sqlite import open_async_database_sqlite, open_database_sqlite
from ._has_id import HasId

__all__ = [
    "IncompatibleError",
    "Serializer",
    "open_async_database_sqlite",
    "open_database_sqlite",
    "HasId",
]
