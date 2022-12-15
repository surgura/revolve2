"""Populations for evolutionary algorithms."""

from ._multiple_unique import multiple_unique
from ._pop_list import PopList
from ._replace_if_better import replace_if_better
from ._topn import topn
from ._tournament import tournament
from ._replace_if import replace_if

__all__ = [
    "PopList",
    "multiple_unique",
    "replace_if_better",
    "topn",
    "tournament",
    "replace_if",
]
