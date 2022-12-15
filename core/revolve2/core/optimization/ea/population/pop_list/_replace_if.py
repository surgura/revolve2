from typing import TypeVar, Callable, List

from revolve2.core.database import Serializable

from .._serializable_measures import SerializableMeasures
from ._pop_list import PopList

TIndividual = TypeVar("TIndividual", bound=Serializable)
TMeasures = TypeVar("TMeasures", bound=SerializableMeasures)


def replace_if(
    original_population: PopList[TIndividual, TMeasures],
    offspring_population: PopList[TIndividual, TMeasures],
    condition: Callable[[TMeasures, TMeasures], bool],
) -> List[int]:
    """
    Compare each individual is offspring population with original population index-wise and replaces if better.

    Populations must be of the same size.

    :param original_population: The original population to replace individuals in. Will not be altered.
    :param offspring_population: The offspring population to take individuals from. Will not be unaltered. Individuals will be copied.
    :param condition: Replace if second argument is better than first argument.
    :returns: For each index in the population, the 0 to take the individual from the original population, or 1 to take it from the offspring.
    """
    return [
        1 if condition(orig.measures, off.measures) else 0
        for orig, off in zip(original_population, offspring_population)
    ]
