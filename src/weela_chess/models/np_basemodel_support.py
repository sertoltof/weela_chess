from pydantic import PlainSerializer, BeforeValidator
from typing import Annotated

import numpy as np
from numpy.typing import NDArray


def nd_array_custom_before_validator(x: list) -> NDArray:
    return np.array(x)


def nd_array_custom_serializer(x: NDArray) -> list:
    # custom serialization logic
    return x.tolist()


NumpyArray = Annotated[
    np.ndarray,
    BeforeValidator(nd_array_custom_before_validator),
    PlainSerializer(nd_array_custom_serializer, return_type=str),
]
