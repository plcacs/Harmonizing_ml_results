from random import Random
from enum import Enum
from typing import Type, Any, Iterator, Dict
from eth2spec.utils.ssz.ssz_typing import (
    View,
    BasicView,
    uint,
    Container,
    List,
    boolean,
    Vector,
    ByteVector,
    ByteList,
    Bitlist,
    Bitvector,
    Union,
)

UINT_BYTE_SIZES = (1, 2, 4, 8, 16, 32)
random_mode_names = ("random", "zero", "max", "nil", "one", "lengthy")


class RandomizationMode(Enum):
    mode_random = 0
    mode_zero = 1
    mode_max = 2
    mode_nil_count = 3
    mode_one_count = 4
    mode_max_count = 5

    def to_name(self) -> str:
        return random_mode_names[self.value]

    def is_changing(self) -> bool:
        return self.value in [0, 4, 5]


def get_random_ssz_object(
    rng: Random,
    typ: Type[View],
    max_bytes_length: int,
    max_list_length: int,
    mode: RandomizationMode,
    chaos: bool,
) -> View:
    """
    Create an object for a given type, filled with random data.
    :param rng: The random number generator to use.
    :param typ: The type to instantiate.
    :param max_bytes_length: The max. length for a random bytes array.
    :param max_list_length: The max. length for a random list.
    :param mode: How to randomize.
    :param chaos: If true, the randomization-mode will be randomly changed.
    :return: The random object instance, of the given type.
    """
    if chaos:
        mode = rng.choice(list(RandomizationMode))
    if issubclass(typ, ByteList):
        if mode == RandomizationMode.mode_nil_count:
            return typ(b"")
        elif mode == RandomizationMode.mode_max_count:
            return typ(get_random_bytes_list(rng, min(max_bytes_length, typ.limit())))
        elif mode == RandomizationMode.mode_one_count:
            return typ(get_random_bytes_list(rng, min(1, typ.limit())))
        elif mode == RandomizationMode.mode_zero:
            return typ(b"\x00" * min(1, typ.limit()))
        elif mode == RandomizationMode.mode_max:
            return typ(b"\xff" * min(1, typ.limit()))
        else:
            return typ(
                get_random_bytes_list(rng, rng.randint(0, min(max_bytes_length, typ.limit())))
            )
    if issubclass(typ, ByteVector):
        if mode == RandomizationMode.mode_zero:
            return typ(b"\x00" * typ.type_byte_length())
        elif mode == RandomizationMode.mode_max:
            return typ(b"\xff" * typ.type_byte_length())
        else:
            return typ(get_random_bytes_list(rng, typ.type_byte_length()))
    elif issubclass(typ, (boolean, uint)):
        if mode == RandomizationMode.mode_zero:
            return get_min_basic_value(typ)
        elif mode == RandomizationMode.mode_max:
            return get_max_basic_value(typ)
        else:
            return get_random_basic_value(rng, typ)
    elif issubclass(typ, (Vector, Bitvector)):
        elem_type = typ.element_cls() if issubclass(typ, Vector) else boolean
        return typ(
            (
                get_random_ssz_object(rng, elem_type, max_bytes_length, max_list_length, mode, chaos)
                for _ in range(typ.vector_length())
            )
        )
    elif issubclass(typ, List) or issubclass(typ, Bitlist):
        length: int = rng.randint(0, min(typ.limit(), max_list_length))
        if mode == RandomizationMode.mode_one_count:
            length = 1
        elif mode == RandomizationMode.mode_max_count:
            length = max_list_length
        elif mode == RandomizationMode.mode_nil_count:
            length = 0
        if typ.limit() < length:
            length = typ.limit()
        elem_type = typ.element_cls() if issubclass(typ, List) else boolean
        return typ(
            (
                get_random_ssz_object(rng, elem_type, max_bytes_length, max_list_length, mode, chaos)
                for _ in range(length)
            )
        )
    elif issubclass(typ, Container):
        fields: Dict[str, Any] = typ.fields()
        return typ(
            **{
                field_name: get_random_ssz_object(rng, field_type, max_bytes_length, max_list_length, mode, chaos)
                for field_name, field_type in fields.items()
            }
        )
    elif issubclass(typ, Union):
        options = typ.options()
        if mode == RandomizationMode.mode_zero:
            selector: int = 0
        elif mode == RandomizationMode.mode_max:
            selector = len(options) - 1
        else:
            selector = rng.randrange(0, len(options))
        elem_type = options[selector]
        if elem_type is None:
            elem = None
        else:
            elem = get_random_ssz_object(rng, elem_type, max_bytes_length, max_list_length, mode, chaos)
        return typ(selector=selector, value=elem)
    else:
        raise Exception(f"Type not recognized: typ={typ}")


def get_random_bytes_list(rng: Random, length: int) -> bytes:
    return bytes((rng.getrandbits(8) for _ in range(length)))


def get_random_basic_value(rng: Random, typ: Type) -> Any:
    if issubclass(typ, boolean):
        return typ(rng.choice((True, False)))
    elif issubclass(typ, uint):
        assert typ.type_byte_length() in UINT_BYTE_SIZES
        return typ(rng.randint(0, 256 ** typ.type_byte_length() - 1))
    else:
        raise ValueError(f"Not a basic type: typ={typ}")


def get_min_basic_value(typ: Type) -> Any:
    if issubclass(typ, boolean):
        return typ(False)
    elif issubclass(typ, uint):
        assert typ.type_byte_length() in UINT_BYTE_SIZES
        return typ(0)
    else:
        raise ValueError(f"Not a basic type: typ={typ}")


def get_max_basic_value(typ: Type) -> Any:
    if issubclass(typ, boolean):
        return typ(True)
    elif issubclass(typ, uint):
        assert typ.type_byte_length() in UINT_BYTE_SIZES
        return typ(256 ** typ.type_byte_length() - 1)
    else:
        raise ValueError(f"Not a basic type: typ={typ}")