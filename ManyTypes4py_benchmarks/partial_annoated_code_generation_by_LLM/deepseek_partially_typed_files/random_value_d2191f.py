from random import Random
from enum import Enum
from typing import Type, Any, Dict, Tuple, Iterator, Union as TypingUnion, cast
from eth2spec.utils.ssz.ssz_typing import View, BasicView, uint, Container, List, boolean, Vector, ByteVector, ByteList, Bitlist, Bitvector, Union

UINT_BYTE_SIZES: Tuple[int, ...] = (1, 2, 4, 8, 16, 32)
random_mode_names: Tuple[str, ...] = ('random', 'zero', 'max', 'nil', 'one', 'lengthy')

class RandomizationMode(Enum):
    mode_random: int = 0
    mode_zero: int = 1
    mode_max: int = 2
    mode_nil_count: int = 3
    mode_one_count: int = 4
    mode_max_count: int = 5

    def to_name(self) -> str:
        return random_mode_names[self.value]

    def is_changing(self) -> bool:
        return self.value in [0, 4, 5]

def get_random_ssz_object(rng: Random, typ: Type[View], max_bytes_length: int, max_list_length: int, mode: RandomizationMode, chaos: bool) -> View:
    """
    Create an object for a given type, filled with random data.
    :param rng: The random number generator to use.
    :param typ: The type to instantiate
    :param max_bytes_length: the max. length for a random bytes array
    :param max_list_length: the max. length for a random list
    :param mode: how to randomize
    :param chaos: if true, the randomization-mode will be randomly changed
    :return: the random object instance, of the given type.
    """
    if chaos:
        mode = rng.choice(list(RandomizationMode))
    if issubclass(typ, ByteList):
        byte_list_typ: Type[ByteList] = cast(Type[ByteList], typ)
        if mode == RandomizationMode.mode_nil_count:
            return byte_list_typ(b'')
        elif mode == RandomizationMode.mode_max_count:
            return byte_list_typ(get_random_bytes_list(rng, min(max_bytes_length, byte_list_typ.limit())))
        elif mode == RandomizationMode.mode_one_count:
            return byte_list_typ(get_random_bytes_list(rng, min(1, byte_list_typ.limit())))
        elif mode == RandomizationMode.mode_zero:
            return byte_list_typ(b'\x00' * min(1, byte_list_typ.limit()))
        elif mode == RandomizationMode.mode_max:
            return byte_list_typ(b'\xff' * min(1, byte_list_typ.limit()))
        else:
            return byte_list_typ(get_random_bytes_list(rng, rng.randint(0, min(max_bytes_length, byte_list_typ.limit()))))
    if issubclass(typ, ByteVector):
        byte_vector_typ: Type[ByteVector] = cast(Type[ByteVector], typ)
        if mode == RandomizationMode.mode_zero:
            return byte_vector_typ(b'\x00' * byte_vector_typ.type_byte_length())
        elif mode == RandomizationMode.mode_max:
            return byte_vector_typ(b'\xff' * byte_vector_typ.type_byte_length())
        else:
            return byte_vector_typ(get_random_bytes_list(rng, byte_vector_typ.type_byte_length()))
    elif issubclass(typ, (boolean, uint)):
        if mode == RandomizationMode.mode_zero:
            return get_min_basic_value(typ)
        elif mode == RandomizationMode.mode_max:
            return get_max_basic_value(typ)
        else:
            return get_random_basic_value(rng, typ)
    elif issubclass(typ, (Vector, Bitvector)):
        vector_typ: Type[Vector] = cast(Type[Vector], typ)
        elem_type: Type[View]
        if issubclass(typ, Vector):
            elem_type = vector_typ.element_cls()
        else:
            elem_type = boolean
        return vector_typ((get_random_ssz_object(rng, elem_type, max_bytes_length, max_list_length, mode, chaos) for _ in range(vector_typ.vector_length())))
    elif issubclass(typ, List) or issubclass(typ, Bitlist):
        list_typ: Type[List] = cast(Type[List], typ)
        length: int = rng.randint(0, min(list_typ.limit(), max_list_length))
        if mode == RandomizationMode.mode_one_count:
            length = 1
        elif mode == RandomizationMode.mode_max_count:
            length = max_list_length
        elif mode == RandomizationMode.mode_nil_count:
            length = 0
        if list_typ.limit() < length:
            length = list_typ.limit()
        elem_type: Type[View]
        if issubclass(typ, List):
            elem_type = list_typ.element_cls()
        else:
            elem_type = boolean
        return list_typ((get_random_ssz_object(rng, elem_type, max_bytes_length, max_list_length, mode, chaos) for _ in range(length)))
    elif issubclass(typ, Container):
        container_typ: Type[Container] = cast(Type[Container], typ)
        fields: Dict[str, Type[View]] = container_typ.fields()
        return container_typ(**{field_name: get_random_ssz_object(rng, field_type, max_bytes_length, max_list_length, mode, chaos) for (field_name, field_type) in fields.items()})
    elif issubclass(typ, Union):
        union_typ: Type[Union] = cast(Type[Union], typ)
        options: Tuple[Type[View], ...] = union_typ.options()
        selector: int
        if mode == RandomizationMode.mode_zero:
            selector = 0
        elif mode == RandomizationMode.mode_max:
            selector = len(options) - 1
        else:
            selector = rng.randrange(0, len(options))
        elem_type: TypingUnion[Type[View], None] = options[selector]
        elem: TypingUnion[View, None]
        if elem_type is None:
            elem = None
        else:
            elem = get_random_ssz_object(rng, elem_type, max_bytes_length, max_list_length, mode, chaos)
        return union_typ(selector=selector, value=elem)
    else:
        raise Exception(f'Type not recognized: typ={typ}')

def get_random_bytes_list(rng: Random, length: int) -> bytes:
    return bytes((rng.getrandbits(8) for _ in range(length)))

def get_random_basic_value(rng: Random, typ: Type[BasicView]) -> BasicView:
    if issubclass(typ, boolean):
        boolean_typ: Type[boolean] = cast(Type[boolean], typ)
        return boolean_typ(rng.choice((True, False)))
    elif issubclass(typ, uint):
        uint_typ: Type[uint] = cast(Type[uint], typ)
        assert uint_typ.type_byte_length() in UINT_BYTE_SIZES
        return uint_typ(rng.randint(0, 256 ** uint_typ.type_byte_length() - 1))
    else:
        raise ValueError(f'Not a basic type: typ={typ}')

def get_min_basic_value(typ: Type[BasicView]) -> BasicView:
    if issubclass(typ, boolean):
        boolean_typ: Type[boolean] = cast(Type[boolean], typ)
        return boolean_typ(False)
    elif issubclass(typ, uint):
        uint_typ: Type[uint] = cast(Type[uint], typ)
        assert uint_typ.type_byte_length() in UINT_BYTE_SIZES
        return uint_typ(0)
    else:
        raise ValueError(f'Not a basic type: typ={typ}')

def get_max_basic_value(typ: Type[BasicView]) -> BasicView:
    if issubclass(typ, boolean):
        boolean_typ: Type[boolean] = cast(Type[boolean], typ)
        return boolean_typ(True)
    elif issubclass(typ, uint):
        uint_typ: Type[uint] = cast(Type[uint], typ)
        assert uint_typ.type_byte_length() in UINT_BYTE_SIZES
        return uint_typ(256 ** uint_typ.type_byte_length() - 1)
    else:
        raise ValueError(f'Not a basic type: typ={typ}')
