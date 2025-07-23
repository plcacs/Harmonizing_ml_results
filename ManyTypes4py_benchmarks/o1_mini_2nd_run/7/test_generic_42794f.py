import pytest
from mimesis import BaseProvider, Generic


class TestGeneric:

    def test_reseed(self, generic: Generic) -> None:
        generic.reseed(4095)
        number_1: float = generic.random.uniform(0, 1000)
        address_1: str = generic.address.address()
        generic.reseed(4095)
        number_2: float = generic.random.uniform(0, 1000)
        address_2: str = generic.address.address()
        assert number_1 == number_2
        assert address_1 == address_2

    def test_str(self, generic: Generic) -> None:
        assert str(generic).startswith('Generic')

    def test_base_person(self, generic: Generic) -> None:
        result: str = generic.person.username()
        assert result is not None

    def test_base_text(self, generic: Generic) -> None:
        result: str = generic.text.words()
        assert result is not None

    def test_base_payment(self, generic: Generic) -> None:
        result: str = generic.payment.bitcoin_address()
        assert result is not None

    def test_base_address(self, generic: Generic) -> None:
        result: str = generic.address.address()
        assert result is not None

    def test_base_food(self, generic: Generic) -> None:
        result: str = generic.food.fruit()
        assert result is not None

    def test_base_finance(self, generic: Generic) -> None:
        result: str = generic.finance.currency_symbol()
        assert result is not None

    def test_base_code(self, generic: Generic) -> None:
        result: str = generic.code.isbn()
        assert result is not None

    def test_base_binary_file(self, generic: Generic) -> None:
        result: bytes = generic.binaryfile.video()
        assert isinstance(result, bytes)

    def test_bad_argument(self, generic: Generic) -> None:
        with pytest.raises(AttributeError):
            _ = generic.bad_argument

    def test_add_providers(self, generic: Generic) -> None:

        class Provider1(BaseProvider):

            @staticmethod
            def one() -> int:
                return 1

        class Provider2(BaseProvider):

            class Meta:
                name: str = 'custom_provider'

            @staticmethod
            def two() -> int:
                return 2

        class Provider3(BaseProvider):

            @staticmethod
            def three() -> int:
                return 3

        class Provider4:

            @staticmethod
            def empty() -> None:
                ...

        class Provider5(BaseProvider):

            @staticmethod
            def five() -> int:
                return 5

        generic.add_providers(Provider1, Provider2, Provider3)
        assert generic.provider1.one() == 1
        assert generic.custom_provider.two() == 2
        assert generic.provider3.three() == 3
        generic += Provider5
        assert generic.provider5.five() == 5
        with pytest.raises(TypeError):
            generic.add_providers(Provider4)
        with pytest.raises(TypeError):
            generic.add_providers(3)

        class UnnamedProvider(BaseProvider):

            @staticmethod
            def nothing() -> None:
                return None

        generic.add_provider(UnnamedProvider)
        assert generic.unnamedprovider.nothing() is None

    def test_add_provider_generic_to_generic(self, generic: Generic) -> None:
        with pytest.raises(TypeError):
            generic.add_provider(Generic)

    def test_add_providers_generic_to_generic(self, generic: Generic) -> None:
        with pytest.raises(TypeError):
            generic.add_providers(Generic)

    def test_add_provider(self, generic: Generic) -> None:

        class CustomProvider(BaseProvider):

            def __init__(self, seed: int, a: str, b: str, c: str) -> None:
                super().__init__(seed=seed)
                self.a: str = a
                self.b: str = b
                self.c: str = c

            class Meta:
                name: str = 'custom_provider'

        generic.add_provider(CustomProvider, a='a', b='b', c='c', seed=4095)
        assert generic.custom_provider.seed != 4095
        assert generic.custom_provider.seed == generic.seed
        assert generic.custom_provider.a == 'a'
        assert generic.custom_provider.b == 'b'
        assert generic.custom_provider.c == 'c'

    def test_dir(self, generic: Generic) -> None:
        providers: list[str] = generic.__dir__()
        for p in providers:
            assert not p.startswith('_')


class TestSeededGeneric:

    @pytest.fixture
    def g1(self, seed: int) -> Generic:
        return Generic(seed=seed)

    @pytest.fixture
    def g2(self, seed: int) -> Generic:
        return Generic(seed=seed)

    def test_generic_address(self, g1: Generic, g2: Generic) -> None:
        assert g1.address.street_number() == g2.address.street_number()
        assert g1.address.street_name() == g2.address.street_name()

    def test_generic_finance(self, g1: Generic, g2: Generic) -> None:
        assert g1.finance.company() == g2.finance.company()

    def test_generic_code(self, g1: Generic, g2: Generic) -> None:
        assert g1.code.locale_code() == g2.code.locale_code()
        assert g1.code.issn() == g2.code.issn()

    def test_generic_cryptographic(self, g1: Generic, g2: Generic) -> None:
        assert g1.cryptographic.uuid() != g2.cryptographic.uuid()
        assert g1.cryptographic.hash() != g2.cryptographic.hash()

    def test_generic_datetime(self, g1: Generic, g2: Generic) -> None:
        assert g1.datetime.week_date() == g2.datetime.week_date()
        assert g1.datetime.day_of_week() == g2.datetime.day_of_week()

    def test_generic_development(self, g1: Generic, g2: Generic) -> None:
        sl1: str = g1.development.software_license()
        sl2: str = g2.development.software_license()
        assert sl1 == sl2

    def test_generic_file(self, g1: Generic, g2: Generic) -> None:
        assert g1.file.size() == g2.file.size()
        assert g1.file.file_name() == g2.file.file_name()

    def test_generic_food(self, g1: Generic, g2: Generic) -> None:
        assert g1.food.dish() == g2.food.dish()
        assert g1.food.spices() == g2.food.spices()

    def test_generic_hardware(self, g1: Generic, g2: Generic) -> None:
        assert g1.hardware.screen_size() == g2.hardware.screen_size()
        assert g1.hardware.cpu() == g2.hardware.cpu()

    def test_generic_internet(self, g1: Generic, g2: Generic) -> None:
        assert g1.internet.content_type() == g2.internet.content_type()

    def test_generic_numbers(self, g1: Generic, g2: Generic) -> None:
        assert g1.numeric.integers() == g2.numeric.integers()

    def test_generic_path(self, g1: Generic, g2: Generic) -> None:
        assert g1.path.root() == g2.path.root()
        assert g1.path.home() == g2.path.home()

    def test_generic_payment(self, g1: Generic, g2: Generic) -> None:
        assert g1.payment.cid() == g2.payment.cid()
        assert g1.payment.paypal() == g2.payment.paypal()

    def test_generic_person(self, g1: Generic, g2: Generic) -> None:
        assert g1.person.birthdate() == g2.person.birthdate()
        assert g1.person.name() == g2.person.name()

    def test_generic_science(self, g1: Generic, g2: Generic) -> None:
        assert g1.science.rna_sequence() == g2.science.rna_sequence()

    def test_generic_transport(self, g1: Generic, g2: Generic) -> None:
        assert g1.transport.airplane() == g2.transport.airplane()
