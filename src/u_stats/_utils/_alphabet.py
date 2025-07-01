import string

_English_alphabet = string.ascii_lowercase + string.ascii_uppercase


# similar to opt_einsum
class Alphabet:
    _instance = None
    _initialized = False

    def __new__(cls) -> "Alphabet":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not Alphabet._initialized:
            self._alphabet = _English_alphabet
            Alphabet._initialized = True

    def __getitem__(self, i: int) -> str:
        if i < 52:
            return self._alphabet[i]
        elif i >= 55296:
            return chr(i + 2048)
        else:
            return chr(i + 140)


ALPHABET: "Alphabet" = Alphabet()
