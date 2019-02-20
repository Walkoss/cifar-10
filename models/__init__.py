from abc import ABC, abstractmethod


class BaseModel(ABC):
    @classmethod
    def variant(cls, variant_name: str = "default", *args, **kwargs):
        variant_method_name = "variant_{}".format(variant_name)
        if not hasattr(cls, variant_method_name):
            raise TypeError(
                "No variant named {} for {}".format(variant_name, cls.__name__)
            )

        variant = getattr(cls, variant_method_name)
        if not callable(variant):
            raise RuntimeError("Variant must be a callable")

        return variant

    @classmethod
    @abstractmethod
    def variant_default(cls, *args, **kwargs):
        pass
