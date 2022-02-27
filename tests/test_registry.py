from pydoc import resolve
from configuration.registry import Registry
from configuration.config import Config
import catalogue


class Reg(Registry):

    misc = catalogue.create("configurations", "misc")
    other = catalogue.create("configurations", "other")


registry = Reg()


@registry.misc("func")
def misc(n):
    def f():
        return n

    return f


cfg = """
[test]
@misc = "func"
n = 5
"""


def test_registry():
    config = Config().from_str(cfg)
    resolved = registry.resolve(config)

    assert resolved["test"]() == 5
